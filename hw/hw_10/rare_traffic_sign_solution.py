# -*- coding: utf-8 -*-
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torch import nn

import os
import csv
import json
from tqdm import tqdm
import pickle
import typing
import cv2

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier

# import seaborn as sns
import matplotlib.pyplot as plt
# from IPython.display import clear_output
import torch.nn.functional as F


CLASSES_CNT = 205
CUDA = 0


class DatasetRTSD(Dataset):
    """
    Класс для чтения и хранения датасета.
    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """
    def __init__(self, root_folders, path_to_classes_json) -> None:
        super(DatasetRTSD, self).__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        self.samples = []  # список пар (путь до картинки, индекс класса)
        self.classes_to_samples = {}  # словарь из списков картинок для каждого класса
        self.transform = A.Compose([
            A.Resize(224, 224),
            # A.CenterCrop(224),
            A.HorizontalFlip(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

        for class_name in self.classes:
            self.classes_to_samples[self.class_to_idx[class_name]] = []

        for root in root_folders:
            for class_dir in os.listdir(root):
                for image in os.listdir(os.path.join(root, class_dir)):
                    self.samples.append((os.path.join(root, class_dir, image), self.class_to_idx[class_dir]))
                    self.classes_to_samples[self.class_to_idx[class_dir]].append(len(self.samples) - 1)

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        image_path, class_idx = self.samples[index]
        img = np.array(Image.open(image_path).convert('RGB'))
        img = self.transform(image=img)
        return img, image_path, class_idx

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        :param path_to_classes_json: путь до classes.json
        """
        class_to_idx = {}
        classes = []
        fd = open(path_to_classes_json, 'r')
        for key, item in json.load(fd).items():
            class_to_idx[key] = item['id']
            classes.append(key)
        return classes, class_to_idx


class TestData(Dataset):
    """
    Класс для чтения и хранения тестового датасета.
    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    def __init__(self, root, path_to_classes_json, annotations_file=None):
        super(TestData, self).__init__()
        self.classes, self.class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
        self.root = root
        self.samples = []  # список путей до картинок

        for image in os.listdir(self.root):
            self.samples.append(image)

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

        self.targets = None  # targets[путь до картинки] = индекс класса
        if annotations_file is not None:
            self.targets = {}
            with open(annotations_file, newline='') as csvfile:
                rd = csv.reader(csvfile, delimiter=',')
                next(rd)
                for row in rd:
                    self.targets[row[0]] = self.class_to_idx[row[1]]

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        image_path = self.samples[index]
        img = np.array(Image.open(os.path.join(self.root, image_path)).convert('RGB'))
        img = self.transform(image=img)
        if self.targets:
            return img, image_path, self.targets.get(image_path, -1)
        else:
            return img, image_path, -1

    def __len__(self):
        return len(self.samples)


class CustomNetwork(torch.nn.Module):
    """
    Класс, реализующий нейросеть для классификации.
    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """
    def __init__(self, features_criterion=None, internal_features=1024):
        super().__init__()

        self.criterion = features_criterion
        self.model = resnet50(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, internal_features),
            nn.ReLU(),
            nn.Linear(internal_features, CLASSES_CNT)
        )
        for child in list(self.model.children()):
            for param in child.parameters():
                param.requires_grad = False

        device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param x: батч с картинками
        """
        logits = self.model(x)
        return logits.argmax(dim=1)


def training_epoch(model, optimizer, criterion, train_loader, tqdm_desc, device):
    train_loss, train_accuracy = 0.0, 0.0
    model.train()
    for images, paths, labels in tqdm(train_loader, desc=tqdm_desc):
        images = images['image']
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.shape[0]
        train_accuracy += (logits.argmax(dim=1) == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    return train_loss, train_accuracy


def full_train(model, optimizer, scheduler, criterion, train_loader, test_loader, num_epochs, device):
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}',
            device=device
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        train_accuracies += [train_accuracy]
        # plot_losses(train_losses, test_losses, train_accuracies, test_accuracies)

    return train_losses, test_losses, train_accuracies, test_accuracies


def train_simple_classifier():
    """Функция для обучения простого классификатора на исходных данных."""
    print(f'using CUDA:{CUDA}')
    num_epochs = 3

    train_dataset = DatasetRTSD(root_folders=['./additonal_files/cropped-train'],
                                path_to_classes_json='./additonal_files/classes.json')

    test_dataset = TestData(root='./additonal_files/smalltest',
                            path_to_classes_json='./additonal_files/classes.json',
                            annotations_file='./additonal_files/smalltest_annotations.csv')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
    model = CustomNetwork()
    lr = 1e-3
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    full_train(model, optimizer, scheduler, criterion, train_loader, test_loader, num_epochs, device=device)
    torch.save(model.state_dict(), "simple_model.pth")
    return model


def apply_classifier(model, test_folder, path_to_classes_json):
    """
    Функция, которая применяет модель и получает её предсказания.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    test_dataset = TestData(root=test_folder,
                            path_to_classes_json=path_to_classes_json)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
    results = []
    model.eval()
    for img, img_path, img_class in test_loader:
        sign_class = model.predict(img['image'].to('cpu')).cpu().detach().numpy().ravel().item()
        results.append({
            'filename': img_path[0],
            'class': test_dataset.classes[sign_class]
        })
    return results


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        reader = csv.DictReader(fhandle)
        for row in reader:
            res[row['filename']] = row['class']
    return res


def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == 'all' or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)


def test_classifier(model, test_folder, path_to_classes_json, annotations_file):
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    output = apply_classifier(model, test_folder, path_to_classes_json)
    output = {elem['filename']: elem['class'] for elem in output}
    gt = read_csv(annotations_file)
    y_pred = []
    y_true = []

    for k, v in output.items():
        y_pred.append(v)
        y_true.append(gt[k])

    with open(path_to_classes_json, "r") as fr:
        classes_info = json.load(fr)
    class_name_to_type = {k: v['type'] for k, v in classes_info.items()}

    total_acc = calc_metric(y_true, y_pred, 'all', class_name_to_type)
    rare_recall = calc_metric(y_true, y_pred, 'rare', class_name_to_type)
    freq_recall = calc_metric(y_true, y_pred, 'freq', class_name_to_type)
    return total_acc, rare_recall, freq_recall


def motion_blur(img, random_angle, kernel_size=5):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel /= kernel_size
    kernel = A.rotate(kernel, angle=random_angle)
    img = cv2.filter2D(img, -1, kernel)
    return img


def generate_sign(icon_path):
    img = Image.open(icon_path)
    if img.mode == 'LA':
        mask_channel = 1
    else:
        mask_channel = 3
    sign_mask = np.array(img)[..., mask_channel]
    img = np.array(img.convert('RGB'))

    random_size = np.random.choice(np.arange(16, 129))
    random_pad_percent = np.random.choice(np.arange(0, 16)) / 100

    transform1 = A.Compose([
        A.Resize(random_size, random_size),
        A.CropAndPad(percent=random_pad_percent),
        A.ColorJitter(hue=0.2, contrast=(1, 1), p=1),
        A.Rotate(limit=(-15, 15))
    ])

    gaussian_kernel_size = max(3, (1 - int(random_size * 0.1) % 2) + int(random_size * 0.1))
    transform2 = A.Compose([
        A.GaussianBlur(blur_limit=(1, gaussian_kernel_size), p=0.3)
    ])
    transformed = transform1(image=img, mask=sign_mask)
    # transformed = transform2(image=transformed['image'], mask=transformed['mask'])
    sign_mask = transformed['mask']
    img = transform2(image=transformed['image'])['image']

    random_angle = np.random.choice(np.arange(-90, 90))
    img = motion_blur(img, random_angle, random_size // 15)

    # plt.axis('off')
    # plt.imshow(img * np.dstack((sign_mask, sign_mask, sign_mask)).astype('bool'))
    return img, sign_mask.astype('bool')


def fuse_with_background(sign_img, sign_mask, background_path):
    background_img = np.array(Image.open(background_path).convert('RGB'))
    crop_size = int(sign_img.shape[0] * 1.05)
    background_img = A.RandomCrop(crop_size, crop_size)(image=background_img)['image']

    offset = (background_img.shape[0] - sign_img.shape[0]) // 2
    y1, y2 = offset, offset + sign_img.shape[0]
    x1, x2 = offset, offset + sign_img.shape[0]

    background_mask = ~sign_mask
    for c in range(0, 3):
        background_img[y1:y2, x1:x2, c] = (sign_mask * sign_img[:, :, c] +
                                           background_mask * background_img[y1:y2, x1:x2, c])
    return background_img


def generate_one_icon(args):
    """
    Функция, генерирующая синтетические данные для одного класса.
    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    icon_path, output_folder, backgrounds_folder_path, sample_quantity = args

    icon_class = os.path.split(icon_path)[1][:-4]
    icon_output_folder = os.path.join(output_folder, icon_class)
    if not os.path.exists(icon_output_folder):
        os.makedirs(icon_output_folder)

    backgrounds_files = os.listdir(backgrounds_folder_path)
    for i in range(sample_quantity):
        sign_img, sign_mask = generate_sign(icon_path)
        bg_number = np.random.choice(len(backgrounds_files))
        plt.imsave(os.path.join(icon_output_folder, f'{icon_class}_{i}.png'),
                   fuse_with_background(sign_img, sign_mask, os.path.join(backgrounds_folder_path, backgrounds_files[bg_number])))


def generate_all_data(output_folder, icons_path, background_path, samples_per_class=1000):
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.
    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    with ProcessPoolExecutor(8) as executor:
        params = [[os.path.join(icons_path, icon_file), output_folder, background_path, samples_per_class]
                  for icon_file in os.listdir(icons_path)]
        list(tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier():
    """Функция для обучения простого классификатора на смеси исходных и ситетических данных."""

    print(f'using CUDA:{CUDA}')
    num_epochs = 1

    train_dataset = DatasetRTSD(root_folders=['./additonal_files/cropped-train', './synthetic_samples'],
                                path_to_classes_json='./additonal_files/classes.json')

    # test_dataset = TestData(root='./additonal_files/smalltest',
    #                        path_to_classes_json='./additonal_files/classes.json',
    #                        annotations_file='./additonal_files/smalltest_annotations.csv')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
    model = CustomNetwork()
    lr = 1e-3
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    full_train(model, optimizer, scheduler, criterion, train_loader, None, num_epochs, device=device)
    torch.save(model.state_dict(), "simple_model_with_synt.pth")
    return model


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """
    def __init__(self, margin):
        super(FeaturesLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, outputs, labels):
        output1, output2 = outputs
        label1, label2 = labels
        distances = (output2 - output1).square().sum(1)
        distances_neg = F.relu(self.margin - (distances + self.eps).sqrt()).square()
        losses = 0.5 * torch.where(label1 == label2, distances, distances_neg)
        return losses.mean()


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.
    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """
    def __init__(self, data_source, elems_per_class, classes_per_batch):
        self.training_data = data_source.samples
        self.class_count = len(data_source.classes)
        self.classes_to_samples = data_source.classes_to_samples
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch
        self.batch_size = elems_per_class * classes_per_batch

        # self.training_label = data_source.train_labels.to(device)
        # self.training_data = self.training_data.type(torch.cuda.FloatTensor)

    def __iter__(self):
        samples = []
        for clas_idx in np.random.choice(self.class_count, size=self.classes_per_batch, replace=False):
            samples_idx = np.random.choice(self.classes_to_samples[clas_idx],
                                           size=self.elems_per_class)
            samples += list(samples_idx)
        yield samples

    def __len__(self):
        return self.batch_size


class ModelWithHead:
    """
    Класс, реализующий модель с головой из kNN.
    :param n_neighbors: Количество соседей в методе ближайших соседей
    """
    def __init__(self, n_neighbors):
        ### YOUR CODE HERE
        pass

    def load_nn(self, nn_weights_path):
        """
        Функция, загружающая веса обученной нейросети.
        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        ### YOUR CODE HERE
        pass

    def load_head(self, knn_path):
        """
        Функция, загружающая веса kNN (с помощью pickle).
        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        ### YOUR CODE HERE
        pass

    def predict(self, imgs):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param imgs: батч с картинками
        """
        # features, model_pred = ### YOUR CODE HERE - предсказание нейросетевой модели
        # features = features / np.linalg.norm(features, axis=1)[:, None]
        # knn_pred = ... ### YOUR CODE HERE - предсказание kNN на features
        # return knn_pred
        pass


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.
    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    def __init__(self, data_source, examples_per_class) -> None:
        self.training_data = data_source.samples
        self.class_count = len(data_source.classes)
        self.classes_to_samples = data_source.classes_to_samples
        self.examples_per_class = examples_per_class
        self.batch_size = examples_per_class * self.class_count

    def __iter__(self):
        samples = []
        for clas_idx in range(self.class_count):
            samples_idx = np.random.choice(self.classes_to_samples[clas_idx],
                                           size=self.examples_per_class)
            samples += list(samples_idx)
        return iter(samples)

    def __len__(self):
        return self.batch_size


def train_head(nn_weights_path, examples_per_class = 20):
    """
    Функция для обучения kNN-головы классификатора.
    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    ### YOUR CODE HERE
    pass
