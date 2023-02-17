import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from PIL import Image

#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from IPython.display import clear_output

from torchvision.models import mobilenet_v2

#from tqdm import tqdm

import os


class BirdsDataset(Dataset):
    def __init__(self, gt, img_dir, *, train=True, val_size=0.3, transform):
        self._items = []

        train_gt = list(gt.items())
        val_gt = []

        if val_size > 0:
            train_gt, val_gt = train_test_split(list(gt.items()), test_size=val_size, shuffle=True, random_state=0)

        gt = train_gt if train else val_gt

        for img_filename, class_id in gt:
            img_path = os.path.join(img_dir, img_filename)
            self._items.append((img_path, class_id))

        self._transform = transform

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, class_id = self._items[index]
        img = Image.open(img_path).convert('RGB')

        if self._transform:
            img = self._transform(img)

        return img, class_id


class BirdsTestDataset(Dataset):
    def __init__(self, img_dir, *, transform):
        self._items = []

        for img_filename in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_filename)
            self._items.append((img_path, img_filename))

        self._transform = transform

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, img_filename = self._items[index]
        img = Image.open(img_path).convert('RGB')

        if self._transform:
            img = self._transform(img)

        return img, img_filename


class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = mobilenet_v2(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
            #nn.Linear(512, 256),
            #nn.BatchNorm1d(256),
            #nn.ReLU(),
            #nn.Linear(256, num_classes)
        )

        for child in list(self.model.children())[:-9]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

"""
def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 15})
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(test_losses) + 1), test_losses, label='test')
    axs[0].set_ylabel('loss')

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='train')
    axs[1].plot(range(1, len(test_accuracies) + 1), test_accuracies, label='test')
    axs[1].set_ylabel('accuracy')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()
"""


def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    train_acc, train_loss = 0.0, 0.0

    for images, class_ids in train_loader:
        images = images.to(device)
        class_ids = class_ids.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, class_ids)

        loss.backward()
        optimizer.step()

        train_acc += (logits.argmax(dim=1) == class_ids).sum().item()
        train_loss += loss.item() * class_ids.shape[0]

    train_acc /= len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)

    return train_acc, train_loss


@torch.no_grad()
def val_epoch(model, criterion, val_loader, device):
    model.eval()
    val_acc, val_loss = 0.0, 0.0

    for images, class_ids in val_loader:
        images = images.to(device)
        class_ids = class_ids.to(device)

        logits = model(images)
        loss = criterion(logits, class_ids)

        val_acc += (logits.argmax(dim=1) == class_ids).sum().item()
        val_loss += loss.item() * class_ids.shape[0]

    val_acc /= len(val_loader.dataset)
    val_loss /= len(val_loader.dataset)

    return val_acc, val_loss


def train(model, optimizer, criterion, scheduler, train_loader, val_loader, device, num_epochs):
    train_accs, train_losses = [], []
    val_accs, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        train_acc, train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        val_acc, val_loss = val_epoch(model, criterion, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        train_accs += [train_acc]
        train_losses += [train_loss]
        val_accs += [val_acc]
        val_losses += [val_loss]

        #plot_losses(train_losses, val_losses, train_accs, val_accs)


def train_all(model, optimizer, criterion, scheduler, train_loader, device, num_epochs):
    train_accs, train_losses = [], []

    for epoch in range(1, num_epochs + 1):
        train_acc, train_loss = train_epoch(model, optimizer, criterion, train_loader, device)

        if scheduler is not None:
            scheduler.step()

        train_accs += [train_acc]
        train_losses += [train_loss]

        # print(f"train acc {train_acc} train loss {train_loss}")


def train_classifier(train_gt, train_img_dir, fast_train=True):
    batch_size = 8
    num_epochs = 15

    img_size = 224

    if fast_train:
        num_epochs = 1

    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomRotation(degrees=30)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    """
    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    """

    train_dataset = BirdsDataset(train_gt, train_img_dir, train=True, val_size=0, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = 50
    model = MobileNet(num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    scheduler = None

    train_all(model, optimizer, criterion, scheduler, train_loader, device, num_epochs)
    #torch.save(model.state_dict(), "birds_model.ckpt")

    return model


def classify(model_path, test_img_dir):
    batch_size = 1
    img_size = 224

    test_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_dataset = BirdsTestDataset(test_img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = 50
    model = MobileNet(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()

    res_dict = {}

    for img, img_path in test_loader:
        logits = model(img.to(device)).cpu().detach().numpy().ravel()
        class_id = logits.argmax()
        res_dict[img_path[0]] = class_id

    return res_dict


"""
if __name__ == "__main__":
    from os.path import join

    def read_csv(filename):
        res = {}
        with open(filename) as fhandle:
            next(fhandle)
            for line in fhandle:
                filename, class_id = line.rstrip('\n').split(',')
                res[filename] = int(class_id)
        return res

    data_dir = "tests/00_test_img_input"

    train_dir = join(data_dir, 'train')
    train_gt = read_csv(join(train_dir, 'gt.csv'))
    train_img_dir = join(train_dir, 'images')
    model = train_classifier(train_gt, train_img_dir, fast_train=False)

    #test_dir = join(data_dir, 'test')
    #model_path = 'birds_model.ckpt'
    #test_img_dir = join(test_dir, 'images')
    #img_classes = classify(model_path, test_img_dir)
"""