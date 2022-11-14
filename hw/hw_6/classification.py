import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset, Dataset, DataLoader

from sklearn.model_selection import train_test_split

from PIL import Image

import numpy as np
import os

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from tqdm import tqdm

import gc
torch.cuda.empty_cache()
gc.collect()


class BirdsDataset(Dataset):
    def __init__(self, gt, img_dir, *, train=True, transform):
        self._items = []

        train_gt, val_gt = train_test_split(list(gt.items()), test_size=0.3, shuffle=True, random_state=0)
        gt = train_gt if train else val_gt

        for img_filename, class_id in gt:
            img_path = os.path.join(img_dir, img_filename)
            self._items.append((img_path, class_id + 1))

        self._transform = transform

        self.classes = np.sort(np.unique([class_id for _, class_id in gt]))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, class_id = self._items[index]
        img = Image.open(img_path).convert('RGB')

        if self._transform:
            img = self._transform(img)

        return img, class_id


class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = torch.nn.Linear(1280, num_classes)

        for child in list(self.model.children())[:-4]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)


def train_epoch(model, optimizer, criterion, train_loader, device, tqdm_desc):
    model.train()
    train_acc, train_loss = 0.0, 0.0

    for images, class_ids in tqdm(train_loader, desc=tqdm_desc):
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
def val_epoch(model, criterion, val_loader, device, tqdm_desc):
    model.eval()
    val_acc, val_loss = 0.0, 0.0

    for images, class_ids in tqdm(val_loader, desc=tqdm_desc):
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
    for epoch in range(1, num_epochs + 1):
        train_acc, train_loss = train_epoch(model, optimizer, criterion, train_loader, device, f'training {epoch}/{num_epochs}')
        val_acc, val_loss = val_epoch(model, criterion, val_loader, device, f'validating {epoch}/{num_epochs}')

        if scheduler is not None:
            scheduler.step()

        print(f"train_acc {train_acc} train_loss {train_loss}")
        print(f"val_acc {val_acc} val_loss {val_loss}")


def train_classifier(train_gt, train_img_dir, fast_train=True):
    batch_size = 64
    num_epochs = 6

    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = BirdsDataset(train_gt, train_img_dir, train=True, transform=train_transform)
    val_dataset = BirdsDataset(train_gt, train_img_dir, train=False, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MobileNet(len(train_dataset.classes)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = None #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    train(model, optimizer, criterion, scheduler, train_loader, val_loader, device, num_epochs)
    torch.save(model.state_dict(), "model.pt")
    return model


def classify(model_path, test_img_dir):
    pass


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

    model = train_classifier(train_gt, train_img_dir, fast_train=True)
