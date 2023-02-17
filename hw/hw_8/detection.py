import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import numpy as np
import os


class FacesDataset(Dataset):
    def __init__(self, train_gt, train_img_dir, img_size, transform=None):
        self._items = []

        for img_filename, img_points in train_gt.items():
            img_path = os.path.join(train_img_dir, img_filename)
            self._items.append((img_path, np.array(img_points)))

        self._img_size = img_size
        self._transform = transform

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, img_points = self._items[index]
        img = Image.open(img_path).convert('RGB')

        #if np.random.randint(0, 1) == 1:
        #    img = transforms.functional.hflip(img)
        #    img_points = img_points.copy()
        #    img_points[::2] = self._img_size - img_points[::2]

        img_points[0::2] = (self._img_size / img.size[0]) * img_points[0::2]
        img_points[1::2] = (self._img_size / img.size[1]) * img_points[1::2]
        img = transforms.Resize((self._img_size, self._img_size))(img)

        if self._transform:
            img = self._transform(img)

        return img, img_points


class FacesTestDataset(Dataset):
    def __init__(self, test_img_dir, img_size, transform=None):
        self._items = []

        for img_filename in os.listdir(test_img_dir):
            img_path = os.path.join(test_img_dir, img_filename)
            self._items.append(img_path)

        self._img_size = img_size
        self._transform = transform

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path = self._items[index]
        img = Image.open(img_path).convert('RGB')
        size = img.size

        img = transforms.Resize((self._img_size, self._img_size))(img)

        if self._transform:
            img = self._transform(img)

        return img, os.path.basename(img_path), size


class FacesModel(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=(img_size // 8)**2 * 256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=28)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x.view(x.size(0), -1))
        return x


def train_epoch(dataloader, model, optimizer, criterion, device):
    loss_log = []

    for img, img_points in dataloader:
        img = img.to(device)
        img_points = img_points.to(device).float()

        optimizer.zero_grad()

        out = model(img)
        loss = criterion(out, img_points)

        loss_log.append(loss.item())

        loss.backward()
        optimizer.step()

    return np.mean(loss_log)


def train_detector(train_gt, train_img_dir, fast_train=True):
    batch_size = 32 # может поменять потом
    img_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = FacesDataset(train_gt, train_img_dir, img_size=img_size, transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FacesModel(img_size).to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    n_epochs = 25

    if fast_train:
        n_epochs = 1

    model.train()

    for epoch in range(n_epochs):
        print(f"Epoch #{epoch}")
        loss = train_epoch(dataloader, model, optimizer, criterion, device)
        print(f"loss {loss}")

    return model


def detect(model_filename, test_img_dir):
    batch_size = 1
    img_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = FacesTestDataset(test_img_dir, img_size=img_size, transform=transform)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FacesModel(img_size).to(device)
    model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))

    res_dict = {}
    model.eval()

    for img, img_path, true_size in dataloader:
        out = model(img.to(device)).cpu().detach().numpy().ravel()
        out[0::2] = (true_size[0] / img_size) * out[0::2]
        out[1::2] = (true_size[1] / img_size) * out[1::2]
        res_dict[img_path[0]] = list(out)

    return res_dict