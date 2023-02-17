#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torch.nn.utils import prune

import cv2

from PIL import Image

import numpy as np
import csv
import os

import gc

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
CUDA = 2

def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()

class ImagesDataset(Dataset):
    def __init__(self, images_paths, transform=None):
        self.samples = images_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, gt_img_path = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        gt_img = Image.open(gt_img_path).convert('L')

        if self.transform:
            img, gt_img = self.transform(img, gt_img)
        
        return img, gt_img
    
class TrainTransform(object):
    def __init__(self):
        #self.transform = A.Compose([
        #    A.Resize(width=224, height=224),
            #T.HorizontalFlip(p=0.5),
            #T.RandomApply([T.RandomRotation(degrees=30)], p=0.5),
        #    ToTensorV2()
        #])
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
    def __call__(self, image, gt_image):
        #images_dict = self.transform(image=image, mask=gt_image)
        #return self.normalize(images_dict['image']), images_dict['mask']
        return self.normalize(self.transform(image)), self.transform(gt_image)

class TestTransform(object):
    def __init__(self):
        #self.transform = A.Compose([
        #    A.Resize(width=224, height=224),
        #    ToTensorV2()
        #])
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
    def __call__(self, image, gt_image):
        #images_dict = self.transform(image=image, mask=gt_image)
        #return self.normalize(images_dict['image']), images_dict['mask']
        return self.normalize(self.transform(image)), self.transform(gt_image)
    
class SaveBestModel:
    def __init__(self, best_val_loss=np.inf):
        self.best_val_loss = best_val_loss
        
    def __call__(self, val_loss, epoch, model, optimizer, scheduler=None, model_path='model/best_model.pth'):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else {}
                }, model_path
            )
            print('New best model with loss {:.5f} is saved'.format(val_loss))

def save_model(epoch, model, optimizer, model_path='model/final_model.pth'):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if scheduler else {}
                }, model_path
    )
    print('Model is saved')
    
def load_model(model, optimizer, scheduler=None, model_path='model/best_model.pth'):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, epoch, scheduler
    
    
# Remove further
#from tqdm import tqdm

def get_iou(gt, pred):
    return (gt & pred).sum() / (gt | pred).sum()

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +\
                                                 target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def train_epoch(model, optimizer, criterion, train_loader, device, tqdm_desc):
    model.train()
    bce_loss_avg, dice_loss_avg = 0.0, 0.0
    
    bce_weight = 0.9
    
    for images, gt_images in train_loader:
    #for images, gt_images in tqdm(train_loader, desc=tqdm_desc):
        images = images.to(device)
        gt_images = gt_images.to(device)

        optimizer.zero_grad()

        logits = model(images)
        bce_loss = criterion(logits, gt_images)

        bce_loss.backward()
        optimizer.step()
        
        pred = torch.sigmoid(logits)
        dice = dice_loss(pred, gt_images)
        
        bce_loss_avg += bce_loss * gt_images.shape[0]
        dice_loss_avg += bce_loss * bce_weight + dice * (1 - bce_weight) * gt_images.shape[0]

    bce_loss_avg /= len(train_loader.dataset)
    dice_loss_avg /= len(train_loader.dataset)

    return bce_loss_avg, dice_loss_avg


@torch.no_grad()
def val_epoch(model, criterion, val_loader, device, tqdm_desc):
    model.eval()

    bce_loss_avg, dice_loss_avg = 0.0, 0.0
    bce_weight = 0.9
        
    #for images, gt_images in tqdm(val_loader, desc=tqdm_desc):
    for images, gt_images in val_loader:
        images = images.to(device)
        gt_images = gt_images.to(device)

        logits = model(images)
        bce_loss = criterion(logits, gt_images)

        pred = torch.sigmoid(logits)
        dice = dice_loss(pred, gt_images)
        
        bce_loss_avg += bce_loss * gt_images.shape[0]
        dice_loss_avg += bce_loss * bce_weight + dice * (1 - bce_weight) * gt_images.shape[0]

    bce_loss_avg /= len(val_loader.dataset)
    dice_loss_avg /= len(val_loader.dataset)

    return bce_loss_avg, dice_loss_avg


def train(model, optimizer, criterion, scheduler, train_loader, val_loader, device, num_epochs, model_saver, continue_training=True, model_path='model/best_model.pth', start_epoch=0):
    
    if continue_training:
        model, optimizer, start_epoch, scheduler = load_model(model, optimizer, scheduler, model_path)
        print(f"Continue training from epoch {start_epoch+1}")
    
    for epoch in range(start_epoch + 1, num_epochs + 1):
        train_bce_loss, train_dice_loss = train_epoch(model, optimizer, criterion, train_loader, device, f'Training epoch {epoch}/{num_epochs}')
        val_bce_loss, val_dice_loss = val_epoch(model, criterion, val_loader, device, f'Validating epoch {epoch}/{num_epochs}')

        if scheduler is not None:
            scheduler.step(val_bce_loss)
        
        print(f"epoch {epoch}")
        print({'train_bce_loss': train_bce_loss.item(), 'train_dice_loss': train_dice_loss.item(), 'val_bce_loss': val_bce_loss.item(), 'val_dice_loss': train_dice_loss.item()})
        model_saver(val_bce_loss, epoch, model, optimizer, scheduler, model_path)
    
def convReLU(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convReLU(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convReLU(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convReLU(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convReLU(256, 256, 1, 0)
        #self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        #self.layer4_1x1 = convReLU(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.conv_up3 = convReLU(256 + 512, 512, 3, 1)
        self.conv_up2 = convReLU(128 + 256, 256, 3, 1)
        self.conv_up1 = convReLU(64 + 256, 256, 3, 1)
        self.conv_up0 = convReLU(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convReLU(3, 64, 3, 1)
        self.conv_original_size1 = convReLU(64, 64, 3, 1)
        self.conv_original_size2 = convReLU(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)

        layer3 = self.layer3_1x1(layer3)
        x = self.upsample(layer3)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet()
        
        # freeze backbone layers
        for l in self.model.base_layers:
            for param in l.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)

class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold
    
def get_model():
    return Model()

def train_model(train_data_path):
    #empty_cache()
    
    images_dir = os.path.join(train_data_path, "images")
    images_paths = []

    for class_dir in sorted(os.listdir(images_dir)):
        for image_path in sorted(os.listdir(os.path.join(images_dir, class_dir))):
            image_path = os.path.splitext(image_path)[0]
            train_image_path = os.path.join(images_dir, class_dir, image_path + ".jpg")
            gt_image_path = os.path.join(train_data_path, "gt", class_dir, image_path + ".png")
            images_paths.append((train_image_path, gt_image_path))
    
    images_paths = np.array(images_paths)
    
    val_size = int(0.25 * len(images_paths))
    train_idx, val_idx = random_split(np.arange(len(images_paths)), [len(images_paths) - val_size, val_size])
    
    train_transform = TrainTransform()
    test_transform = TestTransform()
    
    train_dataset = ImagesDataset(images_paths[train_idx.indices], transform=train_transform)
    val_dataset = ImagesDataset(images_paths[val_idx.indices], transform=test_transform)
    
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
    model_saver = SaveBestModel()

    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    
    num_epochs = 10
    
    train(model, optimizer, criterion, scheduler, train_loader, val_loader, device, num_epochs, model_saver, continue_training=False)

def prune_model(model):
    parameters_to_prune = []

    for module in model.model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune += [(module, 'weight')]
            if module.bias is not None:
                parameters_to_prune += [(module, 'bias')]

    #prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruning, threshold=1e-5)
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.3)
    
    for params, name in parameters_to_prune:
        print("Sparsity in {}: {:.2f}%".format(params, 100. * float(torch.sum(params.weight == 0)) / float(params.weight.nelement())))
        prune.remove(params, name=name)
    
    torch.set_flush_denormal(True)
    print("pruned model")
    
def predict2(model, img_path):
    device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    #state_dict = torch.load('model/best_model.pth')
    #model.load_state_dict(state_dict['model_state_dict'])
    
    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image = Image.open(img_path).convert('RGB')
    image_size = image.size
    image = test_transform(image)
    
    model.eval()
    out = model(image.to(device).unsqueeze(0)).cpu().detach().numpy()
    out = out.reshape(*out.shape[2:])
    out = cv2.resize(out, dsize=image_size, interpolation=cv2.INTER_CUBIC)
    
    return out

@torch.no_grad()
def predict(model, img_path):
    #print("predict")
    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    image = Image.open(img_path).convert('RGB')
    image_size = image.size
    image = test_transform(image)
    
    model.eval()
    out = model(image.unsqueeze(0)).cpu().detach().numpy()
    out = out.reshape(*out.shape[2:])
    out = cv2.resize(out, dsize=image_size, interpolation=cv2.INTER_CUBIC)
    
    return out

if __name__ == "__main__":
    # Saving model
    #model = get_model()
    #model.load_state_dict(torch.load("segmentation_model.pth", map_location="cpu"))
    #prune_model(model)
    #torch.save(model.state_dict(), "segmentation_model_pruned.pth")
    
    #code_dir = os.path.dirname(os.path.abspath(__file__))
    #train_model(os.path.join(code_dir, "tests/00_test_val_input/train"))
    #img_path = os.path.join(code_dir, "tests/00_test_val_input/test/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg")
    #predict(get_model(), img_path)
    pass