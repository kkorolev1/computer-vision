import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2, resnet18
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import copy

# ============================== 1 Classifier model ============================

CUDA = 0


def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
    model = resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Linear(512, 2),
        #nn.BatchNorm1d(256),
        #nn.ReLU(),
        #nn.Linear(256, 128),
        #nn.BatchNorm1d(128),
        #nn.ReLU(),
        #nn.Linear(128, 2)
    )
    model = model.to(device)
    return model


def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    train_acc, train_loss = 0.0, 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        train_acc += (logits.argmax(dim=1) == labels).sum().item()
        train_loss += loss.item() * labels.shape[0]

    train_acc /= len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)

    return train_acc, train_loss


def train(model, optimizer, criterion, scheduler, train_loader, device, num_epochs):

    for epoch in range(1, num_epochs + 1):
        train_acc, train_loss = train_epoch(model, optimizer, criterion, train_loader, device)

        if scheduler is not None:
            scheduler.step()


def fit_cls_model(X, y):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model((40, 100, 1))

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        #T.RandomRotation((-5,5)),
        T.Normalize(mean=0.5, std=0.5),
    ])

    batch_size = 32

    train_dataset = TensorDataset(train_transform(X), y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = None

    n_epochs = 10
    device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')

    train(model, optimizer, criterion, scheduler, train_dataloader, device, n_epochs)

    return model.cpu()

# ============================ 2 Classifier -> FCN =============================
def linear_to_conv(layer):
    in_channels = layer.in_features
    out_channels = layer.out_features

    weights = layer.state_dict()['weight'][:].view(out_channels, in_channels, 1, 1)
    bias = layer.state_dict()['bias'][:]

    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
    conv.state_dict()['weight'][:] = weights
    conv.state_dict()['bias'][:] = bias

    return conv

class DetectionModel(nn.Module):
    def __init__(self, cls_model):
        super().__init__()
        self.cls_model = copy.deepcopy(cls_model)
        self.cls_model.avgpool = nn.Identity()
        self.cls_model.fc[0] = linear_to_conv(self.cls_model.fc[0])


    def forward(self, x):
        x = self.cls_model.conv1(x)
        x = self.cls_model.bn1(x)
        x = self.cls_model.relu(x)
        x = self.cls_model.maxpool(x)
        x = self.cls_model.layer1(x)
        x = self.cls_model.layer2(x)
        x = self.cls_model.layer3(x)
        x = self.cls_model.layer4(x)
        x = self.cls_model.avgpool(x)
        #x = torch.flatten(x, 1)
        x = self.cls_model.fc(x)

        return x

def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    return DetectionModel(cls_model).double()

# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    detections_dict = {}
    threshold = 3
    n_rows, n_cols = 7, 12

    for image_filepath, orig_image in dictionary_of_images.items():
        image = np.zeros((220, 370))
        image[:orig_image.shape[0], :orig_image.shape[1]] = orig_image

        torch_image = torch.from_numpy(image)
        torch_image = torch_image.view(1, 1, *image.shape).double()
        feature_map = detection_model(torch_image).squeeze(dim=0).detach().numpy()

        pos_feature_map = feature_map[0,:]

        boxes_idxs = np.argsort(pos_feature_map.ravel())[:threshold]

        row_indices = boxes_idxs // n_cols
        col_indices = boxes_idxs % n_cols

        detections = np.stack((row_indices, col_indices, np.full(threshold, n_rows), np.full(threshold, n_cols), pos_feature_map[row_indices, col_indices]), axis=1)

        detections_dict[image_filepath] = detections

    return detections_dict


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    return 1
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    return 1
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    return {}
    # your code here /\
