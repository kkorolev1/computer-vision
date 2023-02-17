import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

import numpy as np
import copy

def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """

    model = nn.Sequential(
        nn.Conv2d(in_channels=input_shape[2], out_channels=16, kernel_size=3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Flatten(),

        nn.Linear(in_features=3 * 10 * 64, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=2),
        nn.Softmax()
    )
    return model

def get_cls_model2(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """

    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, padding='valid'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding='valid'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),

        nn.Flatten(),
        nn.Linear(in_features=9856, out_features=2),
        nn.Softmax()
    )
    return model

def fit_cls_model(X, y):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    model = get_cls_model((40, 100, 1))
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    cross_entopy_loss = nn.CrossEntropyLoss()

    model.train()
    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5)
    ])
    nepochs = 25
    for epoch in range(nepochs):
        avg_loss = 0.0
        for data, label in zip(X, y):
            data = data.reshape(1, *data.shape)
            data = transform(data)
            output = model(data)
            loss = cross_entopy_loss(output,  torch.Tensor([label]).to(torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        print("epoch", epoch+1)
        print("loss", avg_loss / len(X))
    #torch.save(model, "classifier_model.pth")
    return model


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    detection_model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 10)),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1),
        nn.Softmax()
    )

    detection_model.eval()
    cls_model.eval()
    
    with torch.no_grad():
        for i in range(12):
            detection_model[i] = copy.deepcopy(cls_model[i])

        detection_model[12].weight.data.copy_(cls_model[13].weight.data.reshape((256, 64, 3, 10)))
        detection_model[12].bias.data.copy_(cls_model[13].bias.data)
    
        detection_model[14].weight.data.copy_(cls_model[15].weight.data.reshape((128, 256, 1, 1)))
        detection_model[14].bias.data.copy_(cls_model[15].bias.data)

        detection_model[16].weight.data.copy_(cls_model[17].weight.data.reshape((2, 128, 1, 1)))
        detection_model[16].bias.data.copy_(cls_model[17].bias.data)

    return detection_model
    
def convert_indxes_to_pix(i, j):
    row_start = 8 * i
    col_start = 8 * j
    return ((row_start, col_start), (40, 100))

def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """

    threshold = 0.9
    preds = {}
    detection_model.eval()

    for filename in dictionary_of_images:
        detections = []
        image = dictionary_of_images[filename]
        img_shape = image.shape
        image = torch.FloatTensor(np.pad(image, ((0, 220 - img_shape[0]), (0, 370 - img_shape[1]))))
        image = image.reshape(1, 1, 220, 370)
        pred = detection_model(image).detach()[0][1]
        pred_shape = (img_shape[0] // 8 - 5, img_shape[1] // 8 - 5)
        pred = pred[:pred_shape[0], :pred_shape[1]]
        for m in range(pred.shape[0]):
            for n in range(pred.shape[1]):
                if pred[m, n].item() > threshold:
                    detections.append([m * 8, n * 8, 40, 100, pred[m, n].item()])
        preds[filename] = detections
    return preds
# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    lu_x = max(first_bbox[0], second_bbox[0])
    lu_y = max(first_bbox[1], second_bbox[1])
    rd_x = min(first_bbox[0] + first_bbox[2], second_bbox[0] + second_bbox[2])
    rd_y = min(first_bbox[1] + first_bbox[3], second_bbox[1] + second_bbox[3])
    if rd_x <= lu_x or rd_y <= lu_y:
        return 0
    else:
        area = (rd_x - lu_x) * (rd_y - lu_y)
        return area / (first_bbox[2] * first_bbox[3] + second_bbox[2] * second_bbox[3] - area)


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
    iou_thr = 0.5
    true_positive = []
    total = []
    true_examples = 0
    for img in pred_bboxes.keys():
        true_examples += len(gt_bboxes[img])
        for pred_box in sorted(pred_bboxes[img], key=lambda x: -x[-1]):
            best_iou = 0
            best_box = None
            for gt_box in gt_bboxes[img]:
                cur_iou = calc_iou(pred_box[:-1], gt_box)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_box = gt_box
            if best_iou >= 0.5:
                gt_bboxes[img].remove(best_box)
                true_positive.append(pred_box[4])
                total.append(pred_box[4])
            else:
                total.append(pred_box[4])

    total = sorted(total, reverse=True)
    true_positive = sorted(true_positive, reverse=True)
    pos = 0
    pr_curve = []
    pr_curve.append([0, 1, 0])
    for i, c in enumerate(total):
        if i < len(total) - 1 and total[i] == total[i + 1]:
            continue
        while pos != len(true_positive) and true_positive[pos] >= c:
            pos += 1
        pr_curve.append((pos / true_examples, pos / (i + 1), c))

    pr_curve.append((len(true_positive) / true_examples, len(true_positive) / len(total), 0))
    res = 0
    for i in range(len(pr_curve) - 1):
        res += (pr_curve[i + 1][0] - pr_curve[i][0]) * (pr_curve[i + 1][1] + pr_curve[i][1]) / 2
    return res


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.1):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    result = {}
    for img in detections_dictionary.keys():
        result[img] = []
        for box in sorted(detections_dictionary[img], key=lambda x: -x[4]):
            for res_box in result[img]:
                if calc_iou(res_box, box) >= iou_thr:
                    break
            else:
                result[img].append(box)
    return result