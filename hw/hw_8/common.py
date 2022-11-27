from detection_and_metrics import get_cls_model, get_detection_model, calc_auc, nms, get_detections
from skimage import img_as_float32
from os.path import join
from torch import load


def calc_detector_auc(img_dir, gt_path, apply_nms=False):
    classifier_model = load('classifier_model.pth')
    detection_model = get_detection_model(classifier_model)
    images_detection = read_for_detection(img_dir, gt_path)
    images_detection_no_answer = {}
    images_detection_only_bboxes = {}
    for img_name, data in images_detection.items():
        images_detection_no_answer[img_name] = data[0]
        images_detection_only_bboxes[img_name] = data[1]
    pred = get_detections(detection_model, images_detection_no_answer)
    if apply_nms:
        pred = nms(pred)
    return calc_auc(pred, images_detection_only_bboxes)


def read_for_detection(img_dir, gt_path):
    from skimage.io import imread
    from json import load

    raw_data = load(open(gt_path))
    data = {}
    for file_name, bboxes in raw_data.items():
        data[file_name] = [img_as_float32(imread(join(img_dir, file_name))), bboxes]
    return data
