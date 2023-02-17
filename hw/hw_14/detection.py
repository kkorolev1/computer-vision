import numpy as np
import os

from PIL import Image
from skimage.transform import resize
from skimage import io
from config import model, VOC_CLASSES, bbox_util
from utils import get_color

import skimage
import torch


def detection_cast(detections):
    """Helper to cast any array to detections numpy array.
    Even empty.
    """
    return np.array(detections, dtype=np.int32).reshape((-1, 5))


def rectangle(shape, ll, rr, line_width=5):
    """Draw rectangle on numpy array.

    rr, cc = rectangle(frame.shape, (ymin, xmin), (ymax, xmax))
    frame[rr, cc] = [0, 255, 0] # Draw green bbox
    """
    ll = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(ll, 0))
    rr = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(rr, 0))
    result = []

    for c in range(line_width):
        for i in range(ll[0] + c, rr[0] - c + 1):
            result.append((i, ll[1] + c))
            result.append((i, rr[1] - c))
        for j in range(ll[1] + c + 1, rr[1] - c):
            result.append((ll[0] + c, j))
            result.append((rr[0] - c, j))

    return tuple(zip(*result))

@torch.no_grad()
def extract_detections(frame, min_confidence=0.6, labels=None):
    """Extract detections from frame.

    frame: numpy array WxHx3
    returns: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
    """
    # Write code here
    # First, convert frame to float and resize to 300x300, convert RGB to BGR
    # then center it with respect to imagenet means

    # imagenet means for BGR
    mean = np.array([103.939, 116.779, 123.68]).reshape(1, 1, 3)
    def image2tensor(image):
        image = image.astype(np.float32) # convert frame to float
        image = resize(image, (300, 300)) # resize image to 300x300
        image = image[..., ::-1] # convert RGB to BGR
        image -= mean
        image = image.transpose([2, 0, 1]) # torch works with CxHxW images
        tensor = torch.tensor(image.copy()).unsqueeze(0)
        # tensor.shape == (1, channels, height, width)
        return tensor

    frame_size = frame.shape
    input_tensor = image2tensor(frame)
    predictions = model(input_tensor).numpy()

    # Then use image2tensor, model(input_tensor), convert output to numpy
    # and bbox_util.detection_out
    # Use help(...) function to help
    results = bbox_util.detection_out(predictions)[0]
    results = (results[results[:, 1] > min_confidence])
    #results = np.array(bbox_util.detection_out(predictions, confidence_threshold=min_confidence)).squeeze(axis=0)

    # Select detections with confidence > min_confidence
    # hint: you can make it passing min_confidence as
    # a confidence_threshold argument of bbox_util.detection_out

    # If label set is known, use it
    if labels is not None:
        result_labels = results[:, 0].astype(np.int32)
        indeces = [i for i, l in enumerate(result_labels) if VOC_CLASSES[l - 1] in labels]
        results = results[indeces]

    # Remove confidence column from result
    results = np.delete(results, 1, axis=1)

    # Resize detection coords to input image shape.
    # Didn't you forget to save it before resize?
    results[:, [1, 3]] *= frame_size[1]
    results[:, [2, 4]] *= frame_size[0]

    # Return result
    return detection_cast(results)


def draw_detections(frame, detections):
    """Draw detections on frame.

    Hint: help(rectangle) would help you.
    Use get_color(label) to select color for detection.
    """
    frame = frame.copy()

    # Write code here
    for detection in detections:
        rr, cc = rectangle(frame.shape, (detection[2], detection[1]), (detection[4], detection[3]))
        frame[rr, cc] = get_color(detection[0])

    return frame


def main():
    dirname = os.path.dirname(__file__)
    frame = Image.open(os.path.join(dirname, 'data', 'test.png'))
    frame = np.array(frame)

    detections = extract_detections(frame)
    frame = draw_detections(frame, detections)

    io.imsave('out.png', frame)
    io.imshow(frame)
    io.show()



if __name__ == '__main__':
    main()
