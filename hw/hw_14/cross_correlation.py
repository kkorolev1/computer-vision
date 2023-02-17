import numpy as np
import os

from moviepy.editor import VideoFileClip
from skimage.feature import match_template
from skimage.color import rgb2gray

from detection import extract_detections, draw_detections, detection_cast
from tracker import Tracker


def gaussian(shape, x, y, dx, dy):
    """Return gaussian for tracking.

    shape: [width, height]
    x, y: gaussian center
    dx, dy: std by x and y axes

    return: numpy array (width x height) with gauss function, center (x, y) and std (dx, dy)
    """
    Y, X = np.mgrid[0:shape[0], 0:shape[1]]
    return np.exp(-(X - x) ** 2 / dx ** 2 - (Y - y) ** 2 / dy ** 2)


class CorrelationTracker(Tracker):
    """Generate detections and building tracklets."""
    def __init__(self, detection_rate=5, **kwargs):
        super().__init__(**kwargs)
        self.detection_rate = detection_rate  # Detection rate
        self.prev_frame = None  # Previous frame (used in cross correlation algorithm)


    def build_tracklet(self, frame):
        """Between CNN execution uses normalized cross-correlation algorithm (match_template)."""
        detections = []
        # Write code here
        # Apply rgb2gray to frame and previous frame
        frame = rgb2gray(frame)
        self.prev_frame = rgb2gray(self.prev_frame)

        # For every previous detection
        # Use match_template + gaussian to extract detection on current frame
        for label, xmin, ymin, xmax, ymax in self.detection_history[-1]:
            # Step 0: Extract prev_bbox from prev_frame

            # Step 1: Extract new_bbox from current frame with the same coordinates

            # Step 2: Calc match_template between previous and new bbox
            # Use padding

            # Step 3: Then multiply matching by gauss function
            # Find argmax(matching * gauss)

            # Step 4: Append to detection list
            prev_bbox = self.prev_frame[ymin: ymax, xmin: xmax]
            new_bbox = frame[ymin: ymax, xmin: xmax]
            corr = match_template(new_bbox, prev_bbox, pad_input=True)

            height = ymax - ymin
            width = xmax - xmin
            corr *= gaussian((height, width), height // 2, width // 2, height, width)

            cur_bbox_y, cur_bbox_x = np.unravel_index(np.argmax(corr), corr.shape)

            # Step 4: Append to detection list
            frame_height = frame.shape[0] - 1
            frame_width = frame.shape[1] - 1

            detections.append([label, min(frame_width, max(xmin - width // 2 + cur_bbox_x, 0)), min(frame_height, max(ymin - height // 2 + cur_bbox_y, 0)),
                               min(frame_width, max(xmax - width // 2 + cur_bbox_x, 0)), min(frame_height, max(ymax - height // 2 + cur_bbox_y, 0))])

        return detection_cast(detections)


    def update_frame(self, frame):
        if not self.frame_index:
            detections = self.init_tracklet(frame)
            self.save_detections(detections)
        elif self.frame_index % self.detection_rate == 0:
            detections = extract_detections(frame, labels=self.labels)
            detections = self.bind_tracklet(detections)
            self.save_detections(detections)
        else:
            detections = self.build_tracklet(frame)

        self.detection_history.append(detections)
        self.prev_frame = frame
        self.frame_index += 1

        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, 'tests', '06_unittest_tracking_input', 'data', 'jogging.mp4'))

    tracker = CorrelationTracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == '__main__':
    main()

