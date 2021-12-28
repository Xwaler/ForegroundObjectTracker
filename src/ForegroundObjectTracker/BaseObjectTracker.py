import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Generator

from .Detection import Detection


DISPLAY_NONE = 0
DISPLAY_OVERLAY = 1
DISPLAY_DEBUG = 2


class BaseObjectTracker(ABC):
    """
    Abstract class
    Foreground Object Tracker implemented with several optical flow algorithms
    """

    def __init__(self, display, write_footage):
        self._GAUSSIAN_THRESHOLD_WINDOW = 31
        self._GAUSSIAN_THRESHOLD_C = 16

        self._BLUR_KERNEL_SIZE = 3

        self._MORPHOLOGY_KERNEL_SIZE = 7
        self._MORPHOLOGY_ITERATIONS = 3

        self._IOU_ASSOCIATION_THRESHOLD = 0.20

        self._CONTOUR_MIN_WIDTH_RATIO = .01
        self._CONTOUR_MAX_WIDTH_RATIO = .50
        self._CONTOUR_MIN_HEIGHT_RATIO = .01
        self._CONTOUR_MAX_HEIGHT_RATIO = .50

        self._DISPLAY = display
        self._DISPLAY_WINDOW_WIDTH = 1500
        self._DISPLAY_FPS = 25
        self._RENDER = write_footage
        self._RENDER_FPS = 16

        self._writer = None
        self._sparse_points_to_track = None
        self._filters = None
        self._known_detections = []

        self._frame = None
        self._previous_frame = None

    @staticmethod
    def _frame_generator(file):
        video = cv2.VideoCapture(file)
        not_finished = video.isOpened()
        while not_finished:
            not_finished, frame = video.read()
            if not_finished:
                yield frame
        video.release()

    def _project_detections_on_new_plane(self, homography):
        rot_scaling_mat = homography[:2, :2]
        translation_vec = homography[:2, 2:]
        projection_vec = homography[2:, :2]
        normalization = homography[2, 2]
        for detection in self._known_detections:
            detection._apply_homography(rot_scaling_mat, translation_vec, projection_vec, normalization)

    def _apply_gaussian_threshold(self, img):
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self._GAUSSIAN_THRESHOLD_WINDOW, -self._GAUSSIAN_THRESHOLD_C
        )

    def _apply_blur(self, img):
        return cv2.medianBlur(img, self._BLUR_KERNEL_SIZE)

    def _apply_morphology_closing(self, img):
        max_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._MORPHOLOGY_KERNEL_SIZE, self._MORPHOLOGY_KERNEL_SIZE)
        )
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, max_kernel, None, None,
                                self._MORPHOLOGY_ITERATIONS, cv2.BORDER_REPLICATE)

    @staticmethod
    def _find_contours(img):
        return contours \
            if (contours := cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]) is not None \
            else []

    def _display_contours(self, contours):
        frame_contours = self._frame.copy()
        for contour in contours:
            cv2.drawContours(frame_contours, [contour], 0, 255, -1)
        return frame_contours

    @staticmethod
    def _get_bounding_rects(contours):
        return [cv2.boundingRect(c) for c in contours]

    def _intersection_over_union(self, rects, previous_known_detections):
        for detection in previous_known_detections:
            area0 = detection.w * detection.h
            max_iou = 0
            max_iou_rect = None
            matching_rects = []
            for rect in rects:
                x1, y1, w1, h1 = rect

                area1 = w1 * h1
                w_inter = min(detection.x + detection.w, x1 + w1) - max(detection.x, x1)
                h_inter = min(detection.y + detection.h, y1 + h1) - max(detection.y, y1)
                if w_inter <= 0 or h_inter <= 0:
                    continue
                area_inter = w_inter * h_inter
                area_union = area0 + area1 - area_inter
                iou = area_inter / (area_union + 1e-16)
                if iou >= self._IOU_ASSOCIATION_THRESHOLD:
                    matching_rects.append(rect)
                    if iou > max_iou:
                        max_iou = iou
                        max_iou_rect = rect
            found = max_iou_rect is not None
            yield detection, found, max_iou_rect, matching_rects

    def _assign_rect_to_detections(self, remaining_rects):
        min_w, max_w, min_h, max_h = self._filters
        remaining_rects = sorted(
            [r for r in remaining_rects if max_w >= r[2] >= min_w and max_h >= r[3] >= min_h],
            key=lambda r: r[2] * r[3], reverse=True
        )

        if len(self._known_detections):
            self._known_detections.sort(key=lambda x: x.w * x.h, reverse=True)
            previous_known_detections = self._known_detections.copy()
            self._known_detections.clear()

            for detection, found, best_rect, matching_rects in self._intersection_over_union(
                    remaining_rects, previous_known_detections
            ):
                if found:
                    detection._correct_and_update_position(best_rect)
                    detection._reinforce()
                    self._known_detections.append(detection)
                else:
                    detection._fade()
                    if np.floor(detection._confidence) > 0:
                        self._known_detections.append(detection)
                for rect in matching_rects:
                    remaining_rects.remove(rect)
        for rect in remaining_rects:
            self._known_detections.append(Detection(rect))

    def getFrame(self):
        return self._frame.copy()

    def getPreviousFrame(self):
        return self._previous_frame.copy()

    def display_detections(self, rect_color=(0, 255, 0), trajectory_color=(0, 0, 255)) -> np.ndarray:
        """
        Returns the actual frame with a detection overlay
        :param rect_color:
        :param trajectory_color:
        :return: image
        """
        frame_contours = self._frame.copy()
        for detection in self._known_detections:
            if detection._has_sufficient_confidence():
                detection.draw(frame_contours, rect_color, trajectory_color)
        return frame_contours

    @staticmethod
    def _add_labels(images, labels):
        labeled_images = []
        for image, label in zip(images, labels):
            label_image = np.full((30, image.shape[1], 3), 255, dtype=np.uint8)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (label_image.shape[1] - text_size[0]) // 2
            text_y = (label_image.shape[0] + text_size[1]) // 2
            cv2.putText(label_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            labeled_images.append(cv2.vconcat([label_image, image]))
        return labeled_images

    @staticmethod
    def _concat_views(images):
        l_size = np.ceil(np.sqrt(len(images))).astype('uint8')
        h_size = np.ceil(len(images) / l_size).astype('uint8')
        for _ in range(l_size * h_size - len(images)):
            images.append(np.zeros_like(images[0], dtype=np.uint8))
        lines = [
            cv2.hconcat(images[k * l_size: (k + 1) * l_size])
            for k in range(h_size)
        ]
        concat = cv2.vconcat(lines)
        return concat

    @staticmethod
    def _resize_with_aspect_ratio(img, width):
        h, w = img.shape[:2]
        r = width / float(w)
        dim = (width, int(h * r))
        return cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    def cv_cleanup(self) -> None:
        """
        Closes cv2 windows and the video writer
        :return:
        """
        if self._writer is not None:
            self._writer.release()
        cv2.destroyAllWindows()

    def run_from_file(self, footage_path: str) -> Generator[list[Detection], None, None]:
        """
        Run the tracker from a file
        :param footage_path:
        :return: Generator of detections for all the frames in the video
        """
        assert os.path.exists(footage_path), f"File not found: {footage_path}"
        for self._frame in self._frame_generator(footage_path):
            yield self.forward(self._frame)
        self.cv_cleanup()

    @abstractmethod
    def forward(self, frame: np.array) -> list[Detection]:
        """
        Feeds one frame to the Tracker
        :param frame:
        :return: detections
        """
        pass
