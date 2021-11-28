import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from .Detection import Detection


class OF(ABC):
    """
    Abstract class
    Foreground Object Tracker implemented with several optical flow algorithms
    """

    def __init__(self):
        self.OPTICAL_FLOW_GRID_SIZE = 15
        self.LK_PARAMS = {
            'winSize': (21, 21),
            'maxLevel': 3,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        }
        self.RANSAC_PROJECTION_THRESHOLD = 10.

        self.GAUSSIAN_THRESHOLD_WINDOW = 31
        self.GAUSSIAN_THRESHOLD_C = 16

        self.BLUR_KERNEL_SIZE = 3

        self.MORPHOLOGY_KERNEL_SIZE = 7
        self.MORPHOLOGY_ITERATIONS = 3

        self.IOU_ASSOCIATION_THRESHOLD = 0.20

        self.CONTOUR_MIN_WIDTH_RATIO = .01
        self.CONTOUR_MAX_WIDTH_RATIO = .50
        self.CONTOUR_MIN_HEIGHT_RATIO = .01
        self.CONTOUR_MAX_HEIGHT_RATIO = .50

        self.DISPLAY_WINDOW_WIDTH = 1680
        self.DISPLAY_FPS = 25
        self.RENDER_FPS = 16

        self.writer = None
        self.frame = None
        self.previous_frame = None
        self.sparse_points_to_track = None
        self.filters = None
        self.known_detections = []

    @staticmethod
    def frame_generator(file):
        video = cv2.VideoCapture(file)
        not_finished = video.isOpened()
        while not_finished:
            not_finished, frame = video.read()
            if not_finished:
                yield frame
        video.release()

    def compute_sparse_points_to_track(self, w, h):
        self.sparse_points_to_track = np.concatenate(
            [
                np.linspace([[0, j]], [[w, j]], self.OPTICAL_FLOW_GRID_SIZE).astype(int)
                for j in range(0, h, h // self.OPTICAL_FLOW_GRID_SIZE)
            ],
            dtype=np.float32
        )

    def filter_best_points_to_track(self):
        return np.array([
            [[x, y]] for [[x, y]] in self.sparse_points_to_track
            if not any(
                detection.x <= x <= detection.x + detection.w and detection.y <= y <= detection.y + detection.h
                for detection in self.known_detections
            )
        ])

    def apply_sparse_optical_flow(self, points_to_track):
        previous_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        if len(points_to_track) >= 4:
            next_, status, error = cv2.calcOpticalFlowPyrLK(
                previous_gray, gray, points_to_track, None, **self.LK_PARAMS
            )
            good_prev = points_to_track[status == 1]
            good_next = next_[status == 1]

            if len(good_next) >= 4:
                homography_matrix, _ = cv2.findHomography(
                    good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=self.RANSAC_PROJECTION_THRESHOLD
                )
                warped_previous_gray = cv2.warpPerspective(previous_gray, homography_matrix,
                                                           dsize=previous_gray.shape[:2][::-1],
                                                           borderMode=cv2.BORDER_REPLICATE)
                warped_diff = cv2.absdiff(warped_previous_gray, gray)
                return warped_diff, homography_matrix
        return previous_gray, np.identity(3, dtype=np.float32)

    def project_detections_on_new_plane(self, homography):
        rot_scaling_mat = homography[:2, :2]
        translation_vec = homography[:2, 2:]
        projection_vec = homography[2:, :2]
        normalization = homography[2, 2]
        for detection in self.known_detections:
            detection.apply_homography(rot_scaling_mat, translation_vec, projection_vec, normalization)

    def apply_dense_optical_flow(self):
        previous_gray_frame = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous_gray_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def apply_gaussian_threshold(self, img):
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self.GAUSSIAN_THRESHOLD_WINDOW, -self.GAUSSIAN_THRESHOLD_C
        )

    def apply_blur(self, img):
        return cv2.medianBlur(img, self.BLUR_KERNEL_SIZE)

    def apply_morphology_closing(self, img):
        max_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.MORPHOLOGY_KERNEL_SIZE, self.MORPHOLOGY_KERNEL_SIZE)
        )
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, max_kernel, None, None,
                                self.MORPHOLOGY_ITERATIONS, cv2.BORDER_REPLICATE)

    @staticmethod
    def find_contours(img):
        return contours \
            if (contours := cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]) is not None \
            else []

    def display_contours(self, contours):
        frame_contours = self.frame.copy()
        for contour in contours:
            cv2.drawContours(frame_contours, [contour], 0, 255, -1)
        return frame_contours

    @staticmethod
    def get_bounding_rects(contours):
        return [cv2.boundingRect(c) for c in contours]

    def union_over_intersection(self, rects, previous_known_detections):
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
                if iou >= self.IOU_ASSOCIATION_THRESHOLD:
                    matching_rects.append(rect)
                    if iou > max_iou:
                        max_iou = iou
                        max_iou_rect = rect
            found = max_iou_rect is not None
            yield detection, found, max_iou_rect, matching_rects

    def assign_rect_to_detections(self, remaining_rects):
        min_w, max_w, min_h, max_h = self.filters
        remaining_rects = sorted(
            [r for r in remaining_rects if max_w >= r[2] >= min_w and max_h >= r[3] >= min_h],
            key=lambda r: r[2] * r[3], reverse=True
        )

        if len(self.known_detections):
            self.known_detections.sort(key=lambda x: x.w * x.h, reverse=True)
            previous_known_detections = self.known_detections.copy()
            self.known_detections.clear()

            for detection, found, best_rect, matching_rects in self.union_over_intersection(
                    remaining_rects, previous_known_detections
            ):
                if found:
                    detection.correct_and_update_position(best_rect)
                    detection.reinforce()
                    self.known_detections.append(detection)
                else:
                    detection.fade()
                    if np.floor(detection.confidence) > 0:
                        self.known_detections.append(detection)
                for rect in matching_rects:
                    remaining_rects.remove(rect)
        for rect in remaining_rects:
            self.known_detections.append(Detection(rect))

    def display_detections(self, rect_color=(0, 255, 0), trajectory_color=(0, 0, 255)):
        frame_contours = self.frame.copy()
        for detection in self.known_detections:
            detection.draw(frame_contours, rect_color, trajectory_color)
        return frame_contours

    @staticmethod
    def add_labels(images, labels):
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
    def concat_views(images):
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
    def resize_with_aspect_ratio(img, width):
        h, w = img.shape[:2]
        r = width / float(w)
        dim = (width, int(h * r))
        return cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    def cv_cleanup(self):
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()

    def run_from_file(self, footage_path, display=True, save_footage=False):
        assert os.path.exists(footage_path), f"File not found: {footage_path}"
        for self.frame in self.frame_generator(footage_path):
            yield self.forward(self.frame, display, save_footage)
        self.cv_cleanup()

    @abstractmethod
    def forward(self, frame, display=True, save_footage=False):
        pass
