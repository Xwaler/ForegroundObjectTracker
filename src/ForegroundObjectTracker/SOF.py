import cv2
import numpy as np
from .BaseObjectTracker import BaseObjectTracker


class SOF(BaseObjectTracker):
    """
    Sparse Optical Flow
    """

    def __init__(self):
        super().__init__()
        self.FEATURE_PARAMS = {
            'maxCorners': 300,
            'qualityLevel': 0.2,
            'minDistance': 2,
            'blockSize': 7,
            'useHarrisDetector': True
        }
        self.LK_PARAMS = {
            'winSize': (21, 21),
            'maxLevel': 3,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        }
        self.RANSAC_PROJECTION_THRESHOLD = 10.

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

    def forward(self, frame, display=True, save_footage=False):
        assert frame is not None, "Frame is None"
        self.frame = frame
        if self.previous_frame is None:
            self.previous_frame = self.frame
            h, w = self.frame.shape[:2]
            self.filters = np.ceil([
                self.CONTOUR_MIN_WIDTH_RATIO * w,
                self.CONTOUR_MAX_WIDTH_RATIO * w,
                self.CONTOUR_MIN_HEIGHT_RATIO * h,
                self.CONTOUR_MAX_HEIGHT_RATIO * h,
            ])
            return []

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_points_to_track = cv2.goodFeaturesToTrack(gray_frame, **self.FEATURE_PARAMS)
        corrected_frame, homography_matrix = self.apply_sparse_optical_flow(best_points_to_track)
        self.project_detections_on_new_plane(homography_matrix)

        threshold = self.apply_gaussian_threshold(corrected_frame)
        filtered = self.apply_blur(threshold)

        morphology = self.apply_morphology_closing(filtered)

        contours = self.find_contours(morphology)
        frame_contours = self.display_contours(contours)

        bounding_rects = self.get_bounding_rects(contours)
        self.assign_rect_to_detections(bounding_rects)

        if display or save_footage:
            frame_detections = self.display_detections()
            labeled_images = self.add_labels([
                self.frame,
                cv2.cvtColor(corrected_frame, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(morphology, cv2.COLOR_GRAY2BGR),
                frame_contours,
                frame_detections
            ], labels=[
                "Original", "Optical flow corrected", "Gaussian threshold",
                "Morphology closing", "Contours", "Detections"
            ])
            processed_frame = self.concat_views(labeled_images)

            if save_footage:
                if self.writer is None:
                    self.writer = cv2.VideoWriter(
                        'output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.RENDER_FPS, processed_frame.shape[:2][::-1]
                    )
                self.writer.write(processed_frame)

            if display:
                resized = self.resize_with_aspect_ratio(processed_frame, width=self.DISPLAY_WINDOW_WIDTH)
                cv2.imshow('Sparse Optical Flow', resized)
                if display and cv2.waitKey(1) in [27, ord('q'), ord('Q')]:
                    exit()

        self.previous_frame = self.frame
        return [detection for detection in self.known_detections if detection.has_sufficient_confidence()]
