import cv2
import numpy as np

from .BaseObjectTracker import BaseObjectTracker, DISPLAY_OVERLAY, DISPLAY_DEBUG


class SOF(BaseObjectTracker):
    """
    Sparse Optical Flow
    """

    def __init__(self, display=DISPLAY_OVERLAY, write_footage=False):
        super().__init__(display, write_footage)
        self._FEATURE_PARAMS = {
            'maxCorners': 300,
            'qualityLevel': 0.2,
            'minDistance': 2,
            'blockSize': 7,
            'useHarrisDetector': True
        }
        self._LK_PARAMS = {
            'winSize': (21, 21),
            'maxLevel': 3,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        }
        self._RANSAC_PROJECTION_THRESHOLD = 10.

    def _apply_sparse_optical_flow(self, points_to_track):
        previous_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        if len(points_to_track) >= 4:
            next_, status, error = cv2.calcOpticalFlowPyrLK(
                previous_gray, gray, points_to_track, None, **self._LK_PARAMS
            )
            good_prev = points_to_track[status == 1]
            good_next = next_[status == 1]

            if len(good_next) >= 4:
                homography_matrix, _ = cv2.findHomography(
                    good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=self._RANSAC_PROJECTION_THRESHOLD
                )
                warped_previous_gray = cv2.warpPerspective(previous_gray, homography_matrix,
                                                           dsize=previous_gray.shape[:2][::-1],
                                                           borderMode=cv2.BORDER_REPLICATE)
                warped_diff = cv2.absdiff(warped_previous_gray, gray)
                return warped_diff, homography_matrix
        return previous_gray, np.identity(3, dtype=np.float32)

    def forward(self, frame):
        assert frame is not None, "Frame is None"
        self.frame = frame
        if self.previous_frame is None:
            self.previous_frame = self.frame
            h, w = self.frame.shape[:2]
            self._filters = np.ceil([
                self._CONTOUR_MIN_WIDTH_RATIO * w,
                self._CONTOUR_MAX_WIDTH_RATIO * w,
                self._CONTOUR_MIN_HEIGHT_RATIO * h,
                self._CONTOUR_MAX_HEIGHT_RATIO * h,
            ])
            return []

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_points_to_track = cv2.goodFeaturesToTrack(gray_frame, **self._FEATURE_PARAMS)
        corrected_frame, homography_matrix = self._apply_sparse_optical_flow(best_points_to_track)
        self._project_detections_on_new_plane(homography_matrix)

        threshold = self._apply_gaussian_threshold(corrected_frame)
        filtered = self._apply_blur(threshold)

        morphology = self._apply_morphology_closing(filtered)

        contours = self._find_contours(morphology)
        frame_contours = self._display_contours(contours)

        bounding_rects = self._get_bounding_rects(contours)
        self._assign_rect_to_detections(bounding_rects)

        if self._DISPLAY or self._RENDER:
            frame_detections = self.display_detections()
            if self._DISPLAY == DISPLAY_DEBUG:
                labeled_images = self._add_labels([
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
                processed_frame = self._concat_views(labeled_images)
            else:
                processed_frame = self._add_labels([frame_detections], ["Detections"])[0]

            if self._RENDER:
                if self._writer is None:
                    self._writer = cv2.VideoWriter(
                        'output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self._RENDER_FPS, processed_frame.shape[:2][::-1]
                    )
                self._writer.write(processed_frame)

            if self._DISPLAY:
                resized = self._resize_with_aspect_ratio(processed_frame, width=self._DISPLAY_WINDOW_WIDTH)
                cv2.imshow('Sparse Optical Flow', resized)
                if cv2.waitKey(1) in [27, ord('q'), ord('Q')]:
                    exit()

        self.previous_frame = self.frame
        return [detection for detection in self._known_detections if detection._has_sufficient_confidence()]
