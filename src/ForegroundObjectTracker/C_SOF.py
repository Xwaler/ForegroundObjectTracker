import cv2
import numpy as np

from .SOF import SOF, DISPLAY_OVERLAY, DISPLAY_DEBUG


class C_SOF(SOF):
    """
    Custom Sparse Optical Flow
    """
    def __init__(self, display=DISPLAY_OVERLAY, write_footage=False):
        super().__init__(display, write_footage)
        self._OPTICAL_FLOW_GRID_SIZE = 15

    def _compute_sparse_points_to_track(self, w, h):
        self.sparse_points_to_track = np.concatenate(
            [
                np.linspace([[0, j]], [[w, j]], self._OPTICAL_FLOW_GRID_SIZE).astype(int)
                for j in range(0, h, h // self._OPTICAL_FLOW_GRID_SIZE)
            ],
            dtype=np.float32
        )

    def _filter_best_points_to_track(self):
        return np.array([
            [[x, y]] for [[x, y]] in self.sparse_points_to_track
            if not any(
                detection.x <= x <= detection.x + detection.w and detection.y <= y <= detection.y + detection.h
                for detection in self.known_detections
            )
        ])

    def forward(self, frame):
        assert frame is not None, "Frame is None"
        self.frame = frame
        if self.previous_frame is None:
            self.previous_frame = self.frame
            h, w = self.frame.shape[:2]
            self._compute_sparse_points_to_track(w, h)
            self.filters = np.ceil([
                self.CONTOUR_MIN_WIDTH_RATIO * w,
                self.CONTOUR_MAX_WIDTH_RATIO * w,
                self.CONTOUR_MIN_HEIGHT_RATIO * h,
                self.CONTOUR_MAX_HEIGHT_RATIO * h,
            ])
            return []

        best_points_to_track = self._filter_best_points_to_track()
        corrected_frame, homography_matrix = self._apply_sparse_optical_flow(best_points_to_track)
        self._project_detections_on_new_plane(homography_matrix)

        threshold = self._apply_gaussian_threshold(corrected_frame)
        filtered = self._apply_blur(threshold)

        morphology = self._apply_morphology_closing(filtered)

        contours = self._find_contours(morphology)
        frame_contours = self._display_contours(contours)

        bounding_rects = self._get_bounding_rects(contours)
        self._assign_rect_to_detections(bounding_rects)

        if self.DISPLAY or self.RENDER:
            frame_detections = self.display_detections()
            if self.DISPLAY == DISPLAY_DEBUG:
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

            if self.RENDER:
                if self.writer is None:
                    self.writer = cv2.VideoWriter(
                        'output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.RENDER_FPS, processed_frame.shape[:2][::-1]
                    )
                self.writer.write(processed_frame)

            if self.DISPLAY:
                resized = self._resize_with_aspect_ratio(processed_frame, width=self.DISPLAY_WINDOW_WIDTH)
                cv2.imshow('Custom Sparse Optical Flow', resized)
                if cv2.waitKey(1) in [27, ord('q'), ord('Q')]:
                    exit()

        self.previous_frame = self.frame
        return [detection for detection in self.known_detections if detection._has_sufficient_confidence()]
