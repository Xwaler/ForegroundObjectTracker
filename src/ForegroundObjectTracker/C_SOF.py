import cv2
import numpy as np
from .SOF import SOF


class C_SOF(SOF):
    """
    Custom Sparse Optical Flow
    """
    def __init__(self):
        super().__init__()
        self.OPTICAL_FLOW_GRID_SIZE = 15

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

    def forward(self, frame, display=True, save_footage=False):
        assert frame is not None, "Frame is None"
        self.frame = frame
        if self.previous_frame is None:
            self.previous_frame = self.frame
            h, w = self.frame.shape[:2]
            self.compute_sparse_points_to_track(w, h)
            self.filters = np.ceil([
                self.CONTOUR_MIN_WIDTH_RATIO * w,
                self.CONTOUR_MAX_WIDTH_RATIO * w,
                self.CONTOUR_MIN_HEIGHT_RATIO * h,
                self.CONTOUR_MAX_HEIGHT_RATIO * h,
            ])
            return []

        best_points_to_track = self.filter_best_points_to_track()
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
                cv2.imshow('Custom Sparse Optical Flow', resized)
                if display and cv2.waitKey(1) in [27, ord('q'), ord('Q')]:
                    exit()

        self.previous_frame = self.frame
        return [detection for detection in self.known_detections if detection.has_sufficient_confidence()]
