import cv2
import numpy as np
from .OF import OF


class DOF(OF):
    """
    Dense Optical Flow
    """

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

        flow = self.apply_dense_optical_flow()
        threshold = self.apply_gaussian_threshold(flow)
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
                cv2.cvtColor(flow, cv2.COLOR_GRAY2BGR),
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
                cv2.imshow('Display', resized)
                if display and cv2.waitKey(1) in [27, ord('q'), ord('Q')]:
                    exit()

        self.previous_frame = self.frame
        return [detection for detection in self.known_detections if detection.has_sufficient_confidence()]
