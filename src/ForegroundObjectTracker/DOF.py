import cv2
import numpy as np

from .BaseObjectTracker import BaseObjectTracker, DISPLAY_OVERLAY, DISPLAY_DEBUG


class DOF(BaseObjectTracker):
    """
    Dense Optical Flow
    """
    def __init__(self, display=DISPLAY_OVERLAY, write_footage=False):
        super().__init__(display, write_footage)
        self._FF_PARAMS = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }

    def _apply_dense_optical_flow(self):
        previous_gray_frame = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous_gray_frame, gray_frame, None, **self._FF_PARAMS)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

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

        flow = self._apply_dense_optical_flow()
        threshold = self._apply_gaussian_threshold(flow)
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
                    cv2.cvtColor(flow, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(morphology, cv2.COLOR_GRAY2BGR),
                    frame_contours,
                    frame_detections
                ], labels=[
                    "Original", "Flow", "Gaussian threshold",
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
                cv2.imshow('Dense Optical Flow', resized)
                if cv2.waitKey(1) in [27, ord('q'), ord('Q')]:
                    exit()

        self.previous_frame = self.frame
        return [detection for detection in self._known_detections if detection._has_sufficient_confidence()]
