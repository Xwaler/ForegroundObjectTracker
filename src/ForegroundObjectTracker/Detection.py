import cv2
import numpy as np


class Detection:
    """
    Detection actively corrected by a Kalman Filter
    Provides trajectory prediction
    """

    def __init__(self, rect):
        self.x, self.y, self.w, self.h = rect

        self._KALMAN_PROCESS_NOISE_COV = 1E-5
        self._KALMAN_MEASUREMENT_NOISE_COV = 1E-1
        self._KALMAN_ERROR_COV = 1E-1
        self._DETECTION_MIN_CONFIDENCE = 5
        self._DETECTION_TRAJECTORY_MIN_CONFIDENCE = 10
        self._DETECTION_FADING_RATE = 0.2

        self._confidence = 2.
        self._kalman = cv2.KalmanFilter(8, 4, 0)
        self._init_kalman()

    def _init_kalman(self):
        self._kalman.transitionMatrix = np.array([
            [1., 0., 0., 0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1., 0., 1.],
            [0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1.],
        ], dtype=np.float32)
        self._kalman.measurementMatrix = np.array([
            [1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.],
        ], dtype=np.float32)
        self._kalman.processNoiseCov = self._KALMAN_PROCESS_NOISE_COV * np.identity(8, dtype=np.float32)
        self._kalman.measurementNoiseCov = self._KALMAN_MEASUREMENT_NOISE_COV * np.identity(4, dtype=np.float32)
        self._kalman.errorCovPost = self._KALMAN_ERROR_COV * np.identity(8, dtype=np.float32)
        self._kalman.statePre = np.array([
            [self.x + .5 * self.w, self.y + .5 * self.h, self.w, self.h, 0., 0., 0., 0.],
        ], dtype=np.float32).T
        self._kalman.statePost = np.array([
            [self.x + .5 * self.w, self.y + .5 * self.h, self.w, self.h, 0., 0., 0., 0.],
        ], dtype=np.float32).T

    def _reinforce(self):
        self._confidence += 1

    def _fade(self):
        self._confidence = self._confidence * (1. - self._DETECTION_FADING_RATE)

    def _apply_homography(self, rot_scaling_mat, translation_vec, projection_vec, normalization):
        state_xy = self._kalman.statePost[0:2]
        self._kalman.statePost[0:2] = (
            state_xy_post := (
                    (rot_scaling_mat.dot(state_xy) + translation_vec) /
                    (projection_vec.dot(state_xy) + normalization)
            )
        )
        self.x, self.y = np.round(state_xy_post.ravel() - .5 * self.get_size()).astype(int)
        state_wh = self._kalman.statePost[2:4]
        self._kalman.statePost[2:4] = (
                (rot_scaling_mat.dot(state_wh)) /
                (projection_vec.dot(state_wh) + normalization)
        )

    def _correct_and_update_position(self, rect):
        x, y, w, h = rect
        measurement = np.array([[x + .5 * w], [y + .5 * h], [w], [h]], dtype=np.float32)
        self._kalman.predict()
        corrected_xy_wh = self._kalman.correct(measurement)[:4].ravel()
        self.x, self.y = np.round(corrected_xy_wh[:2] - .5 * corrected_xy_wh[2:4]).astype(int)
        self.w, self.h = np.round(corrected_xy_wh[2:4]).astype(int)

    def _has_sufficient_confidence(self):
        return np.ceil(self._confidence) >= self._DETECTION_MIN_CONFIDENCE

    def get_position(self) -> np.ndarray:
        """
        Get center position of the detection
        :return:
        """
        return np.round(np.array([self.x, self.y]) + .5 * self.get_size()).astype(int)

    def get_size(self) -> np.ndarray:
        """
        Get rect size of the detection
        :return:
        """
        return np.array([self.w, self.h])

    def get_rect(self) -> np.ndarray:
        """
        Get upper-left and lower-right corners of the detection's rect
        :return: [[x, y], [x, y]]
        """
        return np.array([[self.x, self.y], [self.x + self.w, self.y + self.h]])

    def predict(self, steps: int = 10) -> list[np.ndarray]:
        """
        Predict the detection's position, size, speed and acceleration for the future n steps
        :param steps:
        :return: Prediction in the form [[x, y, w, h, vx, vy, ax, ay], ...]
        """
        assert steps >= 0
        state = self._kalman.statePost.ravel()
        predictions = [state, ]
        for _ in range(steps):
            predictions.append(nextState := state.dot(self._kalman.transitionMatrix.T))
            state = nextState
        return predictions

    def draw(self, img: np.ndarray, rect_color: tuple[int, int, int], trajectory_color: tuple[int, int, int]) -> None:
        """
        Draw this detection's rectangle and trajectory
        :param img: destination image
        :param rect_color:
        :param trajectory_color:
        :return:
        """
        cv2.rectangle(img, *self.get_rect(), color=rect_color, thickness=2)

        if np.ceil(self._confidence) >= self._DETECTION_TRAJECTORY_MIN_CONFIDENCE:
            predictions = self.predict(steps=10)
            for i in range(len(predictions) - 1):
                state = predictions[i]
                nextState = predictions[i + 1]
                cv2.line(img, np.round(state[:2]).astype(int), np.round(nextState[:2]).astype(int),
                         trajectory_color, 2)

    def __repr__(self):
        return f'Detection(x={self.x}, y={self.y}, h={self.w}, w={self.h})'
