import cv2
import numpy as np


class Detection:
    """
    Detection actively corrected by a Kalman Filter
    Provides trajectory prediction
    """

    def __init__(self, rect):
        self.KALMAN_PROCESS_NOISE_COV = 1E-5
        self.KALMAN_MEASUREMENT_NOISE_COV = 1E-1
        self.KALMAN_ERROR_COV = 1E-1
        self.DETECTION_MIN_CONFIDENCE = 5
        self.DETECTION_TRAJECTORY_MIN_CONFIDENCE = 10
        self.DETECTION_FADING_RATE = 0.2

        self.x, self.y, self.w, self.h = rect
        self.confidence = 2.
        self.kalman = cv2.KalmanFilter(8, 4, 0)

        self.init_kalman()

    def init_kalman(self):
        self.kalman.transitionMatrix = np.array([
            [1., 0., 0., 0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1., 0., 1.],
            [0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1.],
        ], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([
            [1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.],
        ], dtype=np.float32)
        self.kalman.processNoiseCov = self.KALMAN_PROCESS_NOISE_COV * np.identity(8, dtype=np.float32)
        self.kalman.measurementNoiseCov = self.KALMAN_MEASUREMENT_NOISE_COV * np.identity(4, dtype=np.float32)
        self.kalman.errorCovPost = self.KALMAN_ERROR_COV * np.identity(8, dtype=np.float32)
        self.kalman.statePre = np.array([
            [self.x + .5 * self.w, self.y + .5 * self.h, self.w, self.h, 0., 0., 0., 0.],
        ], dtype=np.float32).T
        self.kalman.statePost = np.array([
            [self.x + .5 * self.w, self.y + .5 * self.h, self.w, self.h, 0., 0., 0., 0.],
        ], dtype=np.float32).T

    def reinforce(self):
        self.confidence += 1

    def fade(self):
        self.confidence = self.confidence * (1. - self.DETECTION_FADING_RATE)

    def apply_homography(self, rot_scaling_mat, translation_vec, projection_vec, normalization):
        state_xy = self.kalman.statePost[0:2]
        self.kalman.statePost[0:2] = (
            state_xy_post := (
                    (rot_scaling_mat.dot(state_xy) + translation_vec) /
                    (projection_vec.dot(state_xy) + normalization)
            )
        )
        self.x, self.y = np.round(state_xy_post.ravel() - .5 * np.array([self.w, self.h])).astype(int)
        state_wh = self.kalman.statePost[2:4]
        self.kalman.statePost[2:4] = (
                (rot_scaling_mat.dot(state_wh)) /
                (projection_vec.dot(state_wh) + normalization)
        )

    def correct_and_update_position(self, rect):
        x, y, w, h = rect
        measurement = np.array([[x + .5 * w], [y + .5 * h], [w], [h]], dtype=np.float32)
        self.kalman.predict()
        corrected_xy_wh = self.kalman.correct(measurement)[:4].ravel()
        self.x, self.y = np.round(corrected_xy_wh[:2] - .5 * corrected_xy_wh[2:4]).astype(int)
        self.w, self.h = np.round(corrected_xy_wh[2:4]).astype(int)

    def has_sufficient_confidence(self):
        return np.ceil(self.confidence) >= self.DETECTION_MIN_CONFIDENCE

    def draw(self, img, rect_color, trajectory_color):
        if self.has_sufficient_confidence():
            cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), rect_color, 2)

            if np.ceil(self.confidence) >= self.DETECTION_TRAJECTORY_MIN_CONFIDENCE:
                state = self.kalman.statePost.ravel()
                for _ in range(10):
                    nextState = state.dot(self.kalman.transitionMatrix.T)
                    cv2.line(img, np.round(state[:2]).astype(int), np.round(nextState[:2]).astype(int),
                             trajectory_color, 2)
                    state = nextState

    def __repr__(self):
        return f'Object<Detection> (x:{self.x} y:{self.y} h:{self.w} w:{self.h})'
