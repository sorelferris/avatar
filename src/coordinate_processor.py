# src/coordinate_processor.py
import numpy as np


def remap_camera_to_robot(cam_coords: np.ndarray) -> np.ndarray:
    """Remap camera frame (X right, Y down, Z forward) to robot frame (X forward, Y left, Z up)."""
    return np.array([cam_coords[2], -cam_coords[0], -cam_coords[1]])


class LinearScaler:
    def __init__(self, hand_range: float = 0.3, robot_range: float = 0.4) -> None:
        self._scale = robot_range / hand_range
        self._anchor = np.zeros(3)

    def set_anchor(self, hand_pos_robot_frame: np.ndarray) -> None:
        self._anchor = hand_pos_robot_frame.copy()

    def scale(self, hand_pos_robot_frame: np.ndarray) -> np.ndarray:
        return (hand_pos_robot_frame - self._anchor) * self._scale
