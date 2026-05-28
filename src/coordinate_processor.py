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


class EMAFilter:
    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha
        self._state: np.ndarray | None = None

    def update(self, value: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = value.copy()
        else:
            self._state = self._alpha * value + (1 - self._alpha) * self._state
        return self._state.copy()

    def reset(self) -> None:
        self._state = None


def clip_to_workspace(target: np.ndarray, max_radius: float) -> np.ndarray:
    norm = np.linalg.norm(target)
    if norm <= max_radius:
        return target.copy()
    return target * (max_radius / norm)


class CoordinateProcessor:
    def __init__(
        self,
        hand_range: float = 0.3,
        robot_range: float = 0.4,
        ema_alpha: float = 0.3,
        max_radius: float = 0.36,
    ) -> None:
        self._scaler = LinearScaler(hand_range, robot_range)
        self._ema = EMAFilter(ema_alpha)
        self._max_radius = max_radius
        self._last_target = np.zeros(3)

    def set_anchor(self, hand_pos: np.ndarray) -> None:
        """Set anchor from raw camera-frame hand position."""
        robot_pos = remap_camera_to_robot(hand_pos)
        self._scaler.set_anchor(robot_pos)

    def process(
        self, hand_pos: np.ndarray | None, detected: bool
    ) -> tuple[np.ndarray, bool]:
        if not detected or hand_pos is None:
            return self._last_target.copy(), False
        robot_pos = remap_camera_to_robot(hand_pos)
        scaled = self._scaler.scale(robot_pos)
        smoothed = self._ema.update(scaled)
        clipped = clip_to_workspace(smoothed, self._max_radius)
        self._last_target = clipped.copy()
        return clipped, True
