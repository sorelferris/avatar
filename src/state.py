import threading

import numpy as np


class TeleopState:
    """Thread-safe shared state for hand teleoperation.

    Accessed concurrently by camera thread, detection thread, and main loop.
    All reads and writes go through a lock. ``get_left`` / ``get_right`` return
    copies so callers never hold a reference that can be mutated by another
    thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._left_target = np.zeros(3)
        self._right_target = np.zeros(3)
        self._left_detected = False
        self._right_detected = False
        self.fps = 0.0

    # -- public read-only properties (lock-protected) ----------------------

    @property
    def left_hand_detected(self) -> bool:
        with self._lock:
            return self._left_detected

    @property
    def right_hand_detected(self) -> bool:
        with self._lock:
            return self._right_detected

    @property
    def left_target(self) -> np.ndarray:
        with self._lock:
            return self._left_target.copy()

    @property
    def right_target(self) -> np.ndarray:
        with self._lock:
            return self._right_target.copy()

    # -- mutation methods --------------------------------------------------

    def update_left(self, target: np.ndarray | None, detected: bool) -> None:
        with self._lock:
            self._left_detected = detected
            if target is not None:
                self._left_target = target.copy()

    def update_right(self, target: np.ndarray | None, detected: bool) -> None:
        with self._lock:
            self._right_detected = detected
            if target is not None:
                self._right_target = target.copy()

    def get_left(self) -> tuple[np.ndarray, bool]:
        with self._lock:
            return self._left_target.copy(), self._left_detected

    def get_right(self) -> tuple[np.ndarray, bool]:
        with self._lock:
            return self._right_target.copy(), self._right_detected
