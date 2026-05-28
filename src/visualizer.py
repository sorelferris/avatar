# src/visualizer.py
import cv2
import mujoco
import numpy as np


class Visualizer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._model = model
        self._data = data
        self._viewer = None
        self._left_target_site = self._find_site("left_target")
        self._right_target_site = self._find_site("right_target")

    def _find_site(self, name: str) -> int:
        try:
            return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
        except Exception:
            return -1

    def start_viewer(self) -> None:
        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    def update_viewer(self, left_target: np.ndarray | None, right_target: np.ndarray | None) -> None:
        if self._viewer is None:
            return
        if self._left_target_site >= 0 and left_target is not None:
            self._data.site_xpos[self._left_target_site] = left_target
        if self._right_target_site >= 0 and right_target is not None:
            self._data.site_xpos[self._right_target_site] = right_target
        self._viewer.sync()

    def draw_opencv(
        self,
        frame: np.ndarray,
        left_lm: np.ndarray | None,
        right_lm: np.ndarray | None,
        left_target: np.ndarray | None,
        right_target: np.ndarray | None,
        fps: float,
    ) -> np.ndarray:
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        if left_lm is not None:
            self._draw_hand(annotated, left_lm, (255, 0, 0), h, w)
        if right_lm is not None:
            self._draw_hand(annotated, right_lm, (0, 0, 255), h, w)

        y0 = 30
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if left_target is not None:
            txt = f"L: [{left_target[0]:.3f}, {left_target[1]:.3f}, {left_target[2]:.3f}]"
            cv2.putText(annotated, txt, (10, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        if right_target is not None:
            txt = f"R: [{right_target[0]:.3f}, {right_target[1]:.3f}, {right_target[2]:.3f}]"
            cv2.putText(annotated, txt, (10, y0 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return annotated

    def _draw_hand(self, frame, lm, color, h, w):
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17),
        ]
        for i, j in connections:
            pt1 = (int(lm[i][0] * w), int(lm[i][1] * h))
            pt2 = (int(lm[j][0] * w), int(lm[j][1] * h))
            cv2.line(frame, pt1, pt2, color, 2)
        tip_pt = (int(lm[8][0] * w), int(lm[8][1] * h))
        cv2.circle(frame, tip_pt, 8, (0, 0, 255), -1)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
