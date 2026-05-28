# src/hand_detector.py
import os
import urllib.request

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_DEFAULT_MODEL_DIR = os.path.expanduser("~/.cache/mediapipe/models")


def _ensure_model(path: str | None) -> str:
    """Return a valid model path, downloading default if needed."""
    if path is not None and os.path.isfile(path):
        return path
    os.makedirs(_DEFAULT_MODEL_DIR, exist_ok=True)
    dest = os.path.join(_DEFAULT_MODEL_DIR, "hand_landmarker.task")
    if not os.path.isfile(dest):
        print(f"Downloading hand landmarker model to {dest} ...")
        urllib.request.urlretrieve(_DEFAULT_MODEL_URL, dest)
        print("Download complete.")
    return dest


class HandDetector:
    INDEX_FINGER_TIP = 8

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
        model_path: str | None = None,
    ) -> None:
        resolved_path = _ensure_model(model_path)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, frame_rgb: np.ndarray) -> tuple[
        np.ndarray | None,  # left landmarks (21,3) or None
        np.ndarray | None,  # right landmarks (21,3) or None
        np.ndarray | None,  # left index tip (3,) or None
        np.ndarray | None,  # right index tip (3,) or None
    ]:
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += 33  # ~30 fps stepping
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        left_lm = None
        right_lm = None
        left_tip = None
        right_tip = None

        if result.hand_landmarks and result.handedness:
            for hand_lm_list, handedness_list in zip(
                result.hand_landmarks, result.handedness
            ):
                label = handedness_list[0].category_name
                lm = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lm_list]
                )
                tip = lm[self.INDEX_FINGER_TIP].copy()
                if label == "Left":
                    left_lm = lm
                    left_tip = tip
                else:
                    right_lm = lm
                    right_tip = tip

        return left_lm, right_lm, left_tip, right_tip

    def close(self) -> None:
        self._landmarker.close()
