# src/hand_detector.py
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import torch
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
_MIDAS_REPO_DIR = os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master")


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


def _ensure_midas_repo() -> str:
    """Return path to local MiDaS repo, cloning if needed."""
    if os.path.isdir(os.path.join(_MIDAS_REPO_DIR, "midas")):
        return _MIDAS_REPO_DIR
    import subprocess

    os.makedirs(os.path.dirname(_MIDAS_REPO_DIR), exist_ok=True)
    print(f"Cloning MiDaS repo to {_MIDAS_REPO_DIR} ...")
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/intel-isl/MiDaS.git",
            _MIDAS_REPO_DIR,
        ],
    )
    print("Clone complete.")
    return _MIDAS_REPO_DIR


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

        # MiDaS depth estimation
        midas_dir = _ensure_midas_repo()
        self._midas = torch.hub.load(
            midas_dir, "MiDaS_small", source="local", trust_repo=True
        )
        self._midas.eval()
        self._midas_transforms = torch.hub.load(
            midas_dir, "transforms", source="local", trust_repo=True
        )
        self._midas_transform = self._midas_transforms.small_transform
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._midas.to(self._device)

    def detect(
        self, frame_rgb: np.ndarray, frame_bgr: np.ndarray | None = None
    ) -> tuple[
        np.ndarray | None,  # left landmarks (21,3) or None
        np.ndarray | None,  # right landmarks (21,3) or None
        np.ndarray | None,  # left index tip 3D (3,) with MiDaS depth
        np.ndarray | None,  # right index tip 3D (3,) with MiDaS depth
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
                lm = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm_list])
                tip = lm[self.INDEX_FINGER_TIP].copy()
                if label == "Left":
                    left_lm = lm
                    left_tip = tip
                else:
                    right_lm = lm
                    right_tip = tip

        # Upgrade tips to 3D with MiDaS depth when BGR frame is provided
        if left_tip is not None and frame_bgr is not None:
            left_tip = self.get_index_tip_3d(left_tip, frame_bgr)
        if right_tip is not None and frame_bgr is not None:
            right_tip = self.get_index_tip_3d(right_tip, frame_bgr)

        return left_lm, right_lm, left_tip, right_tip

    def estimate_depth(self, frame_bgr: np.ndarray, tip_px: tuple[int, int]) -> float:
        """Estimate depth at pixel coordinate using MiDaS.
        Returns normalized depth value (higher = closer in MiDaS convention).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self._midas_transform(rgb).to(self._device)
        with torch.no_grad():
            prediction = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        # Normalize to [0, 1]
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        # Sample at tip pixel
        x, y = tip_px
        x = max(0, min(x, depth_map.shape[1] - 1))
        y = max(0, min(y, depth_map.shape[0] - 1))
        return float(depth_map[y, x])

    def get_index_tip_3d(
        self,
        tip_normalized: np.ndarray,
        frame_bgr: np.ndarray,
        palm_width_pixels: float | None = None,
    ) -> np.ndarray:
        """Get 3D position of index finger tip with MiDaS depth.
        Returns (x, y, z) in normalized coordinates with z from MiDaS.
        """
        h, w = frame_bgr.shape[:2]
        tip_px = (int(tip_normalized[0] * w), int(tip_normalized[1] * h))
        midas_depth = self.estimate_depth(frame_bgr, tip_px)
        # Scale depth with palm width correction if available
        reference_palm_width = 80.0  # pixels, default
        if palm_width_pixels is not None and palm_width_pixels > 0:
            depth_scale = reference_palm_width / palm_width_pixels
        else:
            depth_scale = 1.0
        z = midas_depth * depth_scale
        return np.array([tip_normalized[0], tip_normalized[1], z])

    def detect_gesture(self, landmarks: np.ndarray | None) -> str | None:
        """Detect hand gesture from landmarks.

        Returns:
            "fist"  — all fingers curled (gripper closes)
            "open"  — fingers extended (gripper opens)
            "palm_closed" — all fingers curled tight toward palm center (reset signal)
            None    — no landmarks

        Detection based on fingertip-to-mcp distances relative to palm size.
        All fingertips close to their MCPs = fist; at least one extended = open;
        all fingertips close to palm center = palm_closed (reset trigger).
        """
        if landmarks is None:
            return None

        # Landmark indices
        MCP = [1, 4, 7, 11]   # MCP joints of thumb, index, middle, ring
        TIP = [4, 8, 12, 16]  # Corresponding fingertips
        PALM = 0              # Wrist base

        # Compute reference palm size (distance from wrist to middle finger MCP)
        palm_size = np.linalg.norm(landmarks[PALM] - landmarks[9])
        if palm_size < 1e-6:
            return None

        # Check if all fingertips are curled toward their MCPs
        all_fingers_closed = True
        for mcp_idx, tip_idx in zip(MCP, TIP):
            dist = np.linalg.norm(landmarks[tip_idx] - landmarks[mcp_idx])
            if dist > 0.5 * palm_size:
                all_fingers_closed = False
                break

        if all_fingers_closed:
            return "fist"

        # Check palm_closed: all fingertips close to palm center
        palm_center = np.mean([landmarks[i] for i in [5, 9, 13, 17]], axis=0)
        avg_tip_dist = np.mean(
            [np.linalg.norm(landmarks[t] - palm_center) for t in TIP]
        )
        if avg_tip_dist < 0.3 * palm_size:
            return "palm_closed"

        return "open"

    def close(self) -> None:
        self._landmarker.close()
