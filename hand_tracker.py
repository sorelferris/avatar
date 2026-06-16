import os
import threading

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

MP_HANDS = mp.solutions.hands
MP_DRAW = mp.solutions.drawing_utils
WINDOW_NAME = "Intel RealSense Gesture Recognition"
DEFAULT_DEVICE_ID = "243322072321"
IDEAL_DISTANCE_RANGE_MM = (300, 1000)


def count_fingers(hand_landmarks: list, handedness: str) -> int:
    """Count the number of fingers extended based on hand landmarks and handedness.
    The logic is based on the relative positions of the fingertip and pip joints.
    For the thumb, we compare x-coordinates (since it extends sideways),
    and for the other fingers, we compare y-coordinates (since they extend upwards).

    Args:
        hand_landmarks: List of 21 hand landmarks from MediaPipe.
        handedness: "Left" or "Right" indicating which hand is detected.
    Returns:
        The number of fingers extended (0 to 5).
    """
    assert len(hand_landmarks) == 21, "Expected 21 hand landmarks"
    assert handedness in ["Left", "Right"], "handedness must be 'Left' or 'Right'"

    finger_count = 0
    tips = [4, 8, 12, 16, 20]  # fingertip landmarks
    pips = [3, 6, 10, 14, 18]  # pip joint landmarks

    # Thumb logic: compare x-coordinates of tip and pip, direction depends on handedness
    if handedness == "Right":
        if hand_landmarks[tips[0]].x < hand_landmarks[pips[0]].x:
            finger_count += 1
    else:
        if hand_landmarks[tips[0]].x > hand_landmarks[pips[0]].x:
            finger_count += 1

    # For other fingers, compare y-coordinates of tip and pip
    for i in range(1, 5):
        if hand_landmarks[tips[i]].y < hand_landmarks[pips[i]].y:
            finger_count += 1

    return finger_count


def read_gesture(hand_landmarks: list, handedness: str) -> str:
    """Read the gesture from the hand landmarks and handedness."""
    num_fingers = count_fingers(hand_landmarks, handedness)
    # 🤛 Fist: 0 fingers extended
    if num_fingers == 0:
        return "Fist"
    # 👍 Thumbs Up: Only thumb extended
    if num_fingers == 1 and hand_landmarks[4].y < hand_landmarks[3].y:
        return "Thumbs Up"
    # ✋ Open Hand: All fingers extended
    if num_fingers == 5:
        return "Open Hand"
    # 👌 OK: 2 fingers extended and thumb tip touching index fingertip
    if (
        num_fingers == 2
        and hand_landmarks[4].y < hand_landmarks[3].y
        and abs(hand_landmarks[4].x - hand_landmarks[8].x) < 0.05
        and abs(hand_landmarks[4].y - hand_landmarks[8].y) < 0.05
    ):
        return "OK"
    return "Unknown"


class HandDetector:
    def __init__(self, device_id=DEFAULT_DEVICE_ID):
        self.device_id = device_id
        self.hands_detector = None
        self.pipeline = None
        self.align = None

        # 共享状态（主线程只读，后台线程写入）
        self.shared_status = {}   # {hand_label: {x, y, depth_mm, fingers, gesture}}
        self.shared_motion = {}   # {hand_label: {x, y, depth_mm}}
        self.motion_anchor = {}    # 内部使用，不暴露

    def _setup_hands_detector(self):
        print("Creating MediaPipe Hands...")
        self.hands_detector = MP_HANDS.Hands(
            static_image_mode=False,  # dealing with video stream
            max_num_hands=2,  # detect up to 2 hands
            model_complexity=1,  # moderate complexity for better accuracy
            min_detection_confidence=0.7,  # return detected hand landmarks only if confidence > 0.7
            min_tracking_confidence=0.7,  # return tracked hand landmarks only if confidence > 0.7
        )
        print("MediaPipe Hands ready")

    def _setup_realsense_pipeline(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.device_id)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        print("Starting pipeline...")
        self.pipeline.start(config)
        print("Pipeline started")
        self.align = rs.align(rs.stream.color)  # align depth to color

    def get_aligned_frames(self):
        ok, frames = self.pipeline.try_wait_for_frames(5000)
        if not ok:
            print("no frame after 5s")
            return None, None

        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("missing frame")
            return None, None

        return color_frame, depth_frame