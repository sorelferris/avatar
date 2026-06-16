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

        # 稳定性等待机制
        self._stable_frames = {}  # {hand_label: count of consecutive 5-finger frames}
        self._stable_threshold = 3  # 连续 3 帧

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

    def _compute_gain(self, displacement: float) -> float:
        """位移越大，增益越高，非线性映射"""
        return 0.0005 + abs(displacement) * 0.0001

    def _sample_depth_mm(
        self, depth_frame, x_depth: int, y_depth: int, radius: int = 2
    ):
        dw, dh = depth_frame.get_width(), depth_frame.get_height()
        x0 = max(0, x_depth - radius)
        x1 = min(dw - 1, x_depth + radius)
        y0 = max(0, y_depth - radius)
        y1 = min(dh - 1, y_depth + radius)

        valid_depths = []
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                depth_m = depth_frame.get_distance(xx, yy)
                if depth_m > 0:
                    valid_depths.append(depth_m * 1000.0)

        if not valid_depths:
            return 0
        return int(np.median(valid_depths))

    def _process_frame(self, color_frame, depth_frame):
        color_image = np.asarray(color_frame.get_data())
        color_image = cv2.flip(color_image, 1)
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        result = self.hands_detector.process(rgb_image)

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                hand_label = handedness.classification[0].label
                landmarks = hand_landmarks.landmark

                # 计算手腕位置和深度
                wrist_x, wrist_y = landmarks[0].x, landmarks[0].y
                dw, dh = depth_frame.get_width(), depth_frame.get_height()
                x_view = int(wrist_x * dw)
                y_view = int(wrist_y * dh)
                x_view = max(0, min(x_view, dw - 1))
                y_view = max(0, min(y_view, dh - 1))
                x_depth = (dw - 1) - x_view
                y_depth = y_view
                depth_mm = self._sample_depth_mm(depth_frame, x_depth, y_depth)

                num_fingers = count_fingers(landmarks, hand_label)
                gesture = read_gesture(landmarks, hand_label)

                # 更新 shared_status
                self.shared_status[hand_label] = {
                    "x": x_view,
                    "y": y_view,
                    "depth_mm": depth_mm,
                    "fingers": num_fingers,
                    "gesture": gesture,
                }

                # 手势使能逻辑
                if num_fingers == 5 and depth_mm > 100:
                    # 稳定性等待
                    self._stable_frames[hand_label] = self._stable_frames.get(hand_label, 0) + 1

                    if self._stable_frames[hand_label] >= self._stable_threshold:
                        # 更新锚点并计算 motion
                        if self.motion_anchor.get(hand_label) is not None:
                            anchor = self.motion_anchor[hand_label]
                            raw_dx = x_view - anchor["x"]
                            raw_dy = y_view - anchor["y"]
                            raw_ddepth = depth_mm - anchor["depth_mm"]
                            self.shared_motion[hand_label] = {
                                "x": raw_dx * self._compute_gain(raw_dx),
                                "y": raw_dy * self._compute_gain(raw_dy),
                                "depth_mm": raw_ddepth * self._compute_gain(raw_ddepth),
                            }
                        else:
                            self.shared_motion[hand_label] = {"x": 0, "y": 0, "depth_mm": 0}

                        # 更新锚点
                        self.motion_anchor[hand_label] = {
                            "x": x_view,
                            "y": y_view,
                            "depth_mm": depth_mm,
                        }
                    else:
                        # 稳定性计数未达标，motion 归零
                        self.shared_motion[hand_label] = {"x": 0, "y": 0, "depth_mm": 0}
                else:
                    # 非五指张开，清除锚点和 motion
                    self._stable_frames[hand_label] = 0
                    self.motion_anchor[hand_label] = None
                    self.shared_motion[hand_label] = {"x": 0, "y": 0, "depth_mm": 0}
        else:
            # 无手部检测，清除所有状态
            self.shared_motion = {}
            self.shared_status = {}
            self.motion_anchor = {}
            self._stable_frames = {}

        return color_image

    def run(self):
        print("Creating MediaPipe Hands...")
        self._setup_hands_detector()
        self._setup_realsense_pipeline()

        print(f"{self.__class__.__name__} started, press Ctrl+C to exit...")
        try:
            while True:
                color_frame, depth_frame = self.get_aligned_frames()
                if color_frame is None or depth_frame is None:
                    continue
                self._process_frame(color_frame, depth_frame)
        except KeyboardInterrupt:
            print("HandDetector thread exiting...")
        finally:
            self.close()

    def close(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        cv2.destroyAllWindows()
        if self.hands_detector is not None:
            self.hands_detector.close()
            self.hands_detector = None