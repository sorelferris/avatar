import os

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

MP_HANDS = mp.solutions.hands
MP_DRAW = mp.solutions.drawing_utils
WINDOW_NAME = "Intel RealSense Gesture Recognition"
DEFAULT_DEVICE_ID = "243322072321"
IDEAL_DISTANCE_RANGE_MM = (
    300,
    1000,
)  # Ideal depth range for hand detection with Intel RealSense D435i


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
    def __init__(self, device_id=DEFAULT_DEVICE_ID, show_window=None):
        self.device_id = device_id
        if show_window is None:
            show_window = bool(os.environ.get("DISPLAY"))
        self.show_window = show_window
        self.hands_detector = None
        self.pipeline = None
        self.align = None

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

    def _draw_hand_overlay(self, color_image, hand_label, lines):
        image_w = color_image.shape[1]
        margin_x = 20
        start_y = 40
        line_gap = 34
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 255, 0)

        is_right = hand_label == "Right"
        for idx, text in enumerate(lines):
            y = start_y + idx * line_gap
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            if is_right:
                x = image_w - text_size[0] - margin_x
            else:
                x = margin_x

            cv2.putText(color_image, text, (x, y), font, font_scale, color, thickness)

    def _draw_distance_hint(self, color_image: np.ndarray, depth_mm: float | None):
        image_h, image_w = color_image.shape[:2]
        d_min, d_max = IDEAL_DISTANCE_RANGE_MM
        d_hint = f"Ideal distance: {d_min}-{d_max}mm"
        if depth_mm is None:
            hint_text = "Move hand into view"
            color = (0, 255, 255)
        else:
            if depth_mm < d_min:
                hint_text = f"Too close ({depth_mm:.0f}mm). {d_hint}"
                color = (0, 165, 255)
            elif depth_mm > d_max:
                hint_text = f"Too far ({depth_mm:.0f}mm). {d_hint}"
                color = (0, 165, 255)
            else:
                hint_text = f"Distance good ({depth_mm:.0f}mm). {d_hint}"
                color = (0, 255, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(hint_text, font, font_scale, thickness)

        x = max(20, (image_w - text_size[0]) // 2)
        y = image_h - 20

        box_top_left = (x - 10, y - text_size[1] - 10)
        box_bottom_right = (x + text_size[0] + 10, y + 10)
        cv2.rectangle(color_image, box_top_left, box_bottom_right, (0, 0, 0), -1)
        cv2.putText(color_image, hint_text, (x, y), font, font_scale, color, thickness)

    def _process_frame(self, color_frame, depth_frame):
        color_image = np.asarray(color_frame.get_data())  # BGR format
        color_image = cv2.flip(color_image, 1)  # provide as mirror view

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # RGB format
        # Process the RGB image with MediaPipe Hands to get hand landmarks and handedness
        result = self.hands_detector.process(rgb_image)
        # Process detected hands
        if result.multi_hand_landmarks and result.multi_handedness:
            # Separately process each detected hand and annotate the color image
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                landmarks = hand_landmarks.landmark  # list of 21 landmarks for one hand
                # Draw hand landmarks
                MP_DRAW.draw_landmarks(
                    image=color_image,
                    landmark_list=hand_landmarks,
                    connections=MP_HANDS.HAND_CONNECTIONS,
                )
                # Count fingers
                num_fingers = count_fingers(landmarks, hand_label)
                # Read wrist position and depth
                wrist_x, wrist_y = landmarks[0].x, landmarks[0].y
                dw, dh = depth_frame.get_width(), depth_frame.get_height()
                # Landmarks are produced on horizontally flipped color_image;
                # map x back to aligned (un-flipped) depth coordinates.
                x_view = int(wrist_x * dw)
                y_view = int(wrist_y * dh)
                x_view = max(0, min(x_view, dw - 1))
                y_view = max(0, min(y_view, dh - 1))

                x_depth = (dw - 1) - x_view
                y_depth = y_view
                depth_mm = depth_frame.get_distance(x_depth, y_depth) * 1000
                # Annotate hand info on the color image
                lines = [f"{hand_label} Hand", f"Fingers: {num_fingers}"]
                if depth_mm is not None:
                    lines.append(f"Depth: {depth_mm:.0f}mm")
                self._draw_hand_overlay(color_image, hand_label, lines)
                self._draw_distance_hint(color_image, depth_mm)

        # self._draw_distance_hint(color_image, hand_depths_mm)

        return color_image

    def run(self):
        if not self.show_window:
            print("No DISPLAY detected, running in headless mode.")

        self._setup_hands_detector()
        self._setup_realsense_pipeline()

        print(f"{self.__class__.__name__} started, press 'q' to exit...")
        try:
            while True:
                color_frame, depth_frame = self.get_aligned_frames()
                if color_frame is None or depth_frame is None:
                    continue

                color_image = self._process_frame(color_frame, depth_frame)

                if self.show_window:
                    cv2.imshow(WINDOW_NAME, color_image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            print("Exiting...")
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


if __name__ == "__main__":
    detector = HandDetector(device_id=DEFAULT_DEVICE_ID)
    detector.run()
