import os

# Avoid repeated Qt font warnings when OpenCV uses the Qt backend.
for _font_dir in (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/freefont",
    "/usr/share/fonts",
):
    if os.path.isdir(_font_dir):
        os.environ.setdefault("QT_QPA_FONTDIR", _font_dir)
        break

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# 1. 初始化 MediaPipe Hands (Solutions API)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
print("Creating MediaPipe Hands...")
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
print("MediaPipe Hands ready")

# 2. 初始化 RealSense 管道与配置
pipeline = rs.pipeline()
config = rs.config()
config.enable_device("243322072321")  # 替换为你的 RealSense 设备 ID
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
print("Starting pipeline...")
pipeline.start(config)
print("Pipeline started")
align = rs.align(rs.stream.color)

has_display = bool(os.environ.get("DISPLAY"))
if not has_display:
    print("No DISPLAY detected, running in headless mode (no cv2.imshow window).")


# 3. 核心手势识别函数：计算伸出的手指数 (0-5)
def count_fingers(hand_landmarks, handedness):
    finger_count = 0
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    if handedness == "Right":
        if hand_landmarks[tips[0]].x < hand_landmarks[pips[0]].x:
            finger_count += 1
    else:
        if hand_landmarks[tips[0]].x > hand_landmarks[pips[0]].x:
            finger_count += 1

    for i in range(1, 5):
        if hand_landmarks[tips[i]].y < hand_landmarks[pips[i]].y:
            finger_count += 1

    return finger_count


print("程序已启动，按 'q' 键退出...")

try:
    while True:
        ok, frames = pipeline.try_wait_for_frames(5000)
        if not ok:
            print(f"no frame after 5s")
            continue
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("missing frame")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.flip(color_image, 1)
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # 5. 使用 MediaPipe Hands 检测手部
        result = hands.process(rgb_image)

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                hand_label = handedness.classification[0].label
                hand_lm_list = hand_landmarks.landmark

                count = count_fingers(hand_lm_list, hand_label)

                mp_draw.draw_landmarks(
                    image=color_image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                )

                cv2.putText(
                    color_image,
                    f"Fingers: {count}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                )
                cv2.putText(
                    color_image,
                    f"Hand: {hand_label}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

                index_finger_tip = hand_lm_list[8]
                dw, dh = depth_frame.get_width(), depth_frame.get_height()
                # Landmarks are computed on horizontally flipped color_image;
                # map x back to aligned (unflipped) depth coordinates.
                pixel_x_view = int(index_finger_tip.x * dw)
                pixel_y_view = int(index_finger_tip.y * dh)
                pixel_x_view = max(0, min(pixel_x_view, dw - 1))
                pixel_y_view = max(0, min(pixel_y_view, dh - 1))

                depth_x = (dw - 1) - pixel_x_view
                depth_y = pixel_y_view
                depth_value = depth_frame.get_distance(depth_x, depth_y)
                if depth_value > 0:
                    depth_mm = depth_value * 1000.0
                    cv2.putText(
                        color_image,
                        f"Depth: {depth_mm:.0f}mm",
                        (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

        if has_display:
            cv2.imshow("RealSense Gesture Recognition", color_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    hands.close()
