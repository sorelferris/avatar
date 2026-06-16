# RealSense 手势控制机械臂实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `realsense_demo.py` 的手势识别改造为线程化 `HandDetector`，在 `main_realsense.py` 中通过 polling 共享状态控制机械臂末端位置，仅在五指张开时启用。

**架构：** 后台线程运行 RealSense + MediaPipe Hands，主循环 polling shared_status / shared_motion，通过 IK 控制双臂。

**技术栈：** pyrealsense2, mediapipe, numpy, skrobot, threading

---

## 文件职责

| 文件 | 职责 |
|---|---|
| `hand_tracker.py`（新建） | 从 `realsense_demo.py` 提取并改造为线程化 `HandDetector`，暴露 `shared_status` / `shared_motion` |
| `main_realsense.py`（修改） | 集成 `HandDetector` 后台线程，polling `shared_motion` 控制双臂 |

---

## 任务 1：创建 hand_tracker.py — 基础结构

**文件：**
- 创建：`/home/sorel/workspace/avatar/hand_tracker.py`

- [ ] **步骤 1：编写文件头部和导入**

```python
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
```

- [ ] **步骤 2：编写 count_fingers 和 read_gesture 辅助函数**

从 `realsense_demo.py:18-73` 复制，不做修改。

- [ ] **步骤 3：编写 HandDetector 类基础框架**

```python
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
```

- [ ] **步骤 4：实现 _setup_hands_detector 和 _setup_realsense_pipeline**

从 `realsense_demo.py:89-108` 复制，print 语句保持。

- [ ] **步骤 5：实现 get_aligned_frames**

从 `realsense_demo.py:111-125` 复制，不做修改。

- [ ] **步骤 6：Commit**

```bash
git add hand_tracker.py
git commit -m "feat(hand_tracker): add HandDetector class foundation
- copy count_fingers/read_gesture from realsense_demo.py
- add shared_status and shared_motion dicts for thread-safe polling"
```

---

## 任务 2：为 hand_tracker.py 添加稳定性等待和锚点逻辑

**文件：**
- 修改：`/home/sorel/workspace/avatar/hand_tracker.py`

- [ ] **步骤 1：在 __init__ 中添加稳定性计数器**

```python
self._stable_frames = {}  # {hand_label: count of consecutive 5-finger frames}
self._stable_threshold = 3  # 连续 3 帧
```

- [ ] **步骤 2：添加 _compute_gain 函数（可变增益）**

```python
def _compute_gain(self, displacement: float) -> float:
    """位移越大，增益越高，非线性映射"""
    return 0.0005 + abs(displacement) * 0.0001
```

- [ ] **步骤 3：重写 _process_frame 实现完整逻辑**

核心逻辑：
1. 检测到手部 → count_fingers + read_gesture
2. num_fingers == 5 且 depth_mm > 100 → 稳定性计数+1
3. 稳定性计数 >= 3 → 更新 anchor，计算 motion
4. 否则 → 清除 anchor，motion 归零
5. num_fingers != 5 → 清除 anchor，motion 归零

```python
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
```

- [ ] **步骤 4：添加 _sample_depth_mm 和 _draw_* 方法**

从 `realsense_demo.py:157-194` 复制 `_sample_depth_mm`、`_draw_hand_overlay`、`_draw_distance_hint` 到 `hand_tracker.py`。

- [ ] **步骤 5：实现 run 方法（后台线程入口）**

```python
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
```

- [ ] **步骤 6：实现 close 方法**

从 `realsense_demo.py:336-343` 复制，不做修改。

- [ ] **步骤 7：Commit**

```bash
git add hand_tracker.py
git commit -m "feat(hand_tracker): add stability waiting, anchor reset, and variable gain logic"
```

---

## 任务 3：修改 main_realsense.py — 集成 HandDetector 后台线程

**文件：**
- 修改：`/home/sorel/workspace/avatar/main_realsense.py`

- [ ] **步骤 1：添加导入**

```python
import time

import numpy as np
from hand_tracker import HandDetector  # 新增
from skrobot import model
from skrobot.coordinates import Coordinates
from skrobot.viewers import ViserViewer
```

- [ ] **步骤 2：修改 move_arm 函数以支持 motion dict 输入**

```python
def move_arm(robot, eef, arm, motion: dict):
    dx = motion.get("x", 0)
    dy = motion.get("y", 0)
    dz = motion.get("depth_mm", 0)

    if dx == 0 and dy == 0 and dz == 0:
        return None
    target_pos = eef.worldpos() + np.array([dx, dy, dz])
    target_rot = [0, 0, 0]
    result = robot.inverse_kinematics(
        Coordinates(pos=target_pos.tolist(), rot=target_rot),
        joint_list=arm,
        move_target=eef,
        rotation_mask=False,
        position_mask=True,
    )
    return result
```

- [ ] **步骤 3：重写 main 函数**

```python
def main() -> None:
    # Load robot model
    robot = model.RobotModel.from_urdf(URDF)
    robot.torso_joint.joint_angle(0.75)

    for joint in robot.joint_list:
        print(f"{joint.name}: {joint.joint_angle():+.3f}, limits=[{joint.min_angle:+.3f}, {joint.max_angle:+.3f}]")

    # 获取左右臂末端和关节列表
    larm_eef = next(x for x in robot.link_list if x.name == "left_arm_link7")
    rarm_eef = next(x for x in robot.link_list if x.name == "right_arm_link7")
    larm_joints = [x for x in robot.joint_list if x.name.startswith("left_arm_joint")]
    rarm_joints = [x for x in robot.joint_list if x.name.startswith("right_arm_joint")]

    viewer = ViserViewer()
    viewer.add(robot)
    viewer.redraw()

    # 启动 HandDetector 后台线程
    hand_detector = HandDetector()
    detector_thread = threading.Thread(target=hand_detector.run, daemon=True)
    detector_thread.start()
    print("HandDetector thread started.")

    print("Ready. Show open hand to control arm.")

    try:
        while True:
            # polling shared_motion
            left_motion = hand_detector.shared_motion.get("Left", {"x": 0, "y": 0, "depth_mm": 0})
            right_motion = hand_detector.shared_motion.get("Right", {"x": 0, "y": 0, "depth_mm": 0})

            move_arm(robot, larm_eef, larm_joints, left_motion)
            move_arm(robot, rarm_eef, rarm_joints, right_motion)

            viewer.redraw()
            time.sleep(0.033)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        hand_detector.close()
        for joint in robot.joint_list:
            joint.joint_angle(0.0)
        robot.torso_joint.joint_angle(0.75)
```

- [ ] **步骤 4：Commit**

```bash
git add main_realsense.py
git commit -m "feat(main_realsense): integrate HandDetector thread for gesture-based arm control"
```

---

## 规格自检

| 规格条目 | 实现位置 |
|---|---|
| 左手控左臂，右手控右臂 | `main_realsense.py: main()` — polling 时区分 Left/Right |
| 仅五指张开时启用控制 | `hand_tracker.py: _process_frame()` — 稳定性等待逻辑 |
| 连续 3 帧 + depth > 100mm 才设锚点 | `hand_tracker.py: _process_frame()` — `_stable_threshold >= 3` |
| 可变增益 | `hand_tracker.py: _compute_gain()` |
| 手部丢失时清除锚点 | `hand_tracker.py: _process_frame()` — else 分支 |
| 共享状态 shared_status / shared_motion | `hand_tracker.py: __init__` — `self.shared_*` |
| RealSense 5s 阻塞不影响主循环 | 后台线程架构保证 |
| STEP_SIZE 逻辑移除，增益在 motion 中 | `main_realsense.py: move_arm()` — 直接使用 dx/dy/dz |

---

## 执行选项

**计划已完成并保存到 `docs/superpowers/plans/2026-06-16-realsense-hand-tracking-arm-control-plan.md`。两种执行方式：**

**1. 子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代

**2. 内联执行** - 在当前会话中使用 executing-plans 执行任务，批量执行并设有检查点

**选哪种方式？**
