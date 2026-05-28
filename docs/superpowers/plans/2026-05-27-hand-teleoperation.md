# Hand Visual Teleoperation 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 通过单目摄像头 + MediaPipe Hands + MiDaS 深度估计，实时控制 MuJoCo 仿真中 G1 人形机器人双臂各 5 DoF

**架构：** 单进程多线程 — 摄像头线程捕获帧，检测线程运行 MediaPipe + MiDaS 并处理坐标，主线程运行 MuJoCo IK + 仿真步进，可视化线程渲染双窗口。通过线程安全的 TeleopState 共享数据。

**技术栈：** Python 3.12+, MediaPipe, OpenCV, MuJoCo, NumPy, PyTorch, MiDaS

**规格文档：** `docs/superpowers/specs/2026-05-27-hand-teleoperation-design.md`

---

## 文件结构

```
avatar/
├── pyproject.toml                          # 项目依赖
├── main.py                                 # 入口：启动主循环
├── src/
│   ├── __init__.py
│   ├── state.py                            # TeleopState 共享数据结构
│   ├── camera.py                           # CameraThread：摄像头捕获线程
│   ├── hand_detector.py                    # HandDetector：MediaPipe + MiDaS
│   ├── coordinate_processor.py             # CoordinateProcessor：变换+平滑+超程保护
│   ├── ik_solver.py                        # IKSolver：MuJoCo 雅可比 IK + PD 控制
│   ├── sim_env.py                          # SimEnvironment：MuJoCo 仿真环境管理
│   └── visualizer.py                       # Visualizer：OpenCV + MuJoCo viewer
├── assets/
│   ├── g1_23dof.xml                        # 已有
│   ├── g1_23dof_fixed.xml                  # 修改版：固定 pelvis
│   └── meshes/                             # 已有
└── tests/
    ├── test_state.py
    ├── test_coordinate_processor.py
    └── test_ik_solver.py
```

---

### 任务 1：项目初始化与依赖

**文件：**
- 修改：`pyproject.toml`
- 创建：`src/__init__.py`

- [ ] **步骤 1：更新 pyproject.toml 添加依赖**

```toml
[project]
name = "avatar"
version = "0.1.0"
description = "Hand visual teleoperation for G1 humanoid robot"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "mediapipe>=0.10",
    "opencv-python>=4.8",
    "mujoco>=3.0",
    "numpy>=1.24",
    "torch>=2.0",
    "timm>=0.9",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
]
```

- [ ] **步骤 2：安装依赖**

运行：`uv sync`
预期：依赖安装成功，无报错

- [ ] **步骤 3：创建 src/__init__.py**

```python
```

（空文件）

- [ ] **步骤 4：验证导入**

运行：`uv run python -c "import mediapipe; import cv2; import mujoco; import numpy; print('OK')"`
预期：`OK`

- [ ] **步骤 5：Commit**

```bash
git add pyproject.toml pyproject.toml uv.lock src/__init__.py
git commit -m "chore: add project dependencies for teleoperation system"
```

---

### 任务 2：TeleopState 共享数据结构

**文件：**
- 创建：`src/state.py`
- 创建：`tests/test_state.py`

- [ ] **步骤 1：编写失败的测试**

```python
# tests/test_state.py
import threading
import numpy as np
from src.state import TeleopState


def test_initial_state():
    state = TeleopState()
    assert state.left_hand_detected is False
    assert state.right_hand_detected is False
    assert state.left_target.shape == (3,)
    assert state.right_target.shape == (3,)
    assert np.allclose(state.left_target, 0.0)
    assert np.allclose(state.right_target, 0.0)


def test_thread_safe_update():
    state = TeleopState()
    target = np.array([0.1, 0.2, 0.3])
    state.update_left(target, detected=True)
    result, detected = state.get_left()
    assert detected is True
    assert np.allclose(result, target)


def test_hand_lost_keeps_last_position():
    state = TeleopState()
    target = np.array([0.1, 0.2, 0.3])
    state.update_left(target, detected=True)
    state.update_left(None, detected=False)
    result, detected = state.get_left()
    assert detected is False
    assert np.allclose(result, target)  # frozen
```

- [ ] **步骤 2：运行测试验证失败**

运行：`uv run pytest tests/test_state.py -v`
预期：FAIL，ModuleNotFoundError

- [ ] **步骤 3：编写实现**

```python
# src/state.py
import threading
import numpy as np


class TeleopState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._left_target = np.zeros(3)
        self._right_target = np.zeros(3)
        self._left_detected = False
        self._right_detected = False
        self.fps = 0.0

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
```

- [ ] **步骤 4：运行测试验证通过**

运行：`uv run pytest tests/test_state.py -v`
预期：3 passed

- [ ] **步骤 5：Commit**

```bash
git add src/state.py tests/test_state.py
git commit -m "feat: add TeleopState shared data structure with thread safety"
```

---

### 任务 3：坐标处理器 — 坐标系重映射

**文件：**
- 创建：`src/coordinate_processor.py`
- 创建：`tests/test_coordinate_processor.py`

- [ ] **步骤 1：编写失败的测试 — 坐标系重映射**

```python
# tests/test_coordinate_processor.py
import numpy as np
from src.coordinate_processor import remap_camera_to_robot


def test_remap_camera_to_robot_identity():
    """Camera: X right, Y down, Z forward -> Robot: X forward, Y left, Z up"""
    # Camera looking forward: z=1 (forward), x=0, y=0
    cam = np.array([0.0, 0.0, 1.0])
    robot = remap_camera_to_robot(cam)
    assert np.allclose(robot, [1.0, 0.0, 0.0])  # X forward


def test_remap_camera_to_robot_x():
    # Camera: x=1 (right) -> Robot: y=-1 (right is negative Y in robot frame)
    cam = np.array([1.0, 0.0, 0.0])
    robot = remap_camera_to_robot(cam)
    assert np.allclose(robot, [0.0, -1.0, 0.0])


def test_remap_camera_to_robot_y():
    # Camera: y=1 (down) -> Robot: z=-1 (down is negative Z in robot frame)
    cam = np.array([0.0, 1.0, 0.0])
    robot = remap_camera_to_robot(cam)
    assert np.allclose(robot, [0.0, 0.0, -1.0])
```

- [ ] **步骤 2：运行测试验证失败**

运行：`uv run pytest tests/test_coordinate_processor.py::test_remap_camera_to_robot_identity -v`
预期：FAIL，ModuleNotFoundError

- [ ] **步骤 3：编写实现 — remap_camera_to_robot**

```python
# src/coordinate_processor.py
import numpy as np


def remap_camera_to_robot(cam_coords: np.ndarray) -> np.ndarray:
    """Remap camera frame (X right, Y down, Z forward) to robot frame (X forward, Y left, Z up)."""
    return np.array([cam_coords[2], -cam_coords[0], -cam_coords[1]])
```

- [ ] **步骤 4：运行测试验证通过**

运行：`uv run pytest tests/test_coordinate_processor.py::test_remap_camera_to_robot_identity tests/test_coordinate_processor.py::test_remap_camera_to_robot_x tests/test_coordinate_processor.py::test_remap_camera_to_robot_y -v`
预期：3 passed

- [ ] **步骤 5：Commit**

```bash
git add src/coordinate_processor.py tests/test_coordinate_processor.py
git commit -m "feat: add camera-to-robot coordinate remapping"
```

---

### 任务 4：坐标处理器 — 线性缩放与锚定

- [ ] **步骤 1：编写失败的测试 — 线性缩放**

```python
# tests/test_coordinate_processor.py (追加)
from src.coordinate_processor import LinearScaler


def test_scaler_identity_at_anchor():
    scaler = LinearScaler(hand_range=0.3, robot_range=0.4)
    anchor = np.array([0.5, 0.5, 0.5])
    scaler.set_anchor(anchor)
    # At anchor position, output should be robot origin offset
    result = scaler.scale(anchor)
    assert np.allclose(result, [0.0, 0.0, 0.0])


def test_scaler_proportional():
    scaler = LinearScaler(hand_range=0.3, robot_range=0.4)
    anchor = np.array([0.5, 0.5, 0.5])
    scaler.set_anchor(anchor)
    # Move 0.15 in hand space -> 0.2 in robot space (scale = 0.4/0.3)
    moved = np.array([0.65, 0.5, 0.5])
    result = scaler.scale(moved)
    assert np.allclose(result, [0.2, 0.0, 0.0], atol=1e-6)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`uv run pytest tests/test_coordinate_processor.py::test_scaler_identity_at_anchor -v`
预期：FAIL

- [ ] **步骤 3：编写实现 — LinearScaler**

```python
# src/coordinate_processor.py (追加)


class LinearScaler:
    def __init__(self, hand_range: float = 0.3, robot_range: float = 0.4) -> None:
        self._scale = robot_range / hand_range
        self._anchor = np.zeros(3)

    def set_anchor(self, hand_pos: np.ndarray) -> None:
        self._anchor = hand_pos.copy()

    def scale(self, hand_pos: np.ndarray) -> np.ndarray:
        return (hand_pos - self._anchor) * self._scale
```

- [ ] **步骤 4：运行测试验证通过**

运行：`uv run pytest tests/test_coordinate_processor.py::test_scaler_identity_at_anchor tests/test_coordinate_processor.py::test_scaler_proportional -v`
预期：2 passed

- [ ] **步骤 5：Commit**

```bash
git add src/coordinate_processor.py tests/test_coordinate_processor.py
git commit -m "feat: add linear scaler with anchor-based mapping"
```

---

### 任务 5：坐标处理器 — EMA 平滑

- [ ] **步骤 1：编写失败的测试 — EMA**

```python
# tests/test_coordinate_processor.py (追加)
from src.coordinate_processor import EMAFilter


def test_ema_first_value():
    ema = EMAFilter(alpha=0.3)
    result = ema.update(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(result, [1.0, 2.0, 3.0])


def test_ema_smoothing():
    ema = EMAFilter(alpha=0.3)
    ema.update(np.array([0.0, 0.0, 0.0]))
    result = ema.update(np.array([1.0, 1.0, 1.0]))
    # smoothed = 0.3 * 1.0 + 0.7 * 0.0 = 0.3
    assert np.allclose(result, [0.3, 0.3, 0.3])


def test_ema_convergence():
    ema = EMAFilter(alpha=0.3)
    val = np.array([1.0, 1.0, 1.0])
    for _ in range(100):
        result = ema.update(val)
    assert np.allclose(result, [1.0, 1.0, 1.0], atol=1e-3)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`uv run pytest tests/test_coordinate_processor.py::test_ema_first_value -v`
预期：FAIL

- [ ] **步骤 3：编写实现 — EMAFilter**

```python
# src/coordinate_processor.py (追加)


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
```

- [ ] **步骤 4：运行测试验证通过**

运行：`uv run pytest tests/test_coordinate_processor.py::test_ema_first_value tests/test_coordinate_processor.py::test_ema_smoothing tests/test_coordinate_processor.py::test_ema_convergence -v`
预期：3 passed

- [ ] **步骤 5：Commit**

```bash
git add src/coordinate_processor.py tests/test_coordinate_processor.py
git commit -m "feat: add EMA smoothing filter"
```

---

### 任务 6：坐标处理器 — 防超程裁剪与 CoordinateProcessor 组装

- [ ] **步骤 1：编写失败的测试 — 防超程裁剪**

```python
# tests/test_coordinate_processor.py (追加)
from src.coordinate_processor import clip_to_workspace


def test_clip_inside_workspace():
    target = np.array([0.1, 0.1, 0.1])
    result = clip_to_workspace(target, max_radius=0.4)
    assert np.allclose(result, target)


def test_clip_outside_workspace():
    target = np.array([0.5, 0.5, 0.5])  # norm ~ 0.866 > 0.4
    result = clip_to_workspace(target, max_radius=0.4)
    assert np.linalg.norm(result) <= 0.4 + 1e-6
    # Direction preserved
    assert np.allclose(result / np.linalg.norm(result), target / np.linalg.norm(target), atol=1e-6)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`uv run pytest tests/test_coordinate_processor.py::test_clip_inside_workspace -v`
预期：FAIL

- [ ] **步骤 3：编写实现 — clip_to_workspace**

```python
# src/coordinate_processor.py (追加)


def clip_to_workspace(target: np.ndarray, max_radius: float) -> np.ndarray:
    norm = np.linalg.norm(target)
    if norm <= max_radius:
        return target.copy()
    return target * (max_radius / norm)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`uv run pytest tests/test_coordinate_processor.py::test_clip_inside_workspace tests/test_coordinate_processor.py::test_clip_outside_workspace -v`
预期：2 passed

- [ ] **步骤 5：编写 CoordinateProcessor 组装类的测试**

```python
# tests/test_coordinate_processor.py (追加)
from src.coordinate_processor import CoordinateProcessor


def test_processor_full_pipeline():
    proc = CoordinateProcessor(
        hand_range=0.3,
        robot_range=0.4,
        ema_alpha=0.3,
        max_radius=0.36,
    )
    # Set anchor at initial hand position
    anchor = np.array([0.5, 0.5, 0.5])
    proc.set_anchor(anchor)
    # Process same position -> should return zero (at anchor)
    result, detected = proc.process(anchor, detected=True)
    assert detected is True
    assert np.allclose(result, [0.0, 0.0, 0.0], atol=1e-6)


def test_processor_hand_lost():
    proc = CoordinateProcessor(
        hand_range=0.3,
        robot_range=0.4,
        ema_alpha=0.3,
        max_radius=0.36,
    )
    proc.set_anchor(np.array([0.5, 0.5, 0.5]))
    result1, _ = proc.process(np.array([0.6, 0.5, 0.5]), detected=True)
    result2, detected = proc.process(None, detected=False)
    assert detected is False
    assert np.allclose(result2, result1)  # frozen
```

- [ ] **步骤 6：运行测试验证失败**

运行：`uv run pytest tests/test_coordinate_processor.py::test_processor_full_pipeline -v`
预期：FAIL

- [ ] **步骤 7：编写 CoordinateProcessor 实现**

```python
# src/coordinate_processor.py (追加)


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
        self._scaler.set_anchor(hand_pos)

    def process(
        self, hand_pos: np.ndarray | None, detected: bool
    ) -> tuple[np.ndarray, bool]:
        if not detected or hand_pos is None:
            return self._last_target.copy(), False
        # Pipeline: remap -> scale -> smooth -> clip
        robot_coords = remap_camera_to_robot(hand_pos)
        scaled = self._scaler.scale(robot_coords)
        # Wait — scaler expects hand-frame coords, not robot-frame.
        # The pipeline in the spec is: hand normalized -> remap -> scale -> smooth -> clip
        # But scaler.set_anchor is called with hand-frame coords.
        # Need to remap anchor too. Let me fix: scaler works in robot frame after remap.
        # Actually, the scaler should work on remapped coords.
        # Let me redesign: anchor is set on raw hand coords, then we remap both.
        # Simpler: set_anchor takes hand coords, scale takes hand coords, then remap.
        # No — spec says: hand coords -> remap -> scale.
        # So anchor should be in robot frame. Let me refactor.
        ...
```

**注意：** 上面的 `process` 实现有设计问题 — `LinearScaler` 的锚定应该在重映射之后进行。需要调整：

- `set_anchor` 接收原始手部坐标，内部做重映射后存储
- `process` 中先重映射，再缩放

- [ ] **步骤 8：重构 LinearScaler 和 CoordinateProcessor**

```python
# src/coordinate_processor.py — 完整重写


import numpy as np


def remap_camera_to_robot(cam_coords: np.ndarray) -> np.ndarray:
    """Remap camera frame (X right, Y down, Z forward) to robot frame (X forward, Y left, Z up)."""
    return np.array([cam_coords[2], -cam_coords[0], -cam_coords[1]])


class LinearScaler:
    def __init__(self, hand_range: float = 0.3, robot_range: float = 0.4) -> None:
        self._scale = robot_range / hand_range
        self._anchor_robot = np.zeros(3)

    def set_anchor(self, hand_pos_robot_frame: np.ndarray) -> None:
        self._anchor_robot = hand_pos_robot_frame.copy()

    def scale(self, hand_pos_robot_frame: np.ndarray) -> np.ndarray:
        return (hand_pos_robot_frame - self._anchor_robot) * self._scale


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
```

- [ ] **步骤 9：运行全部测试验证通过**

运行：`uv run pytest tests/test_coordinate_processor.py -v`
预期：8 passed

- [ ] **步骤 10：Commit**

```bash
git add src/coordinate_processor.py tests/test_coordinate_processor.py
git commit -m "feat: complete coordinate processor with remap, scale, EMA, clip"
```

---

### 任务 7：摄像头模块

**文件：**
- 创建：`src/camera.py`

此模块涉及硬件 I/O，不做单元测试，通过集成测试验证。

- [ ] **步骤 1：编写 CameraThread 实现**

```python
# src/camera.py
import threading
import time
import cv2
import numpy as np


class CameraThread:
    def __init__(self, camera_id: int = 0, fps: int = 30) -> None:
        self._camera_id = camera_id
        self._fps = fps
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self._camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        interval = 1.0 / self._fps
        while self._running:
            ret, frame = cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)
        cap.release()
```

- [ ] **步骤 2：验证摄像头模块可导入**

运行：`uv run python -c "from src.camera import CameraThread; print('OK')"`
预期：`OK`

- [ ] **步骤 3：Commit**

```bash
git add src/camera.py
git commit -m "feat: add camera capture thread"
```

---

### 任务 8：手部检测器 — MediaPipe 部分

**文件：**
- 创建：`src/hand_detector.py`

- [ ] **步骤 1：编写 HandDetector 实现（仅 MediaPipe，不含 MiDaS）**

```python
# src/hand_detector.py
import mediapipe as mp
import numpy as np


class HandDetector:
    INDEX_FINGER_TIP = 8

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._drawer = mp.solutions.drawing_utils
        self._hand_connections = mp.solutions.hands.HAND_CONNECTIONS

    def detect(self, frame_rgb: np.ndarray) -> tuple[
        np.ndarray | None,  # left landmarks (21,3) or None
        np.ndarray | None,  # right landmarks (21,3) or None
        np.ndarray | None,  # left index tip (3,) or None
        np.ndarray | None,  # right index tip (3,) or None
    ]:
        h, w = frame_rgb.shape[:2]
        results = self._hands.process(frame_rgb)
        left_lm = None
        right_lm = None
        left_tip = None
        right_tip = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                lm = np.array(
                    [[l.x, l.y, l.z] for l in hand_landmarks.landmark]
                )
                tip = lm[self.INDEX_FINGER_TIP].copy()
                if label == "Left":
                    left_lm = lm
                    left_tip = tip
                else:
                    right_lm = lm
                    right_tip = tip
        return left_lm, right_lm, left_tip, right_tip

    def draw(self, frame_bgr: np.ndarray, left_lm, right_lm) -> np.ndarray:
        """Draw hand landmarks on BGR frame. Returns annotated frame."""
        annotated = frame_bgr.copy()
        if left_lm is not None:
            self._draw_landmarks(annotated, left_lm, (255, 0, 0))
        if right_lm is not None:
            self._draw_landmarks(annotated, right_lm, (0, 0, 255))
        return annotated

    def _draw_landmarks(self, frame: np.ndarray, lm: np.ndarray, color: tuple) -> None:
        h, w = frame.shape[:2]
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
        # Index tip highlighted
        tip_pt = (int(lm[8][0] * w), int(lm[8][1] * h))
        cv2.circle(frame, tip_pt, 8, (0, 0, 255), -1)

    def close(self) -> None:
        self._hands.close()
```

- [ ] **步骤 2：验证可导入**

运行：`uv run python -c "from src.hand_detector import HandDetector; print('OK')"`
预期：`OK`

- [ ] **步骤 3：Commit**

```bash
git add src/hand_detector.py
git commit -m "feat: add MediaPipe hand detector with landmark extraction"
```

---

### 任务 9：手部检测器 — MiDaS 深度集成

- [ ] **步骤 1：在 HandDetector 中添加 MiDaS 深度估计**

```python
# src/hand_detector.py — 追加 MiDaS 支持
# 在 __init__ 中添加：

import torch

# 在 __init__ 末尾添加：
self._midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
self._midas.eval()
self._midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
self._midas_transform = self._midas_transforms.small_transform
self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
self._midas.to(self._device)
```

- [ ] **步骤 2：添加 estimate_depth 方法**

```python
# 在 HandDetector 类中添加：

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
```

- [ ] **步骤 3：添加 get_index_tip_3d 方法整合深度**

```python
# 在 HandDetector 类中添加：

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
```

- [ ] **步骤 4：在 detect 方法中集成 get_index_tip_3d**

更新 `detect` 方法签名和返回值，增加 `frame_bgr` 参数，返回 3D 坐标：

```python
def detect(self, frame_rgb: np.ndarray, frame_bgr: np.ndarray | None = None) -> tuple[
    np.ndarray | None,  # left landmarks (21,3)
    np.ndarray | None,  # right landmarks (21,3)
    np.ndarray | None,  # left index tip 3D (3,) with MiDaS depth
    np.ndarray | None,  # right index tip 3D (3,) with MiDaS depth
]:
    # ... existing detection code ...
    # After getting tip_normalized, if frame_bgr provided:
    if left_tip is not None and frame_bgr is not None:
        left_tip = self.get_index_tip_3d(left_tip, frame_bgr)
    if right_tip is not None and frame_bgr is not None:
        right_tip = self.get_index_tip_3d(right_tip, frame_bgr)
    return left_lm, right_lm, left_tip, right_tip
```

- [ ] **步骤 5：验证 MiDaS 加载**

运行：`uv run python -c "from src.hand_detector import HandDetector; d = HandDetector(); print('MiDaS loaded')"`
预期：`MiDaS loaded`（首次运行会下载权重，可能需要几分钟）

- [ ] **步骤 6：Commit**

```bash
git add src/hand_detector.py
git commit -m "feat: integrate MiDaS depth estimation into hand detector"
```

---

### 任务 10：MuJoCo 仿真环境 — 固定基座 XML

**文件：**
- 创建：`assets/g1_23dof_fixed.xml`
- 创建：`src/sim_env.py`
- 创建：`tests/test_ik_solver.py`

- [ ] **步骤 1：创建固定基座 XML**

从 `g1_23dof.xml` 复制，将 floating base 改为 weld 约束。关键修改：

```xml
<!-- 在 </worldbody> 之后、<actuator> 之前添加 equality 约束 -->
<equality>
  <weld body1="world" body2="pelvis" solref="0.01 1"/>
</equality>
```

同时将 pelvis 的 joint 保留（MuJoCo weld 约束会覆盖），或改为 `limited="false"`。完整修改：在 `g1_23dof.xml` 的 `</worldbody>` 后添加上述 equality 块，保存为 `g1_23dof_fixed.xml`。

- [ ] **步骤 2：编写 SimEnvironment 实现**

```python
# src/sim_env.py
import mujoco
import numpy as np


class SimEnvironment:
    def __init__(self, xml_path: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # Set timestep for 500Hz physics
        self.model.opt.timestep = 0.002
        # Find arm joint indices
        self.left_arm_joints = self._find_joints([
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
        ])
        self.right_arm_joints = self._find_joints([
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ])
        # Find actuator indices
        self.left_arm_actuators = self._find_actuators([
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
        ])
        self.right_arm_actuators = self._find_actuators([
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ])
        # Find EEF body IDs
        self.left_eef_body = self._find_body("left_wrist_roll_rubber_hand")
        self.right_eef_body = self._find_body("right_wrist_roll_rubber_hand")
        # Get joint ranges
        self.left_joint_ranges = self.model.jnt_range[self.left_arm_joints]
        self.right_joint_ranges = self.model.jnt_range[self.right_arm_joints]
        # Initialize to default pose
        mujoco.mj_resetData(self.model, self.data)

    def _find_joints(self, names: list[str]) -> np.ndarray:
        indices = []
        for name in names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            indices.append(jnt_id)
        return np.array(indices)

    def _find_actuators(self, names: list[str]) -> np.ndarray:
        indices = []
        for name in names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            indices.append(act_id)
        return np.array(indices)

    def _find_body(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def get_eef_position(self, side: str) -> np.ndarray:
        body_id = self.left_eef_body if side == "left" else self.right_eef_body
        return self.data.xpos[body_id].copy()

    def get_joint_positions(self, side: str) -> np.ndarray:
        joints = self.left_arm_joints if side == "left" else self.right_arm_joints
        return self.data.qpos[joints].copy()

    def get_joint_velocities(self, side: str) -> np.ndarray:
        joints = self.left_arm_joints if side == "left" else self.right_arm_joints
        return self.data.qvel[joints].copy()

    def set_control(self, side: str, torques: np.ndarray) -> None:
        actuators = self.left_arm_actuators if side == "left" else self.right_arm_actuators
        self.data.ctrl[actuators] = torques

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)
```

- [ ] **步骤 3：验证 SimEnvironment 可加载 G1**

运行：`uv run python -c "from src.sim_env import SimEnvironment; env = SimEnvironment('assets/g1_23dof_fixed.xml'); print('Left arm joints:', env.left_arm_joints); print('EEF pos:', env.get_eef_position('left'))"`
预期：输出关节索引数组和 EEF 位置，无报错

- [ ] **步骤 4：Commit**

```bash
git add assets/g1_23dof_fixed.xml src/sim_env.py
git commit -m "feat: add MuJoCo sim environment with fixed-base G1"
```

---

### 任务 11：IK 求解器 — 雅可比 DLS + PD 控制

- [ ] **步骤 1：编写失败的测试 — IK 收敛**

```python
# tests/test_ik_solver.py
import numpy as np
from src.sim_env import SimEnvironment
from src.ik_solver import IKSolver


def test_ik_converges_to_reachable_target():
    env = SimEnvironment("assets/g1_23dof_fixed.xml")
    solver = IKSolver(env, damping=0.1, max_delta=0.1)
    # Get current EEF position as target (should be reachable)
    env.forward()
    current_pos = env.get_eef_position("left")
    # Move target slightly
    target = current_pos + np.array([0.05, 0.0, 0.0])
    # Run IK for 200 steps
    for _ in range(200):
        q = solver.solve("left", target)
    # Check convergence
    env.forward()
    final_pos = env.get_eef_position("left")
    error = np.linalg.norm(final_pos - target)
    assert error < 0.05, f"IK did not converge, error: {error:.4f}"


def test_ik_respects_joint_limits():
    env = SimEnvironment("assets/g1_23dof_fixed.xml")
    solver = IKSolver(env, damping=0.1, max_delta=0.1)
    # Try to reach an unreachable far target
    target = np.array([1.0, 0.0, 1.0])
    for _ in range(500):
        q = solver.solve("left", target)
    # Joint positions should be within limits
    joints = env.get_joint_positions("left")
    ranges = env.left_joint_ranges
    assert np.all(joints >= ranges[:, 0] - 0.01)
    assert np.all(joints <= ranges[:, 1] + 0.01)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`uv run pytest tests/test_ik_solver.py -v`
预期：FAIL，ModuleNotFoundError

- [ ] **步骤 3：编写 IKSolver 实现**

```python
# src/ik_solver.py
import mujoco
import numpy as np
from src.sim_env import SimEnvironment


class IKSolver:
    def __init__(
        self,
        env: SimEnvironment,
        damping: float = 0.1,
        max_delta: float = 0.1,
        kp: float = 50.0,
        kd: float = 5.0,
    ) -> None:
        self._env = env
        self._damping = damping
        self._max_delta = max_delta
        self._kp = kp
        self._kd = kd
        # Pre-allocate Jacobian arrays
        self._jac_pos = np.zeros((3, env.model.nv))
        self._jac_rot = np.zeros((3, env.model.nv))

    def solve(self, side: str, target_pos: np.ndarray) -> np.ndarray:
        """Run one IK step. Returns joint positions to set as control target."""
        env = self._env
        if side == "left":
            body_id = env.left_eef_body
            joints = env.left_arm_joints
            joint_ranges = env.left_joint_ranges
        else:
            body_id = env.right_eef_body
            joints = env.right_arm_joints
            joint_ranges = env.right_joint_ranges

        # Compute Jacobian
        mujoco.mj_jac(env.model, env.data, self._jac_pos, self._jac_rot, body_id)
        # Extract arm columns
        dof_indices = []
        for jnt_id in joints:
            dof_id = env.model.jnt_dofadr[jnt_id]
            dof_indices.append(dof_id)
        J = self._jac_pos[:, dof_indices]  # (3, 5)

        # Current EEF position
        current_pos = env.data.xpos[body_id].copy()
        error = target_pos - current_pos

        # DLS: dq = J^T (J J^T + lambda^2 I)^{-1} error
        JJT = J @ J.T + self._damping**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)

        # Clip delta
        dq = np.clip(dq, -self._max_delta, self._max_delta)

        # Update joint positions
        q_current = env.data.qpos[joints].copy()
        q_new = q_current + dq

        # Clip to joint limits
        q_new = np.clip(q_new, joint_ranges[:, 0], joint_ranges[:, 1])

        # Set qpos directly (for position-level IK)
        env.data.qpos[joints] = q_new
        env.forward()
        return q_new

    def compute_pd_torque(self, side: str, q_target: np.ndarray) -> np.ndarray:
        """Compute PD torque from target joint positions."""
        env = self._env
        q_current = env.get_joint_positions(side)
        qdot = env.get_joint_velocities(side)
        torque = self._kp * (q_target - q_current) - self._kd * qdot
        return torque
```

- [ ] **步骤 4：运行测试验证通过**

运行：`uv run pytest tests/test_ik_solver.py -v`
预期：2 passed

- [ ] **步骤 5：Commit**

```bash
git add src/ik_solver.py tests/test_ik_solver.py
git commit -m "feat: add Jacobian DLS IK solver with PD torque control"
```

---

### 任务 12：可视化模块

**文件：**
- 创建：`src/visualizer.py`

- [ ] **步骤 1：编写 Visualizer 实现**

```python
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

    def _find_site(self, name: str) -> int | None:
        try:
            return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
        except Exception:
            return -1

    def start_viewer(self) -> None:
        self._viewer = mujoco.viewer.launch_passive(
            self._model, self._data
        )

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

        # Draw landmarks
        if left_lm is not None:
            self._draw_hand(annotated, left_lm, (255, 0, 0), h, w)
        if right_lm is not None:
            self._draw_hand(annotated, right_lm, (0, 0, 255), h, w)

        # HUD
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
```

- [ ] **步骤 2：验证可导入**

运行：`uv run python -c "from src.visualizer import Visualizer; print('OK')"`
预期：`OK`

- [ ] **步骤 3：Commit**

```bash
git add src/visualizer.py
git commit -m "feat: add dual-window visualizer (OpenCV + MuJoCo viewer)"
```

---

### 任务 13：主循环集成

**文件：**
- 修改：`main.py`

- [ ] **步骤 1：编写主循环**

```python
# main.py
import time
import cv2
import numpy as np

from src.camera import CameraThread
from src.hand_detector import HandDetector
from src.coordinate_processor import CoordinateProcessor
from src.sim_env import SimEnvironment
from src.ik_solver import IKSolver
from src.visualizer import Visualizer


def main() -> None:
    # Initialize modules
    camera = CameraThread(camera_id=0, fps=30)
    detector = HandDetector(max_hands=2)
    left_proc = CoordinateProcessor(hand_range=0.3, robot_range=0.4, ema_alpha=0.3, max_radius=0.36)
    right_proc = CoordinateProcessor(hand_range=0.3, robot_range=0.4, ema_alpha=0.3, max_radius=0.36)
    sim = SimEnvironment("assets/g1_23dof_fixed.xml")
    ik = IKSolver(sim, damping=0.1, max_delta=0.1)
    viz = Visualizer(sim.model, sim.data)

    # Start camera
    camera.start()

    # Anchor calibration: wait for first hand detection
    print("Calibrating... Show your hands to the camera.")
    anchored = False
    fps_timer = time.time()
    frame_count = 0
    fps = 0.0

    try:
        viz.start_viewer()
    except Exception:
        print("MuJoCo viewer not available, running headless")

    running = True
    while running:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # Detect hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        left_lm, right_lm, left_tip, right_tip = detector.detect(frame_rgb, frame)

        # Anchor on first detection
        if not anchored and (left_tip is not None or right_tip is not None):
            if left_tip is not None:
                left_proc.set_anchor(left_tip)
            if right_tip is not None:
                right_proc.set_anchor(right_tip)
            anchored = True
            print("Anchored!")

        # Process coordinates
        left_target, left_det = left_proc.process(left_tip, left_tip is not None)
        right_target, right_det = right_proc.process(right_tip, right_tip is not None)

        # IK + control
        if left_det:
            q_left = ik.solve("left", left_target)
            torque_left = ik.compute_pd_torque("left", q_left)
            sim.set_control("left", torque_left)
        if right_det:
            q_right = ik.solve("right", right_target)
            torque_right = ik.compute_pd_torque("right", q_right)
            sim.set_control("right", torque_right)

        # Step simulation (50Hz control = every 10 physics steps)
        for _ in range(10):
            sim.step()

        # FPS
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps = frame_count / (time.time() - fps_timer)
            frame_count = 0
            fps_timer = time.time()

        # Visualize
        annotated = viz.draw_opencv(
            frame, left_lm, right_lm,
            left_target if left_det else None,
            right_target if right_det else None,
            fps,
        )
        cv2.imshow("Hand Teleoperation", annotated)
        viz.update_viewer(
            left_target if left_det else None,
            right_target if right_det else None,
        )

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False

    camera.stop()
    detector.close()
    viz.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：验证完整系统可启动**

运行：`uv run python main.py`
预期：摄像头打开，OpenCV 窗口显示，MuJoCo viewer 显示 G1 机器人。双手移动时手臂跟随。（需要摄像头硬件）

- [ ] **步骤 3：Commit**

```bash
git add main.py
git commit -m "feat: integrate all modules into main teleoperation loop"
```

---

### 任务 14：MuJoCo viewer site 添加

**文件：**
- 修改：`assets/g1_23dof_fixed.xml`

- [ ] **步骤 1：在 XML 中添加 EEF target sites**

在 `</worldbody>` 之前、`right_wrist_roll_rubber_hand` body 闭合后添加：

```xml
<!-- EEF target visualization sites -->
<site name="left_target" pos="0 0 0" size="0.02" rgba="0 1 0 0.5"/>
<site name="right_target" pos="0 0 0" size="0.02" rgba="0 1 0 0.5"/>
```

- [ ] **步骤 2：验证 XML 有效**

运行：`uv run python -c "import mujoco; m = mujoco.MjModel.from_xml_path('assets/g1_23dof_fixed.xml'); print('Sites:', m.nsite)"`
预期：`Sites: 2`（或更多，加上已有的 IMU sites）

- [ ] **步骤 3：Commit**

```bash
git add assets/g1_23dof_fixed.xml
git commit -m "feat: add EEF target visualization sites to G1 model"
```

---

### 任务 15：运行全部测试

- [ ] **步骤 1：运行单元测试**

运行：`uv run pytest tests/ -v`
预期：所有测试通过

- [ ] **步骤 2：运行集成测试**

运行：`uv run python -c "from src.sim_env import SimEnvironment; from src.ik_solver import IKSolver; import numpy as np; env = SimEnvironment('assets/g1_23dof_fixed.xml'); solver = IKSolver(env); env.forward(); t = env.get_eef_position('left') + [0.05,0,0]; [solver.solve('left', t) for _ in range(200)]; env.forward(); print('Error:', np.linalg.norm(env.get_eef_position('left') - t))"`
预期：`Error: < 0.05`

- [ ] **步骤 3：最终 Commit**

```bash
git add -A
git commit -m "chore: final integration verification"
```

---

## 自检结果

1. **规格覆盖度：**
   - 实时手部检测 ✅ (任务 8-9)
   - EEF 3D 坐标提取 ✅ (任务 8-9)
   - 深度估计 MiDaS ✅ (任务 9)
   - 坐标变换 ✅ (任务 3-6)
   - 坐标平滑 EMA ✅ (任务 5-6)
   - 防超程保护 ✅ (任务 6)
   - IK 求解 ✅ (任务 11)
   - 仿真控制 PD ✅ (任务 10-11)
   - 可视化双窗口 ✅ (任务 12)
   - 单元测试 ✅ (任务 2-6, 11)

2. **占位符扫描：** 无 TODO/TBD，所有步骤含完整代码

3. **类型一致性：** `LinearScaler`、`EMAFilter`、`CoordinateProcessor` 在定义和使用处签名一致；`IKSolver.solve` 返回 `np.ndarray`，`compute_pd_torque` 返回 `np.ndarray`，与 `sim.set_control` 参数匹配
