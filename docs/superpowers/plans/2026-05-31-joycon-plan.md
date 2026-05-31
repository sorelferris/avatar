# Joycon 控制类实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现 Nintendo Switch Joycon 游戏手柄的通用控制类，支持左右双柄蓝牙连接与实时事件读取

**架构：** 单一 JoyconManager 类管理所有设备，后台线程轮询状态变化并通过回调分发事件

**技术栈：** Python, pyjoycon, threading

---

## 文件结构

- **创建**: `src/joycon.py` - JoyconManager 和 JoyconDevice 类，以及 Side 枚举和 JoyconState 数据类

---

## 任务 1：基础框架与枚举定义

**文件：**
- 创建：`src/joycon.py`

- [ ] **步骤 1：创建 joycon.py 基础结构**

```python
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any
import threading
import logging

logger = logging.getLogger(__name__)


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class JoyconState:
    """Current state of a Joycon controller."""
    buttons: dict[str, bool]
    sticks: dict[str, tuple[float, float]]
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]


class JoyconDevice:
    """Wrapper for a single Joycon device."""

    def __init__(self, mac: str, pyjoycon_instance: Any):
        self.mac = mac
        self._instance = pyjoycon_instance
        self.side: Side | None = None
        self._last_state: JoyconState | None = None


class JoyconManager:
    """Manages all Joycon devices and dispatches events via callbacks."""

    def __init__(self, poll_interval: float = 0.01):
        self._poll_interval = poll_interval
        self._devices: dict[str, JoyconDevice] = {}
        self._running = False
        self._thread: threading.Thread | None = None

        # Callbacks
        self.on_button: Callable[[Side, str, bool], None] | None = None
        self.on_stick: Callable[[Side, float, float], None] | None = None
        self.on_imu: Callable[[Side, tuple, tuple], None] | None = None

    def scan(self, timeout: float = 5.0) -> int:
        """Scan for Joycon devices. Returns number of devices found."""
        ...

    def start(self) -> None:
        """Start the event polling thread."""
        ...

    def stop(self) -> None:
        """Stop the event polling thread."""
        ...

    def get_state(self, mac: str) -> JoyconState | None:
        """Get current state of a device by MAC address."""
        ...

    def get_left_mac(self) -> str | None:
        """Get MAC address of left Joycon."""
        ...

    def get_right_mac(self) -> str | None:
        """Get MAC address of right Joycon."""
        ...
```

- [ ] **步骤 2：Commit**

```bash
git add src/joycon.py
git commit -m "feat: add JoyconManager skeleton with Side/JoyconState types

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## 任务 2：实现 scan() 设备发现

**文件：**
- 修改：`src/joycon.py`

- [ ] **步骤 1：实现 scan 方法**

在 `JoyconManager` 类中实现：

```python
def scan(self, timeout: float = 5.0) -> int:
    """Scan for Joycon devices. Returns number of devices found."""
    try:
        from pyjoycon import JoyConManager
    except ImportError:
        logger.warning("pyjoycon not installed, cannot scan")
        return 0

    manager = JoyConManager()
    joycons = manager.get_joycons()
    count = 0

    for mac, instance in joycons.items():
        if mac not in self._devices:
            device = JoyconDevice(mac, instance)
            self._devices[mac] = device
            count += 1
            logger.info(f"Found Joycon: {mac}")

    return count
```

- [ ] **步骤 2：Commit**

```bash
git add src/joycon.py
git commit -m "feat: implement JoyconManager.scan() device discovery

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## 任务 3：实现 start/stop 生命周期

**文件：**
- 修改：`src/joycon.py`

- [ ] **步骤 1：实现 start/stop 方法**

```python
def start(self) -> None:
    """Start the event polling thread."""
    if self._running:
        return
    self._running = True
    self._thread = threading.Thread(target=self._poll_loop, daemon=True)
    self._thread.start()
    logger.info("JoyconManager started")

def stop(self) -> None:
    """Stop the event polling thread."""
    if not self._running:
        return
    self._running = False
    if self._thread is not None:
        self._thread.join(timeout=2.0)
        self._thread = None
    logger.info("JoyconManager stopped")

def _poll_loop(self) -> None:
    """Background polling loop."""
    while self._running:
        for device in list(self._devices.values()):
            self._poll_device(device)
        import time
        time.sleep(self._poll_interval)
```

- [ ] **步骤 2：Commit**

```bash
git add src/joycon.py
git commit -m "feat: implement JoyconManager start/stop lifecycle

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## 任务 4：实现设备轮询与事件分发

**文件：**
- 修改：`src/joycon.py`

- [ ] **步骤 1：实现 _poll_device 和状态比较逻辑**

```python
def _poll_device(self, device: JoyconDevice) -> None:
    """Poll a single device and dispatch events on state change."""
    try:
        status = device._instance.get_status()
    except Exception as e:
        logger.warning(f"Failed to poll {device.mac}: {e}")
        return

    current_state = self._parse_status(status, device.side)

    if device._last_state is not None:
        self._dispatch_events(device, device._last_state, current_state)

    device._last_state = current_state


def _parse_status(self, status: dict, side: Side | None) -> JoyconState:
    """Parse pyjoycon status dict into JoyconState."""
    buttons = status.get("buttons", {})
    sticks = status.get("analyst", {})  # Note: pyjoycon may differ
    accel = status.get("accel", (0, 0, 0))
    gyro = status.get("gyro", (0, 0, 0))
    return JoyconState(
        buttons=buttons,
        sticks=sticks,
        accel=tuple(accel),
        gyro=tuple(gyro),
    )


def _dispatch_events(self, device: JoyconDevice, old: JoyconState, new: JoyconState) -> None:
    """Compare states and dispatch callback events."""
    side = device.side or Side.LEFT

    # Button events
    for key, pressed in new.buttons.items():
        was_pressed = old.buttons.get(key, False)
        if pressed != was_pressed and self.on_button:
            self.on_button(side, key, pressed)

    # Stick events (dispatch when value changes significantly)
    for stick_name, (x, y) in new.sticks.items():
        old_x, old_y = old.sticks.get(stick_name, (0, 0))
        if abs(x - old_x) > 0.01 or abs(y - old_y) > 0.01:
            if self.on_stick:
                self.on_stick(side, x, y)

    # IMU events
    if self.on_imu:
        self.on_imu(side, new.accel, new.gyro)
```

- [ ] **步骤 2：Commit**

```bash
git add src/joycon.py
git commit -m "feat: implement device polling and event dispatch

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## 任务 5：实现状态查询方法

**文件：**
- 修改：`src/joycon.py`

- [ ] **步骤 1：实现 get_state, get_left_mac, get_right_mac**

```python
def get_state(self, mac: str) -> JoyconState | None:
    """Get current state of a device by MAC address."""
    device = self._devices.get(mac)
    if device is None:
        return None
    return device._last_state


def get_left_mac(self) -> str | None:
    """Get MAC address of left Joycon."""
    for mac, device in self._devices.items():
        if device.side == Side.LEFT:
            return mac
    return None


def get_right_mac(self) -> str | None:
    """Get MAC address of right Joycon."""
    for mac, device in self._devices.items():
        if device.side == Side.RIGHT:
            return mac
    return None
```

- [ ] **步骤 2：Commit**

```bash
git add src/joycon.py
git commit -m "feat: implement state query methods

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## 任务 6：实现自动手柄类型检测

**文件：**
- 修改：`src/joycon.py`

- [ ] **步骤 1：在 scan 中检测左右 Joycon**

pyjoycon 的 JoyConManager 可以通过设备类型判断左右手柄。修改 scan 方法：

```python
def scan(self, timeout: float = 5.0) -> int:
    """Scan for Joycon devices. Returns number of devices found."""
    try:
        from pyjoycon import JoyConManager
    except ImportError:
        logger.warning("pyjoycon not installed, cannot scan")
        return 0

    manager = JoyConManager()
    joycons = manager.get_joycons()
    count = 0

    for mac, instance in joycons.items():
        if mac not in self._devices:
            device = JoyconDevice(mac, instance)
            # Detect side from device type
            device.side = self._detect_side(instance)
            self._devices[mac] = device
            count += 1
            logger.info(f"Found Joycon: {mac} ({device.side.value})")

    return count


def _detect_side(self, instance: Any) -> Side:
    """Detect if this is a left or right Joycon."""
    try:
        device_type = instance.get_device_type()
        if "left" in str(device_type).lower():
            return Side.LEFT
        elif "right" in str(device_type).lower():
            return Side.RIGHT
    except Exception:
        pass
    return Side.LEFT  # default
```

- [ ] **步骤 2：Commit**

```bash
git add src/joycon.py
git commit -m "feat: auto-detect left/right Joycon type

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## 计划已完成并保存到 `docs/superpowers/plans/2026-05-31-joycon-plan.md`

两种执行方式：

**1. 子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代

**2. 内联执行** - 在当前会话中使用 executing-plans 执行任务，批量执行并设有检查点

选哪种方式？