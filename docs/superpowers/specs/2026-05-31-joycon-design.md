# Joycon 控制类设计规格

**日期**: 2026-05-31
**目标**: 实现 Nintendo Switch Joycon 游戏手柄的通用控制类，支持左右双柄蓝牙连接与实时事件读取

---

## 1. 通信机制

- **蓝牙 HID** 通过 `pyjoycon` 库连接，无需额外硬件
- 轮询间隔: 10ms (可配置)

---

## 2. 架构

```
JoyconManager (主入口)
    ├── _devices: dict[mac, JoyconDevice]
    ├── _running: bool
    ├── _thread: Thread
    └── 回调接口
        ├── on_button(side, key, pressed)
        ├── on_stick(side, x, y)
        └── on_imu(side, acc, gyro)

JoyconDevice (单个手柄封装)
    ├── mac: str
    ├── side: Side.LEFT | Side.RIGHT | None
    ├── _pyjoycon_instance
    ├── _last_state
    └── _poll() → 触发回调
```

---

## 3. 数据结构

```python
class Side(Enum):
    LEFT = "left"
    RIGHT = "right"

class JoyconState:
    buttons: dict[str, bool]                          # "a", "b", "x", "y", ...
    sticks: dict[str, tuple[float, float]]           # "left": (x, y), "right": (x, y)
    accel: tuple[float, float, float]                # (x, y, z) m/s²
    gyro: tuple[float, float, float]                 # (x, y, z) rad/s
```

---

## 4. 公开 API

```python
class JoyconManager:
    def __init__(self, poll_interval: float = 0.01): ...

    def scan(self, timeout: float = 5.0) -> int: ...
    on_button: Callable[[Side, str, bool], None]
    on_stick: Callable[[Side, float, float], None]
    on_imu: Callable[[Side, tuple, tuple], None]
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_state(self, mac: str) -> JoyconState | None: ...
    def get_left_mac(self) -> str | None: ...
    def get_right_mac(self) -> str | None: ...
```

---

## 5. 错误处理

- `scan()` 无设备不抛异常，返回 0
- `start()`/`stop()` 幂等调用
- 蓝牙断开时最多重连 3 次
- 后台线程 join 超时 2s

---

## 6. 数据读取内容

完整读取：
- 数字按钮: A/B/X/Y, L/R/ZL/ZR, +/−/Home/Capture
- 左/右摇杆: X/Y 值 + 按下状态
- 六轴惯性传感器: 加速度计 + 陀螺仪

---

## 7. 实现策略

- 方案 1: 单一 JoyconManager 类管理所有设备
- 回调模式驱动事件
- 后台线程轮询 + 状态变化检测触发回调