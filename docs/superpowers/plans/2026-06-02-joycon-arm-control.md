# JoyCon 机械臂控制实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 使用 JoyCon 右摇杆控制 SO101 机械臂末端位置，肩键控制 Z 轴，A 键切换夹爪

**架构：** JoyCon R 摇杆输入 → 位置积分 → IK 求解 → 机械臂控制。复用现有 SimEnvironment、IKSolver、Visualizer 组件。

**技术栈：** pyjoycon, mujoco, pinocchio, numpy

---

## 文件结构

- **修改：** `src/joycon_utils.py` — 添加 R 摇杆/肩键/A 键辅助方法
- **创建：** `main_joycon.py` — JoyCon 控制入口（复用 sim_env、ik_solver、visualizer）

---

## 任务 1：扩展 JoyCon 工具类

**文件：** `src/joycon_utils.py`

- [ ] **步骤 1：添加辅助方法**

在 `JoyCon` 类中添加以下方法：

```python
# 右摇杆归一化值（死区过滤后）
# 返回 (x, y)，范围约 -1 到 1
def get_R_analog(self) -> tuple[float, float]:
    status = self.get_status()
    R = status["R"]["analog-sticks"]["right"]
    h = R["horizontal"]
    v = R["vertical"]
    # 中性值约 2048，死区 ±100
    DEADZONE = 100
    if abs(h - 2048) < DEADZONE:
        h = 2048
    if abs(v - 2048) < DEADZONE:
        v = 2048
    x = (h - 2048) / 2048.0  # -1 到 1
    y = (v - 2048) / 2048.0  # -1 到 1（Y 轴反向）
    return x, y

# R/ZR 肩键状态：ZR=1, R=1 都按=0, 否则=0
# 返回 1=上升, -1=下降, 0=静止
def get_R_shoulder(self) -> int:
    status = self.get_status()
    R = status["R"]["buttons"]["right"]
    zr = R["zr"]
    r = R["r"]
    if zr and not r:
        return 1   # ZR pressed = 上升
    if r and not zr:
        return -1  # R pressed = 下降
    return 0

# R A 键单次按下检测（边缘触发）
# 首次调用返回 False，之后按一次返回 True，然后等松开再按才返回 True
def get_A_pressed(self) -> bool:
    status = self.get_status()
    current = status["R"]["buttons"]["right"]["a"]
    pressed = current and not getattr(self, "_A_last", 0)
    self._A_last = current
    return pressed

# 获取夹爪命令（A 键切换）
def get_gripper_toggle(self) -> bool:
    return self.get_A_pressed()
```

- [ ] **步骤 2：运行测试验证**

创建 `tests/test_joycon_utils.py`：
```python
import sys
sys.path.insert(0, "src")
from joycon_utils import JoyCon

def test_joycon_init():
    # 需要实际 JoyCon 设备，skip 如果不可用
    try:
        joycon = JoyCon(calibration_seconds=0.1)
    except RuntimeError:
        print("JoyCon not found, skipping")
        return
    # 测试辅助方法存在
    assert hasattr(joycon, "get_R_analog")
    assert hasattr(joycon, "get_R_shoulder")
    assert hasattr(joycon, "get_A_pressed")
    print("JoyCon helper methods OK")
```

运行：`python -c "from src.joycon_utils import JoyCon; print('Import OK')"`

- [ ] **步骤 3：Commit**

```bash
git add src/joycon_utils.py
git commit -m "feat: add JoyCon helper methods for arm control"
```

---

## 任务 2：创建 JoyCon 主控制程序

**文件：** `main_joycon.py`

- [ ] **步骤 1：编写程序框架**

```python
"""JoyCon teleoperation for SO101 robot arm.

Right stick X/Y -> arm X/Y position
R/ZR shoulders -> Z axis up/down
A button -> gripper toggle
"""

import time
import numpy as np

from src.joycon_utils import JoyCon
from src.sim_env import SimEnvironment
from src.ik_solver import IKSolver
from src.visualizer import Visualizer

URDF = "assets/SO101/so101_new_calib.urdf"
XML = "assets/SO101/scene.xml"

GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 1.2

# 控制参数
STICK_SENSITIVITY = 0.002  # 每帧位置增量
Z_SENSITIVITY = 0.003      # Z 轴每帧增量
MAX_POSITION_DELTA = 0.05   # 每步最大位置变化


def main() -> None:
    # 初始化组件
    joycon = JoyCon(calibration_seconds=2.0)
    print("JoyCon initialized and calibrated.")

    sim = SimEnvironment(XML)
    ik = IKSolver(URDF, damping=0.1, max_delta=0.1)
    viz = Visualizer(sim.model, sim.data)

    # 初始化夹爪状态
    gripper_pos = GRIPPER_OPEN
    gripper_toggle_pending = False

    # 获取初始末端位置作为控制起点
    initial_pos = sim.get_eef_position()
    current_target = initial_pos.copy()
    print(f"Initial EEF position: {initial_pos}")

    try:
        viz.start_viewer()
    except Exception:
        print("MuJoCo viewer not available, running headless")

    try:
        joycon.get_R_analog()  # 初始化 A_pressed 状态
        while True:
            # 读取 JoyCon 输入
            dx, dy = joycon.get_R_analog()
            z_dir = joycon.get_R_shoulder()

            # 计算位置增量
            delta = np.array([dx * STICK_SENSITIVITY, -dy * STICK_SENSITIVITY, z_dir * Z_SENSITIVITY])
            delta = np.clip(delta, -MAX_POSITION_DELTA, MAX_POSITION_DELTA)

            # 更新目标位置
            current_target += delta

            # 限幅（球形工作空间）
            max_radius = 0.36
            norm = np.linalg.norm(current_target)
            if norm > max_radius:
                current_target *= max_radius / norm

            # IK 求解
            q_current = sim.get_joint_positions()
            q_new = ik.solve(q_current, current_target)
            sim.set_control(q_new)

            # 夹爪切换检测
            if joycon.get_gripper_toggle():
                gripper_pos = GRIPPER_CLOSED if gripper_pos > (GRIPPER_CLOSED + GRIPPER_OPEN) / 2 else GRIPPER_OPEN
                print(f"Gripper: {'open' if gripper_pos > 0.6 else 'closed'}")

            sim.set_gripper(gripper_pos)

            # 物理步骤
            for _ in range(10):
                sim.step()

            # 可视化更新
            viz.update_viewer(current_target, None)

            # 按键退出
            # （JoyCon 无键盘输入，通过 MuJoCo viewer 窗口关闭或手动 Ctrl+C）

    except KeyboardInterrupt:
        print("\nExiting...")

    viz.close()
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：验证导入**

运行：`python -c "from main_joycon import main; print('Import OK')"`

预期：无输出或 Import OK

- [ ] **步骤 3：Commit**

```bash
git add main_joycon.py src/joycon_utils.py
git commit -m "feat: add JoyCon-based arm teleoperation control"
```

---

## 自检

- [ ] 规格覆盖度：JoyCon R 摇杆 → X/Y，肩键 → Z，A → 夹爪切换 ✓
- [ ] 占位符扫描：无 TODO/待定 ✓
- [ ] 类型一致性：所有方法在任务 1 中定义，任务 2 中使用相同签名 ✓
- [ ] 限幅：`clip_to_workspace` 逻辑在任务 2 中实现 ✓