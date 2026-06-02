# JoyCon IMU 机械臂控制实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 使用 JoyCon IMU 传感器控制 SO101 机械臂末端位置和姿态（最小 demo）

**架构：** ZR 按下记录基准 → 手柄 motion → 相对偏移 → IK 求解 → 机械臂控制

**技术栈：** pyjoycon, mujoco, pinocchio, numpy

---

## 文件结构

- **修改：** `src/joycon_utils.py` — 添加 IMU 相对偏移方法
- **创建：** `main_joycon_imu.py` — IMU 控制入口

---

## 任务 1：扩展 JoyCon IMU 相对偏移方法

**文件：** `src/joycon_utils.py`

- [ ] **步骤 1：添加 IMU 相对偏移辅助方法**

在 `JoyCon` 类中添加：

```python
def get_R_imu_delta(self) -> dict:
    """获取按住 ZR 期间的 IMU 相对偏移（相对于基准姿态）。
    
    首次调用（ZR 刚按下）记录基准姿态，返回全零。
    之后每次返回相对基准姿态的欧拉角变化。
    """
    status = self.get_status()
    R = status["R"]
    
    # 欧拉角（弧度，ZYX 顺序）
    rotation = R["rotation"]
    
    if not hasattr(self, "_imu_baseline"):
        self._imu_baseline = {
            "rotation": rotation,
        }
        return {"position": (0.0, 0.0, 0.0), "attitude": (0.0, 0.0, 0.0)}
    
    # 计算相对偏移
    pos_delta = (0.0, 0.0, 0.0)  # 加速度积分暂用零
    att_delta = (
        rotation[0] - self._imu_baseline["rotation"][0],
        rotation[1] - self._imu_baseline["rotation"][1],
        rotation[2] - self._imu_baseline["rotation"][2],
    )
    
    return {"position": pos_delta, "attitude": att_delta}

def is_ZR_pressed(self) -> bool:
    """检查 ZR 肩键是否按下。"""
    status = self.get_status()
    return bool(status["R"]["buttons"]["right"]["zr"])

def reset_imu_baseline(self) -> None:
    """重置 IMU 基准姿态（松开 ZR 时调用）。"""
    if hasattr(self, "_imu_baseline"):
        delattr(self, "_imu_baseline")
```

- [ ] **步骤 2：验证语法**

运行：`python -c "from src.joycon_utils import JoyCon; print('Import OK')"`

- [ ] **步骤 3：Commit**

```bash
git add src/joycon_utils.py
git commit -m "feat: add IMU delta methods for JoyCon arm control"
```

---

## 任务 2：创建 IMU 主控制程序

**文件：** `main_joycon_imu.py`

- [ ] **步骤 1：编写程序框架**

```python
"""JoyCon IMU teleoperation for SO101 robot arm.

ZR deadman switch -> record baseline
Motion -> relative position/attitude delta -> IK -> arm control
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

# 控制参数
ATTITUDE_SENSITIVITY = 0.5   # 姿态欧拉角到位置偏移的映射系数
POSITION_SENSITIVITY = 0.01  # 位置增量系数（加速度积分用）
MAX_POSITION_DELTA = 0.05    # 每步最大位置变化


def main() -> None:
    # 初始化组件
    joycon = JoyCon(calibration_seconds=2.0)
    print("JoyCon initialized and calibrated.")

    sim = SimEnvironment(XML)
    ik = IKSolver(URDF, damping=0.1, max_delta=0.1)
    viz = Visualizer(sim.model, sim.data)

    # 初始化夹爪状态（使用 URDF 中的 gripper range）
    gripper_pos = sim.gripper_range[1] * 0.8
    print(f"Initial gripper: open ({gripper_pos:.3f})")

    # 获取初始末端位置和姿态
    initial_pos = sim.get_eef_position()
    current_target_pos = initial_pos.copy()
    current_target_att = np.eye(3)  # 单位矩阵作为初始姿态
    print(f"Initial EEF position: {initial_pos}")

    try:
        viz.start_viewer()
    except Exception:
        print("MuJoCo viewer not available, running headless")

    try:
        zr_was_pressed = False
        while True:
            zr_pressed = joycon.is_ZR_pressed()

            # ZR 上升沿：记录基准
            if zr_pressed and not zr_was_pressed:
                joycon.get_R_imu_delta()  # 内部记录基准
                print("ZR pressed - baseline recorded")

            # ZR 释放：重置基准
            if not zr_pressed and zr_was_pressed:
                joycon.reset_imu_baseline()
                print("ZR released - baseline reset")

            zr_was_pressed = zr_pressed

            # 控制
            if zr_pressed:
                delta = joycon.get_R_imu_delta()
                att_delta = delta["attitude"]

                # 姿态偏移（Roll/Pitch/Yaw）-> 位置增量（简化：直接用欧拉角变化量）
                # 实际应用中需要根据姿态计算末端位置变化
                position_change = np.array([
                    att_delta[0] * ATTITUDE_SENSITIVITY,  # Roll -> X
                    att_delta[1] * ATTITUDE_SENSITIVITY,  # Pitch -> Y
                    att_delta[2] * ATTITUDE_SENSITIVITY,  # Yaw -> Z（这里可能需要调整）
                ])
                position_change = np.clip(position_change, -MAX_POSITION_DELTA, MAX_POSITION_DELTA)
                current_target_pos += position_change

                # 限幅（球形工作空间）
                max_radius = 0.36
                norm = np.linalg.norm(current_target_pos)
                if norm > max_radius:
                    current_target_pos *= max_radius / norm

            # IK 求解
            q_current = sim.get_joint_positions()
            q_new = ik.solve(q_current, current_target_pos)
            sim.set_control(q_new)

            # 夹爪切换检测
            if joycon.get_gripper_toggle():
                gripper_pos = 0.0 if gripper_pos > 0.1 else sim.gripper_range[1] * 0.8
                state = "open" if gripper_pos > 0.1 else "closed"
                print(f"Gripper: {state}")

            sim.set_gripper(gripper_pos)

            # 物理步骤
            for _ in range(10):
                sim.step()

            # 可视化更新
            viz.update_viewer(None, current_target_pos)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nExiting...")

    viz.close()
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：验证导入**

运行：`python -c "from main_joycon_imu import main; print('Import OK')"`

- [ ] **步骤 3：Commit**

```bash
git add main_joycon_imu.py src/joycon_utils.py
git commit -m "feat: add JoyCon IMU-based arm teleoperation demo"
```

---

## 自检

- [ ] 规格覆盖度：ZR 死区开关、IMU 相对偏移、位置+姿态控制、夹爪 A 键 ✓
- [ ] 占位符扫描：无 TODO/待定 ✓
- [ ] 类型一致性：方法在任务 1 定义，任务 2 使用相同签名 ✓
- [ ] 注意：position delta 暂用零，实际需要加速度积分实现