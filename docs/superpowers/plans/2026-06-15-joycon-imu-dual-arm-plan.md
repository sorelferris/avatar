# Joy-Con IMU 双臂控制实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 使用左右 Joy-Con 手柄 IMU 姿态分别控制 SO101 双臂机械臂末端位置

**架构：** 左 Joy-Con IMU 方向变化量映射到左臂末端位移，右 Joy-Con 映射到右臂。按住 L/R 键激活对应手臂控制。使用 skrobot 的 `inverse_kinematics()` 求解 IK，ViserViewer 可视化。

**技术栈：** pyjoycon, skrobot, numpy

---

## 文件结构

- 创建：`main_joycon_imu.py` — 主程序，单文件实现 Joy-Con IMU 双臂控制

---

## 实现步骤

### 任务 1：创建主程序框架

**文件：** 创建 `main_joycon_imu.py`

- [ ] **步骤 1：创建文件并写入框架代码**

```python
"""Joy-Con IMU 双臂控制

左 Joy-Con IMU -> 左臂末端位置
右 Joy-Con IMU -> 右臂末端位置
按住 L 键激活左臂控制，按住 R 键激活右臂控制
"""

from pathlib import Path
import time

import numpy as np
from skrobot import model
from skrobot.coordinates import Coordinates
from skrobot.viewers import ViserViewer

from src.joycon_utils import JoyCon

URDF = "assets/SO101/so101_new_calib.urdf"

# 控制参数
SENSE_SCALE = 0.5       # IMU 方向变化量缩放因子（m/frame）
MAX_DELTA = 0.05        # 单帧最大位置变化量（m）
MAX_RADIUS = 0.36       # 末端位置限幅半径（m）


def main() -> None:
    # 加载机器人模型
    robot = model.RobotModel.from_urdf(URDF)
    viewer = ViserViewer()
    viewer.add(robot)
    viewer.redraw()
    time.sleep(5)  # 等待 viewer 启动

    # 初始化 Joy-Con（包含校准）
    joycon = JoyCon(calibration_seconds=2.0)

    # 初始化末端位置记录（上一帧方向向量）
    prev_dir_L = None
    prev_dir_R = None

    print("Ready. 按住 L 键控制左臂，按住 R 键控制右臂。")

    while True:
        # 获取 IMU 数据
        imu = joycon.get_imu()
        status = joycon.get_status()

        dir_L = imu["L"]["direction"]
        dir_R = imu["R"]["direction"]
        l_pressed = status["L"]["buttons"].get("l", 0)
        r_pressed = status["R"]["buttons"].get("r", 0)

        # 首次初始化（等待方向向量有效）
        if prev_dir_L is None:
            prev_dir_L = np.array(dir_L)
            prev_dir_R = np.array(dir_R)
            continue

        # 姿态变化量
        delta_L = np.array(dir_L) - prev_dir_L
        delta_R = np.array(dir_R) - prev_dir_R

        # 获取当前末端位置
        larm_eef = next(x for x in robot.link_list if x.name == "left_arm_link7")
        rarm_eef = next(x for x in robot.link_list if x.name == "right_arm_link7")
        current_pos_L = larm_eef.worldpos()
        current_pos_R = rarm_eef.worldpos()

        target_pos_L = current_pos_L.copy()
        target_pos_R = current_pos_R.copy()

        # 按住 L 键：左臂跟随 IMU
        if l_pressed:
            delta_pos_L = delta_L * SENSE_SCALE
            delta_pos_L = np.clip(delta_pos_L, -MAX_DELTA, MAX_DELTA)
            target_pos_L += delta_pos_L
            # 限幅
            norm = np.linalg.norm(target_pos_L)
            if norm > MAX_RADIUS:
                target_pos_L *= MAX_RADIUS / norm

        # 按住 R 键：右臂跟随 IMU
        if r_pressed:
            delta_pos_R = delta_R * SENSE_SCALE
            delta_pos_R = np.clip(delta_pos_R, -MAX_DELTA, MAX_DELTA)
            target_pos_R += delta_pos_R
            # 限幅
            norm = np.linalg.norm(target_pos_R)
            if norm > MAX_RADIUS:
                target_pos_R *= MAX_RADIUS / norm

        # IK 求解
        larm_joints = [x for x in robot.joint_list if x.name.startswith("left_arm_joint")]
        rarm_joints = [x for x in robot.joint_list if x.name.startswith("right_arm_joint")]

        if l_pressed:
            target_L = Coordinates(pos=target_pos_L.tolist(), rot=[0, 0, 0])
            robot.inverse_kinematics(
                target_L,
                joint_list=larm_joints,
                move_target=larm_eef,
                rotation_mask=False,
                position_mask=True,
            )

        if r_pressed:
            target_R = Coordinates(pos=target_pos_R.tolist(), rot=[0, 0, 0])
            robot.inverse_kinematics(
                target_R,
                joint_list=rarm_joints,
                move_target=rarm_eef,
                rotation_mask=False,
                position_mask=True,
            )

        # 更新记录
        prev_dir_L = np.array(dir_L)
        prev_dir_R = np.array(dir_R)

        # 可视化更新
        viewer.redraw()
        time.sleep(0.033)


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：运行验证**

运行：`python main_joycon_imu.py`

预期：Joy-Con 初始化，校准完成后进入控制循环。ViserViewer 启动并显示机器人模型。

---

## 自检清单

1. 规格覆盖度：IMU 姿态映射 ✓，LR 键激活 ✓，双手独立控制 ✓，球形限幅 ✓，skrobot IK ✓
2. 占位符扫描：无 TODO/待定
3. 类型一致性：`direction` 返回 vec3，转换为 np.array 使用一致