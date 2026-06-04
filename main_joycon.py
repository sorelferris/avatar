"""JoyCon teleoperation for SO101 robot arm.

Right stick X/Y -> arm X/Y position
R/ZR shoulders -> Z axis up/down
A button -> gripper toggle
"""

import time
import numpy as np

from src.joycon_utils import JoyCon
from src.sim_env import MujocoEnvironment
from src.ik_solver import IKSolver

URDF = "assets/SO101/so101_new_calib.urdf"
XML = "assets/SO101/scene.xml"

GRIPPER_CLOSED = 0.0

# 控制参数
STICK_SENSITIVITY = 0.005  # 每帧位置增量
Z_SENSITIVITY = 0.005  # Z 轴每帧增量
MAX_POSITION_DELTA = 0.05  # 每步最大位置变化


def main() -> None:
    joycon = JoyCon(calibration_seconds=2.0)
    env = MujocoEnvironment(XML)
    ik = IKSolver(URDF, damping=0.1, max_delta=0.1)

    # 初始化夹爪状态
    gripper_pos = env.gripper_range[1] * 0.8

    # 获取初始末端位置作为控制起点
    initial_pos = env.get_eef_position()
    current_target = initial_pos.copy()
    print(f"Initial EEF position: {initial_pos}")

    try:
        joycon.get_R_analog()  # 初始化 A_pressed 状态
        while True:
            # 读取 JoyCon 输入
            dx, dy = joycon.get_R_analog()
            z_dir = joycon.get_R_shoulder()

            # 计算位置增量
            delta = np.array(
                [dx * STICK_SENSITIVITY, -dy * STICK_SENSITIVITY, z_dir * Z_SENSITIVITY]
            )
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
                gripper_pos = (
                    GRIPPER_CLOSED
                    if gripper_pos > sim.gripper_range[1] * 0.4
                    else sim.gripper_range[1] * 0.8
                )
                print(f"Gripper: {'open' if gripper_pos > 0.6 else 'closed'}")

            sim.set_gripper(gripper_pos)

            # 物理步骤
            for _ in range(10):
                sim.step()

            # 可视化更新
            viz.update_viewer(None, current_target)

            time.sleep(0.02)

            # 按键退出
            # （JoyCon 无键盘输入，通过 MuJoCo viewer 窗口关闭或手动 Ctrl+C）

    except KeyboardInterrupt:
        print("\nExiting...")

    print("Done.")


if __name__ == "__main__":
    main()
