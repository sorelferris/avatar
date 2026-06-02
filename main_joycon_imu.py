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

    # 获取初始末端位置
    initial_pos = sim.get_eef_position()
    current_target_pos = initial_pos.copy()
    print(f"Initial EEF position: {initial_pos}")

    try:
        viz.start_viewer()
    except Exception:
        print("MuJoCo viewer not available, running headless")

    # ZR 状态跟踪
    zr_was_pressed = False

    try:
        while True:
            # ZR 上升沿：记录基准（在 get_R_imu_delta 内部处理）
            # ZR 下降沿：重置基准
            if zr_was_pressed and not joycon.is_ZR_pressed():
                joycon.reset_imu_baseline()
                # 重置位置到初始位置
                current_target_pos = initial_pos.copy()
                print("ZR released - baseline reset, position restored")
            zr_was_pressed = joycon.is_ZR_pressed()

            # 获取 IMU 相对偏移
            delta = joycon.get_R_imu_delta()
            att_delta = delta["attitude"]

            # 姿态偏移（Roll/Pitch/Yaw）-> 位置增量
            position_change = np.array([
                att_delta.x * ATTITUDE_SENSITIVITY,
                att_delta.y * ATTITUDE_SENSITIVITY,
                att_delta.z * ATTITUDE_SENSITIVITY,
            ])
            position_change = np.clip(position_change, -MAX_POSITION_DELTA, MAX_POSITION_DELTA)
            current_target_pos += position_change

            # 限幅（球形工作空间，相对于初始位置）
            max_radius = 0.36
            relative_pos = current_target_pos - initial_pos
            norm = np.linalg.norm(relative_pos)
            if norm > max_radius:
                current_target_pos = initial_pos + relative_pos * (max_radius / norm)

            # IK 求解
            q_current = sim.get_joint_positions()
            q_new = ik.solve(q_current, current_target_pos)
            sim.set_control(q_new)

            # 夹爪切换检测
            gripper_threshold = sim.gripper_range[1] * 0.1
            if joycon.get_gripper_toggle():
                gripper_pos = 0.0 if gripper_pos > gripper_threshold else sim.gripper_range[1] * 0.8
                state = "open" if gripper_pos > gripper_threshold else "closed"
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