"""Joy-Con IMU 双臂控制

左 Joy-Con IMU -> 左臂末端位置（按住 L 键激活）
右 Joy-Con IMU -> 右臂末端位置（按住 R 键激活）
"""

import time

import numpy as np
from skrobot import model
from skrobot.coordinates import Coordinates
from skrobot.viewers import ViserViewer

from src.joycon_utils import JoyCon

URDF = "/home/sorel/workspace/robot_description/urdf/robotic_description.urdf"

# 控制参数
SENSE_SCALE = 0.5  # IMU 方向变化量缩放因子（m/frame）
MAX_DELTA = 0.05  # 单帧最大位置变化量（m）
MAX_RADIUS = 0.5  # 末端位置限幅半径（m）


def main() -> None:
    # 加载机器人模型
    robot = model.RobotModel.from_urdf(URDF)
    robot.torso_joint.joint_angle(0.75)

    for joint in robot.joint_list:
        print(
            f"{joint.name}: {joint.joint_angle():+.3f}, "
            f"limits=[{joint.min_angle:+.3f}, {joint.max_angle:+.3f}]"
        )

    # 双臂末端与关节
    larm_eef = next(x for x in robot.link_list if x.name == "left_arm_link7")
    rarm_eef = next(x for x in robot.link_list if x.name == "right_arm_link7")
    larm_joints = [x for x in robot.joint_list if x.name.startswith("left_arm_joint")]
    rarm_joints = [x for x in robot.joint_list if x.name.startswith("right_arm_joint")]

    viewer = ViserViewer()
    viewer.add(robot)
    viewer.redraw()
    # time.sleep(5)  # 等待 viewer 启动

    # 初始化 Joy-Con（包含校准）
    joycon = JoyCon(calibration_seconds=2.0)

    # 上一帧 IMU 方向
    prev_dir_L = None
    prev_dir_R = None

    print("Ready. 按住 L 键控制左臂，按住 R 键控制右臂。")

    while True:
        imu = joycon.get_imu()
        status = joycon.get_status()

        dir_L = imu["L"]["direction"]
        dir_R = imu["R"]["direction"]
        l_pressed = status["L"]["buttons"].get("zl", 0)
        r_pressed = status["R"]["buttons"].get("zr", 0)

        # 首次初始化
        if prev_dir_L is None:
            prev_dir_L = np.array(dir_L)
            prev_dir_R = np.array(dir_R)
            continue

        # IMU 变化量
        delta_L = np.array(dir_L) - prev_dir_L
        delta_R = np.array(dir_R) - prev_dir_R

        # L 键：左臂位置 IK
        if l_pressed:
            current_pos_L = larm_eef.worldpos()
            print(f"当前左臂末端位置: {current_pos_L}")
            target_pos_L = current_pos_L.copy()

            delta_pos_L = delta_L * SENSE_SCALE
            delta_pos_L = np.clip(delta_pos_L, -MAX_DELTA, MAX_DELTA)
            target_pos_L += delta_pos_L
            print(f"目标左臂末端位置: {target_pos_L}")

            norm = np.linalg.norm(target_pos_L)
            if norm > MAX_RADIUS:
                target_pos_L *= MAX_RADIUS / norm

            target_coords = Coordinates(pos=target_pos_L.tolist(), rot=[0, 0, 0])
            result = robot.inverse_kinematics(
                target_coords,
                joint_list=larm_joints,
                move_target=larm_eef,
                rotation_mask=False,
                position_mask=True,
            )
            print(f"IK 结果: {result}")

        # R 键：右臂位置 IK
        if r_pressed:
            current_pos_R = rarm_eef.worldpos()
            target_pos_R = current_pos_R.copy()

            delta_pos_R = delta_R * SENSE_SCALE
            delta_pos_R = np.clip(delta_pos_R, -MAX_DELTA, MAX_DELTA)
            target_pos_R += delta_pos_R

            norm = np.linalg.norm(target_pos_R)
            if norm > MAX_RADIUS:
                target_pos_R *= MAX_RADIUS / norm

            target_coords = Coordinates(pos=target_pos_R.tolist(), rot=[0, 0, 0])
            robot.inverse_kinematics(
                target_coords,
                joint_list=rarm_joints,
                move_target=rarm_eef,
                rotation_mask=False,
                position_mask=True,
            )

        # 更新记录
        prev_dir_L = np.array(dir_L)
        prev_dir_R = np.array(dir_R)

        # 可视化
        viewer.redraw()
        time.sleep(0.033)


if __name__ == "__main__":
    main()
