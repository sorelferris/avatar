import threading
import time

import numpy as np
from hand_tracker import HandDetector
from skrobot import model
from skrobot.coordinates import Coordinates
from skrobot.viewers import ViserViewer

URDF = "/home/sorel/workspace/robot_description/urdf/robotic_description.urdf"
MAX_Z_ADJUST = 0.3   # 最大抬升量（安全上限）
Z_STEP = 0.05        # 每步尝试的抬升量


def move_arm(robot, eef, arm, motion: dict):
    # DO NOT CHANGE THE ORDER OF dx, dy, dz
    dx = -motion.get("depth_mm", 0)
    dy = -motion.get("x", 0)
    dz = -motion.get("y", 0)

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
    if result is not None:
        return result

    # IK 失败：逐步抬高 Z 轴，直到能伸到目标
    for z_adjust in np.arange(Z_STEP, MAX_Z_ADJUST + Z_STEP, Z_STEP):
        adjusted_pos = target_pos + np.array([0, 0, z_adjust])
        result = robot.inverse_kinematics(
            Coordinates(pos=adjusted_pos.tolist(), rot=target_rot),
            joint_list=arm,
            move_target=eef,
            rotation_mask=False,
            position_mask=True,
        )
        if result:
            return result

    return None


def main() -> None:
    # Load robot model
    robot = model.RobotModel.from_urdf(URDF)
    robot.torso_joint.joint_angle(0.75)

    for joint in robot.joint_list:
        print(f"{joint.name}: {joint.joint_angle():+.3f}, limits=[{joint.min_angle:+.3f}, {joint.max_angle:+.3f}]")

    # Find end effectors and arm joints
    larm_eef = next(x for x in robot.link_list if x.name == "left_arm_link7")
    rarm_eef = next(x for x in robot.link_list if x.name == "right_arm_link7")
    larm_joints = [x for x in robot.joint_list if x.name.startswith("left_arm_joint")]
    rarm_joints = [x for x in robot.joint_list if x.name.startswith("right_arm_joint")]

    # Create viewer
    viewer = ViserViewer()
    viewer.add(robot)
    viewer.redraw()

    # Start HandDetector in background thread
    hand_detector = HandDetector(show_window=True)
    detector_thread = threading.Thread(target=hand_detector.run, daemon=True)
    detector_thread.start()
    print("HandDetector thread started.")
    time.sleep(2)  # wait for pipeline to initialize

    print("Ready. Show open hand to control arm.")
    try:
        while True:
            # polling shared_motion
            motion_L = hand_detector.shared_motion.get("Left", {"x": 0, "y": 0, "depth_mm": 0})
            motion_R = hand_detector.shared_motion.get("Right", {"x": 0, "y": 0, "depth_mm": 0})

            move_arm(robot, larm_eef, larm_joints, motion_L)
            move_arm(robot, rarm_eef, rarm_joints, motion_R)

            viewer.redraw()
            time.sleep(0.033)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        hand_detector.close()
        for joint in robot.joint_list:
            joint.joint_angle(0.0)
        robot.torso_joint.joint_angle(0.75)


if __name__ == "__main__":
    main()
