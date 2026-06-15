import time

import numpy as np
from skrobot import model
from skrobot.coordinates import Coordinates
from skrobot.viewers import ViserViewer

from src.joycon_utils import JoyCon

URDF = "/home/sorel/workspace/robot_description/urdf/robotic_description.urdf"
STEP_SIZE = 0.05


def move_arm(robot, eef, arm, dx: float, dy: float, dz: float):
    if dx == 0 and dy == 0 and dz == 0:
        return None
    target_pos = eef.worldpos() + np.array([dx * STEP_SIZE, dy * STEP_SIZE, dz * 0.5 * STEP_SIZE])
    target_rot = [0, 0, 0]  # No rotation change
    result = robot.inverse_kinematics(
        Coordinates(pos=target_pos.tolist(), rot=target_rot),
        joint_list=arm,
        move_target=eef,
        rotation_mask=False,
        position_mask=True,
    )
    return result


def main() -> None:
    # Load robot model and set initial pose
    robot = model.RobotModel.from_urdf(URDF)
    robot.torso_joint.joint_angle(0.75)

    for joint in robot.joint_list:
        print(f"{joint.name}: {joint.joint_angle():+.3f}, limits=[{joint.min_angle:+.3f}, {joint.max_angle:+.3f}]")

    # Get left and right arm end effector links and joints
    larm_eef = next(x for x in robot.link_list if x.name == "left_arm_link7")
    rarm_eef = next(x for x in robot.link_list if x.name == "right_arm_link7")
    larm_joints = [x for x in robot.joint_list if x.name.startswith("left_arm_joint")]
    rarm_joints = [x for x in robot.joint_list if x.name.startswith("right_arm_joint")]

    viewer = ViserViewer()
    viewer.add(robot)
    viewer.redraw()

    # Initialize Joy-Con with calibration
    joycon = JoyCon(calibration_seconds=2.0)

    print("Ready.")

    while True:
        status = joycon.get_status()

        ly, lx = joycon.get_L_analog()
        ry, rx = joycon.get_R_analog()

        lz = status["L"]["buttons"].get("l", 0) - status["L"]["buttons"].get("zl", 0)
        rz = status["R"]["buttons"].get("r", 0) - status["R"]["buttons"].get("zr", 0)

        move_arm(robot, larm_eef, larm_joints, lx, ly, lz)
        move_arm(robot, rarm_eef, rarm_joints, rx, ry, rz)

        reset = status["R"]["buttons"].get("home", 0)
        if reset:
            for joint in robot.joint_list:
                joint.joint_angle(0.0)
            robot.torso_joint.joint_angle(0.75)

        viewer.redraw()
        time.sleep(0.033)


if __name__ == "__main__":
    main()
