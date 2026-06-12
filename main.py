from pathlib import Path
import time

import numpy as np
from skrobot import model
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import matrix2rpy
from skrobot.viewers import ViserViewer


# ----------------------------------------------------------------------- main
def main(urdf: Path):
    assert urdf.exists(), f"URDF file {urdf} does not exist."

    # ---------------------------------------------------------------- data
    robot = model.RobotModel.from_urdf(urdf)
    robot.torso_joint.joint_angle(0.75)

    for joint in robot.joint_list:
        print(
            f"{joint.name}: {joint.joint_angle():+.3f}, "
            f"limits=[{joint.min_angle:+.3f}, {joint.max_angle:+.3f}]"
        )

    larm_eef = next(x for x in robot.link_list if x.name == "left_arm_link7")
    rarm_eef = next(x for x in robot.link_list if x.name == "right_arm_link7")
    larm_joints = [x for x in robot.joint_list if x.name.startswith("left_arm_joint")]
    rarm_joints = [x for x in robot.joint_list if x.name.startswith("right_arm_joint")]

    # ---------------------------------------------------------------- viewer
    viewer = ViserViewer()
    viewer.add(robot)
    viewer.show()  # starts HTTP/WS server — does NOT block
    viewer.redraw()  # push initial pose
    time.sleep(5)  # wait for viewer to start

    target_pos = larm_eef.worldpos() + np.array([0.00, 0.0, 0.35])
    target = Coordinates(pos=target_pos.tolist(), rot=[0.1, 0, 0])
    result = robot.inverse_kinematics(
        target,
        joint_list=larm_joints,
        move_target=larm_eef,
        rotation_mask=True,
        position_mask=True,
    )
    print(target)
    print(result)
    viewer.redraw()

    # -------------------------------------------------------------- blocking
    viewer._server.sleep_forever()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the main function with a specified URDF file."
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="assets/SO101/so101_new_calib.urdf",
        help="Path to the URDF file.",
    )
    args = parser.parse_args()
    main(urdf=Path(args.urdf))
