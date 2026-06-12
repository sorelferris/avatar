from pathlib import Path
import time

import numpy as np
from skrobot import model
from skrobot.coordinates import Coordinates
from skrobot.viewers import ViserViewer


def main(urdf: Path):
    assert urdf.exists(), f"URDF file {urdf} does not exist."

    # ---------------------------------------------------------------- data
    # Robot is the data source. We can build and mutate it independently of
    # any viewer.
    robot = model.RobotModel.from_urdf(urdf)
    robot.torso_joint.joint_angle(0.75)  # Set torso to 0.75
    for joint in robot.joint_list:
        print(
            f"{joint.name}: {joint.joint_angle()}, limits=[{joint.min_angle}, {joint.max_angle}]"
        )
    larm_links = [x for x in robot.link_list if x.name.startswith("left")]
    larm_eef = [x for x in larm_links if x.name == "left_arm_link7"][0]
    print(f"L arm links: {[x.name for x in larm_links]}")
    print(f"L arm eef: {larm_eef.name}")
    rarm_links = [x for x in robot.link_list if x.name.startswith("right")]
    rarm_eef = [x for x in rarm_links if x.name == "right_arm_link7"][0]
    print(f"R arm links: {[x.name for x in rarm_links]}")
    print(f"R arm eef: {rarm_eef.name}")
    # -------------------------------------------------------------- viewer
    # Viewer is an output sink. It snapshots the current robot pose when
    # redraw() is called, so we can interleave robot mutations and redraws
    # in any order from the main thread.
    viewer = ViserViewer()
    viewer.add(robot)
    viewer.show()  # starts the HTTP/WS server, prints URL — does NOT block

    # Example: IK on the left arm — note the redraw after the solve.
    # Example: IK on the left arm.
    #
    # Workspace note: this 7-DOF arm + torso reach is ~0.6 m from the
    # default pose. Targets beyond that distance will fail no matter how
    # many iterations you give the solver — the error plateaus at the
    # reachable boundary. Keep targets inside that radius.
    move_target = [l for l in robot.link_list if l.name == "left_arm_link7"][0]
    target_coords = Coordinates(
        pos=[0.50, 0.10, 0.70],  # ~0.40 m from initial pose, comfortably reachable
        rot=[0.0, np.deg2rad(30), np.deg2rad(-30)],
    )

    # Recommended API: joint_list (acts on joints directly). link_list is
    # the legacy spelling and works, but joint_list avoids a downstream
    # union_link_list derivation step.
    arm_joints = [
        j
        for j in robot.joint_list
        if j.name == "torso_joint" or j.name.startswith("left_arm_joint")
    ]

    result = robot.inverse_kinematics(
        target_coords,
        joint_list=arm_joints,
        move_target=move_target,
        rotation_mask=True,
        position_mask=True,
        stop=2000,
        thre=0.001,
        rthre=np.deg2rad(1.0),
    )

    # Diagnostic: how close did we actually get?
    final_pos = move_target.worldpos()
    final_rot = move_target.worldrot()
    pos_err = np.linalg.norm(final_pos - target_coords.translation)
    R_err = final_rot.T @ target_coords.rotation
    cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = np.rad2deg(np.arccos(cos_theta))

    print(f"Left arm IK result: {result}")
    print(f"  target pos = {target_coords.translation.round(3)}")
    print(f"  reached    = {final_pos.round(3)}")
    print(f"  pos err    = {pos_err * 1000:.2f} mm")
    print(f"  rot err    = {rot_err_deg:.2f} deg")
    if result is not False and pos_err < 0.005:
        print("IK solved within 5 mm tolerance.")
        viewer.redraw()
        time.sleep(0.05)
    else:
        print(
            f"IK did NOT converge within tolerance. "
            f"Try a closer target (current distance ~{pos_err:.2f} m, "
            f"reachable radius from initial pose is ~0.6 m)."
        )

    # -------------------------------------------------------------- blocking
    # ViserViewer.show() does NOT block. Without sleep_forever() the process
    # exits and the viser server shuts down immediately. Only call this
    # after all your setup is done — anything you want to visualize must
    # happen above.
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
