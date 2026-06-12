"""Sweep IK configurations to find one that converges."""

from pathlib import Path

import numpy as np
from skrobot import model
from skrobot.coordinates import Coordinates

URDF = Path("/home/sorel/workspace/robot_description/urdf/robotic_description.urdf")
robot = model.RobotModel.from_urdf(URDF)

# Candidates for move_target
ee_candidates = {
    "left_arm_link7": [l for l in robot.link_list if l.name == "left_arm_link7"][0],
    "left_finger_center_link": [
        l for l in robot.link_list if l.name == "left_finger_center_link"
    ][0],
    "left_left_finger_link": [
        l for l in robot.link_list if l.name == "left_left_finger_link"
    ][0],
}

# IK configs: (label, move_target, rotation_mask, stop)
configs = []
for ee_name, ee in ee_candidates.items():
    configs.append((f"{ee_name} | pos-only", ee, False, 1000))
    configs.append((f"{ee_name} | pos+rot", ee, True, 1000))
    configs.append((f"{ee_name} | pos+rot | stop=5000", ee, True, 5000))


def reset_robot():
    robot.torso_joint.joint_angle(0.75)
    for j in robot.joint_list:
        if j.name.startswith("left_arm_joint"):
            j.joint_angle(0.0)


def evaluate(label, move_target, rotation_mask, stop):
    reset_robot()
    target = Coordinates(
        pos=[0.8, -0.3, 0.8],
        rot=[0.0, np.deg2rad(30), np.deg2rad(-30)],
    )
    larm_links = [l for l in robot.link_list if l.name.startswith("left")]
    try:
        result = robot.inverse_kinematics(
            target,
            link_list=larm_links,
            move_target=move_target,
            rotation_mask=rotation_mask,
            position_mask=True,
            stop=stop,
            thre=0.001,
            rthre=np.deg2rad(1.0),
        )
    except Exception as e:
        return f"{label}: EXCEPTION {type(e).__name__}: {e}"

    if result is False:
        # measure final error
        ee_pos = move_target.worldpos()
        ee_rot = move_target.worldrot()
        pos_err = np.linalg.norm(ee_pos - target.translation)
        R_err = ee_rot.T @ target.rotation
        cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rot_err = np.rad2deg(np.arccos(cos_theta))
        return f"FAIL | {label} | pos_err={pos_err:.3f}m rot_err={rot_err:.1f}°"
    else:
        ee_pos = move_target.worldpos()
        ee_rot = move_target.worldrot()
        pos_err = np.linalg.norm(ee_pos - target.translation)
        R_err = ee_rot.T @ target.rotation
        cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rot_err = np.rad2deg(np.arccos(cos_theta))
        return f"OK   | {label} | pos_err={pos_err:.4f}m rot_err={rot_err:.2f}°"


for label, mt, rm, stop in configs:
    print(evaluate(label, mt, rm, stop))
