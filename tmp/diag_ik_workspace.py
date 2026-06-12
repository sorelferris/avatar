"""Test IK at progressively further targets to map the workspace."""

from pathlib import Path

import numpy as np
from skrobot import model
from skrobot.coordinates import Coordinates

URDF = Path("/home/sorel/workspace/robot_description/urdf/robotic_description.urdf")
robot = model.RobotModel.from_urdf(URDF)

robot.torso_joint.joint_angle(0.75)

move_target = [l for l in robot.link_list if l.name == "left_arm_link7"][0]
larm_links = [l for l in robot.link_list if l.name.startswith("left")]


def try_ik(target_pos, target_rpy_deg, label):
    target = Coordinates(
        pos=target_pos,
        rot=[np.deg2rad(r) for r in target_rpy_deg],
    )
    try:
        result = robot.inverse_kinematics(
            target,
            link_list=larm_links,
            move_target=move_target,
            rotation_mask=True,
            position_mask=True,
            stop=2000,
            thre=0.001,
            rthre=np.deg2rad(1.0),
        )
    except Exception as e:
        print(f"  {label}: EXC {type(e).__name__}: {e}")
        return
    if result is False:
        ee_p = move_target.worldpos()
        print(f"  {label}: FAIL final_pos={ee_p.round(3)}")
    else:
        ee_p = move_target.worldpos()
        ee_r = move_target.worldrot()
        R_err = ee_r.T @ target.rotation
        cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rot_err = np.rad2deg(np.arccos(cos_theta))
        print(f"  {label}: OK pos={ee_p.round(3)} rot_err={rot_err:.2f}°")


# (1) Easy: tiny offset from initial pose
ee0 = move_target.worldpos().copy()
print(f"Initial ee worldpos = {ee0}")

# Try several reachable-ish targets
candidates = [
    ("+0.05x", [ee0[0] + 0.05, ee0[1], ee0[2]], [0, 0, 0]),
    ("+0.10x", [ee0[0] + 0.10, ee0[1], ee0[2]], [0, 0, 0]),
    ("+0.20x", [ee0[0] + 0.20, ee0[1], ee0[2]], [0, 0, 0]),
    ("+0.40x", [ee0[0] + 0.40, ee0[1], ee0[2]], [0, 0, 0]),
    ("+0.60x", [ee0[0] + 0.60, ee0[1], ee0[2]], [0, 0, 0]),
    ("orig-target", [0.8, -0.3, 0.8], [0, 30, -30]),
    ("origin-frame-0", [0.3, 0.0, 0.5], [0, 0, 0]),
    ("origin-frame-1", [0.5, 0.0, 0.3], [0, 0, 0]),
    ("origin-frame-2", [0.0, 0.5, 0.3], [0, 0, 0]),
]

for label, p, r in candidates:
    try_ik(p, r, label)
