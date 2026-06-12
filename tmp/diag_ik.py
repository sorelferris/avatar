"""Diagnostic: print end-effector initial pose vs target, and inspect larm_link_list."""

from pathlib import Path

import numpy as np
from skrobot import model
from skrobot.coordinates import Coordinates

URDF = Path("/home/sorel/workspace/robot_description/urdf/robotic_description.urdf")
robot = model.RobotModel.from_urdf(URDF)

# 1) Torso lift (mirrors main.py)
robot.torso_joint.joint_angle(0.75)

# 2) Default arm angle vector (all zeros unless URDF has non-zero defaults)
print("=== Default angle vector ===")
for j in robot.joint_list:
    print(f"  {j.name}: {j.joint_angle():.4f}")

# 3) Inspect larm_link_list — order matters, last one is move_target
larm_link_list = [link for link in robot.link_list if link.name.startswith("left")]
print("\n=== larm_link_list (order = insertion order from URDF) ===")
for i, link in enumerate(larm_link_list):
    print(f"  [{i}] {link.name}")
print(f"\n  move_target = larm_link_list[-1] = {larm_link_list[-1].name}")

# 4) Initial end-effector world pose
ee = larm_link_list[-1]
ee_pos = ee.worldpos()
ee_rot = ee.worldrot()
print(f"\n=== Initial end-effector pose ===")
print(f"  pos = {ee_pos}")
print(f"  rot (3x3) =\n{ee_rot}")

# 5) Target pose
target = Coordinates(
    pos=[0.8, -0.3, 0.8],
    rot=[0.0, np.deg2rad(30), np.deg2rad(-30)],
)
print(f"\n=== Target pose (Coordinates defaults to world frame) ===")
print(f"  pos = {target.translation}")
print(f"  rotation matrix =\n{target.rotation}")
print(f"  euler rpy (deg) = {np.rad2deg(target.rpy())}")

# 6) Distance + orientation error
dist = np.linalg.norm(ee_pos - target.translation)
print(f"\n=== Initial error ===")
print(f"  position error: {dist:.4f} m")

# Orientation error: trace of R_err^T R_target = 1 + 2cos(theta)
R_err = ee_rot.T @ target.rotation
cos_theta = (np.trace(R_err) - 1.0) / 2.0
cos_theta = np.clip(cos_theta, -1.0, 1.0)
print(f"  orientation error: {np.rad2deg(np.arccos(cos_theta)):.2f} deg")

# 7) Joint limits — IK may saturate near bounds
print("\n=== Left-arm joint limits ===")
for j in robot.joint_list:
    if j.name.startswith("left_arm_joint"):
        print(
            f"  {j.name}: [{j.min_angle:+.3f}, {j.max_angle:+.3f}]  init={j.joint_angle():+.3f}"
        )

# 8) Sanity: is move_target actually an end-effector? Check its world position relative to fingertip links
print("\n=== Last few links of left arm (likely finger area) ===")
for link in larm_link_list[-6:]:
    p = link.worldpos()
    print(f"  {link.name}: pos = {p}")
