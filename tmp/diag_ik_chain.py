"""Inspect the joint chain skrobot builds from the URDF."""

from pathlib import Path

import numpy as np
from skrobot import model

URDF = Path("/home/sorel/workspace/robot_description/urdf/robotic_description.urdf")
robot = model.RobotModel.from_urdf(URDF)

print("=== All joints (parsed by skrobot) ===")
for j in robot.joint_list:
    print(
        f"  {j.name}: type={j.type}  limits=[{j.min_angle:+.3f}, {j.max_angle:+.3f}]  init={j.joint_angle():+.4f}"
    )

print("\n=== angle_vector length ===")
av = robot.angle_vector()
print(f"  len(av) = {len(av)}")
print(f"  av = {av}")

print("\n=== Parent chain from left_arm_link7 back to root ===")
target = [l for l in robot.link_list if l.name == "left_arm_link7"][0]
link = target
while link is not None:
    pj = link.parent_joint
    print(
        f"  link={link.name}  parent_joint={pj.name if pj else None}  type={pj.type if pj else None}"
    )
    link = pj.parent_link if pj else None

print("\n=== Try setting a single joint and observe worldpos change ===")
move_target = [l for l in robot.link_list if l.name == "left_arm_link7"][0]
ee0 = move_target.worldpos().copy()
print(f"  initial ee worldpos = {ee0}")

robot.left_arm_joint1.joint_angle(0.5)
ee1 = move_target.worldpos().copy()
print(f"  after joint1 = 0.5 rad: ee = {ee1}  delta = {np.linalg.norm(ee1 - ee0):.4f}")

robot.left_arm_joint3.joint_angle(-0.5)
ee2 = move_target.worldpos().copy()
print(
    f"  after joint3 = -0.5 rad: ee = {ee2}  delta from initial = {np.linalg.norm(ee2 - ee0):.4f}"
)
