"""Minimal IK test using joint_list explicitly and position-only."""

from pathlib import Path

import numpy as np
from skrobot import model
from skrobot.coordinates import Coordinates

URDF = Path("/home/sorel/workspace/robot_description/urdf/robotic_description.urdf")
robot = model.RobotModel.from_urdf(URDF)
robot.torso_joint.joint_angle(0.75)

mt = [l for l in robot.link_list if l.name == "left_arm_link7"][0]
ee0 = mt.worldpos().copy()
print(f"initial ee = {ee0}")

# Build explicit joint_list for left arm + torso
arm_joints = [
    j
    for j in robot.joint_list
    if j.name
    in (
        "torso_joint",
        "left_arm_joint1",
        "left_arm_joint2",
        "left_arm_joint3",
        "left_arm_joint4",
        "left_arm_joint5",
        "left_arm_joint6",
        "left_arm_joint7",
    )
]
print(f"arm_joints = {[j.name for j in arm_joints]}")

# Build link_list (one per joint, the link AFTER the joint)
arm_links = []
for j in arm_joints:
    arm_links.append(j.child_link)
print(f"arm_links = {[l.name for l in arm_links]}")

# Tiny target offset from initial pose
target_pos = ee0 + np.array([0.0, 0.0, 0.05])
print(f"\nTarget pos = {target_pos} (5cm above initial)")

target = Coordinates(pos=target_pos.tolist())
print(f"target coords worldpos = {target.worldpos()}  (should equal pos)")

print("\n--- IK attempt 1: position only, link_list, move_target=link7 ---")
result = robot.inverse_kinematics(
    target,
    link_list=arm_links,
    move_target=mt,
    rotation_mask=False,
    position_mask=True,
    stop=200,
    thre=0.001,
)
print(f"  result = {result}")
print(f"  ee after = {mt.worldpos().round(4)}")

# Reset and try with joint_list (recommended spelling)
print("\n--- Reset and try IK attempt 2: joint_list spelling ---")
robot.torso_joint.joint_angle(0.75)
for j in robot.joint_list:
    if j.name.startswith("left_arm_joint"):
        j.joint_angle(0.0)
print(f"  reset ee = {mt.worldpos().round(4)}")

result = robot.inverse_kinematics(
    target,
    joint_list=arm_joints,
    move_target=mt,
    rotation_mask=False,
    position_mask=True,
    stop=200,
    thre=0.001,
)
print(f"  result = {result}")
print(f"  ee after = {mt.worldpos().round(4)}")

print("\n--- IK attempt 3: pos+rot, joint_list ---")
result = robot.inverse_kinematics(
    target,
    joint_list=arm_joints,
    move_target=mt,
    rotation_mask=True,
    position_mask=True,
    stop=200,
    thre=0.001,
    rthre=np.deg2rad(1.0),
)
print(f"  result = {result}")
print(f"  ee after = {mt.worldpos().round(4)}")
print(f"  ee rot =\n{mt.worldrot().round(3)}")
