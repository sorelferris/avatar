"""Test right arm reach + closer left-arm targets."""

from pathlib import Path

import numpy as np
from skrobot import model
from skrobot.coordinates import Coordinates

URDF = Path("/home/sorel/workspace/robot_description/urdf/robotic_description.urdf")
robot = model.RobotModel.from_urdf(URDF)
robot.torso_joint.joint_angle(0.75)


def make_chain(prefix):
    joints = [
        j
        for j in robot.joint_list
        if j.name == "torso_joint" or j.name.startswith(prefix + "_joint")
    ]
    links = [j.child_link for j in joints]
    mt = [l for l in robot.link_list if l.name == f"{prefix}_link7"][0]
    return joints, links, mt


def try_ik(joints, links, mt, target_pos, target_rpy_deg=(0, 0, 0), label=""):
    target = Coordinates(
        pos=target_pos,
        rot=[np.deg2rad(r) for r in target_rpy_deg],
    )
    try:
        result = robot.inverse_kinematics(
            target,
            joint_list=joints,
            move_target=mt,
            rotation_mask=False,
            position_mask=True,
            stop=500,
            thre=0.001,
        )
    except Exception as e:
        print(f"  {label}: EXC {type(e).__name__}: {e}")
        return
    if result is False:
        ee = mt.worldpos()
        d = np.linalg.norm(ee - np.array(target_pos))
        print(f"  {label}: FAIL  final_err={d:.3f}m  ee={ee.round(3)}")
    else:
        ee = mt.worldpos()
        d = np.linalg.norm(ee - np.array(target_pos))
        print(f"  {label}: OK    err={d:.4f}m  ee={ee.round(3)}")


# Left arm: chain summary
lj, ll, lmt = make_chain("left_arm")
rj, rl, rmt = make_chain("right_arm")
print(f"left  initial ee = {lmt.worldpos().round(3)}")
print(f"right initial ee = {rmt.worldpos().round(3)}")

print("\n=== Left arm: progressively closer targets (1.13m down to 0.4m) ===")
for d in [1.13, 0.90, 0.75, 0.60, 0.45, 0.30]:
    target_pos = np.array([0.1737, 0.1960, 0.7872]) + np.array(
        [d / np.sqrt(3), d / np.sqrt(3), d / np.sqrt(3)]
    )
    try_ik(lj, ll, lmt, target_pos.tolist(), label=f"left arm d={d}m")

print("\n=== Original target (1.13m, user spec) ===")
try_ik(
    lj, ll, lmt, [0.8, -0.3, 0.8], (0, 30, -30), label="left arm [0.8,-0.3,0.8] pos+rot"
)

print("\n=== Same target on right arm (y mirrored?) ===")
# right arm is mirrored; targets with negative y may be easier for right
try_ik(
    rj,
    rl,
    rmt,
    [0.8, -0.3, 0.8],
    (0, 30, -30),
    label="right arm [0.8,-0.3,0.8] pos+rot",
)
try_ik(
    rj, rl, rmt, [0.8, 0.3, 0.8], (0, 30, -30), label="right arm [0.8, 0.3,0.8] pos+rot"
)
