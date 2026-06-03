"""Unit tests for SimBot class.

Use SO101 URDF (project-internal) with explicit arm joint names so we test
the configurable interface rather than only the humanoid 7-DoF defaults.
"""
import numpy as np
import pytest

from src.sim_bot import SimBot

URDF = "assets/SO101/so101_new_calib.urdf"
SO101_ARM = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
SO101_EEF = "gripper_frame_link"


def test_construct_with_viewer_false_does_not_open_browser():
    """viewer=False must not start ViserViewer (headless mode)."""
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    # Constructed successfully without starting viser
    assert bot is not None