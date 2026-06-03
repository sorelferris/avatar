"""SimBot — humanoid dual-arm visualization and control layer.

Wraps a scikit-robot RobotModel + ViserViewer, with built-in Pinocchio IK
solvers for the left and right arms. Designed as a thin layer for JoyCon
teleoperation visualization; can also be driven standalone for animation
demos.
"""
import numpy as np
from skrobot.model import RobotModel
from skrobot.viewers import ViserViewer

from src.ik_solver import IKSolver


_DEFAULT_RIGHT_ARM = [f"right_arm_joint{i}" for i in range(1, 8)]
_DEFAULT_LEFT_ARM = [f"left_arm_joint{i}" for i in range(1, 8)]


class SimBot:
    """Dual-arm robot simulator with real-time viser visualization.

    Parameters
    ----------
    urdf_path : str
        Path to the robot URDF file.
    right_arm_joints : list[str] | None
        Names of right arm joints, in kinematic order. Defaults to
        ``["right_arm_joint1", ..., "right_arm_joint7"]``.
    left_arm_joints : list[str] | None
        Names of left arm joints. Defaults to humanoid 7-DoF names.
    right_eef_frame : str
        Pinocchio frame name for the right end-effector. Must exist in
        the URDF as a frame.
    left_eef_frame : str
        Pinocchio frame name for the left end-effector.
    viewer : bool
        If True (default), start a ViserViewer for browser visualization.
        Set False for headless testing.
    auto_redraw : bool
        If True (default), every ``set_joints`` call triggers
        ``viewer.redraw()``. Pass ``redraw=False`` to a single call to
        override.
    """

    def __init__(
        self,
        urdf_path: str,
        right_arm_joints: list[str] | None = None,
        left_arm_joints: list[str] | None = None,
        right_eef_frame: str = "right_arm_link7",
        left_eef_frame: str = "left_arm_link7",
        viewer: bool = True,
        auto_redraw: bool = True,
    ) -> None:
        self._robot = RobotModel.from_urdf(urdf_path)
        self._right_arm_joints = right_arm_joints or _DEFAULT_RIGHT_ARM
        self._left_arm_joints = left_arm_joints or _DEFAULT_LEFT_ARM
        self._right_eef_frame = right_eef_frame
        self._left_eef_frame = left_eef_frame
        self._auto_redraw = auto_redraw

        if viewer:
            self._viewer = ViserViewer()
            self._viewer.add(self._robot)
        else:
            self._viewer = None