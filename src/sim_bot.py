"""SimBot — humanoid dual-arm visualization and control layer.

Wraps a scikit-robot RobotModel + ViserViewer, with built-in Pinocchio IK
solvers for the left and right arms. Designed as a thin layer for JoyCon
teleoperation visualization; can also be driven standalone for animation
demos.
"""
import time

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

        # IK solvers (one per arm). Pinocchio model is independent of
        # the skrobot RobotModel and the ViserViewer.
        self._ik_right = IKSolver(
            urdf_path=urdf_path,
            eef_frame=right_eef_frame,
            arm_joint_names=self._right_arm_joints,
        )
        self._ik_left = IKSolver(
            urdf_path=urdf_path,
            eef_frame=left_eef_frame,
            arm_joint_names=self._left_arm_joints,
        )

    def _joint_index(self, name: str) -> int:
        """Return the index of joint `name` in robot.joint_list.

        Raises KeyError if the joint name is not found.
        """
        for i, j in enumerate(self._robot.joint_list):
            if j.name == name:
                return i
        raise KeyError(f"Unknown joint name: {name!r}")

    def _validate_side(self, side: str) -> str:
        """Validate side parameter and return it unchanged.

        Raises ValueError if side is not 'right' or 'left'.
        """
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {side!r}")
        return side

    def get_angles(self) -> np.ndarray:
        """Return the full joint angle vector of the robot."""
        return self._robot.angle_vector().copy()

    def set_joints(
        self,
        name_to_angle: dict[str, float],
        redraw: bool | None = None,
    ) -> None:
        """Set joint angles by name, keeping unspecified joints unchanged.

        Parameters
        ----------
        name_to_angle : dict[str, float]
            Mapping from joint name (as in ``robot.joint_list``) to
            target angle in radians. Joints not in the dict keep their
            current value (delta semantics).
        redraw : bool | None
            If None (default), use the constructor's ``auto_redraw``.
            Pass False to skip redraw for this call, True to force it.
        """
        current = self._robot.angle_vector().copy()
        for name, angle in name_to_angle.items():
            current[self._joint_index(name)] = float(angle)
        self._robot.angle_vector(current)

        should_redraw = self._auto_redraw if redraw is None else redraw
        if should_redraw and self._viewer is not None:
            self._viewer.redraw()

    def get_arm_angles(self, side: str) -> np.ndarray:
        """Return the joint angles of one arm in kinematic order.

        Parameters
        ----------
        side : str
            "right" or "left".
        """
        self._validate_side(side)
        names = self._right_arm_joints if side == "right" else self._left_arm_joints
        result = np.zeros(len(names))
        for i, name in enumerate(names):
            result[i] = self._robot.joint_list[self._joint_index(name)].joint_angle()
        return result

    def get_eef_position(self, side: str) -> np.ndarray:
        """Return the end-effector position in world frame.

        Parameters
        ----------
        side : str
            "right" or "left".
        """
        self._validate_side(side)
        eef_frame = self._right_eef_frame if side == "right" else self._left_eef_frame
        for link in self._robot.link_list:
            if link.name == eef_frame:
                return np.asarray(link.worldpos(), dtype=np.float64)
        raise KeyError(f"EEF frame {eef_frame!r} not found in robot link_list")

    def solve_ik(
        self,
        side: str,
        target_pos: np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run a single IK step toward the target end-effector position.

        Performs ONE damped-least-squares step. For multi-step convergence
        to a target, use ``IKSolver.solve_to_convergence()`` directly.

        Parameters
        ----------
        side : str
            "right" or "left".
        target_pos : array-like, shape (3,)
            Desired EEF position in world frame.
        q_init : array-like, shape (n_arm_joints,) | None
            Initial arm joint angles. If None, uses current angles.

        Returns
        -------
        q_new : np.ndarray, shape (n_arm_joints,)
            New arm joint angles after one IK step. Joint-limit-clamped
            and step-size-limited (max_delta=0.1 rad by default).
        """
        self._validate_side(side)
        ik = self._ik_right if side == "right" else self._ik_left
        if q_init is None:
            q_init = self.get_arm_angles(side)
        return ik.solve(
            np.asarray(q_init, dtype=np.float64),
            np.asarray(target_pos, dtype=np.float64),
        )

    def show(self, open_browser: bool = True) -> None:
        """Print viewer URL and optionally open in browser.

        No-op if ``viewer=False`` was passed to the constructor.
        """
        if self._viewer is not None:
            self._viewer.show(open_browser=open_browser)

    def redraw(self) -> None:
        """Force a viewer redraw. No-op if viewer is disabled."""
        if self._viewer is not None:
            self._viewer.redraw()

    def wait_until_close(self) -> None:
        """Block until KeyboardInterrupt (Ctrl-C). No-op if viewer disabled."""
        if self._viewer is not None:
            self._viewer.wait_until_close()
        else:
            # Block on KeyboardInterrupt for parity with viewer mode.
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass

    def close(self) -> None:
        """Close the viewer. Idempotent; no-op if viewer disabled."""
        if self._viewer is not None:
            self._viewer.close()