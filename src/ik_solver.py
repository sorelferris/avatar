"""Independent IK solver using Pinocchio.

No MuJoCo dependency — can be used for both simulation and real robot control.
"""

import numpy as np
import pinocchio as pin


class IKSolver:
    """Position-based IK solver for serial chain arms using Pinocchio.

    Uses damped least-squares (DLS) Jacobian IK to solve for joint positions
    that place the end-effector at a target 3D position.
    """

    def __init__(
        self,
        urdf_path: str,
        eef_frame: str = "gripper_frame_link",
        arm_joint_names: list[str] | None = None,
        damping: float = 0.1,
        max_delta: float = 0.1,
    ) -> None:
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()

        # EEF frame
        self._eef_id = self._model.getFrameId(eef_frame)

        # Arm joints (exclude gripper/fixed joints)
        if arm_joint_names is None:
            if "so102" in urdf_path:
                arm_joint_names = [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_yaw",
                    "wrist_roll",
                ]
            else:
                arm_joint_names = [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ]
        self._arm_joint_ids = []
        self._arm_q_ids = []
        for name in arm_joint_names:
            jid = self._model.getJointId(name)
            self._arm_joint_ids.append(jid)
            self._arm_q_ids.append(self._model.joints[jid].idx_q)

        self._arm_q_ids = np.array(self._arm_q_ids, dtype=int)

        # Joint limits for arm joints
        self._q_min = self._model.lowerPositionLimit[self._arm_q_ids]
        self._q_max = self._model.upperPositionLimit[self._arm_q_ids]

        self._damping = damping
        self._max_delta = max_delta

    @property
    def n_arm_joints(self) -> int:
        return len(self._arm_q_ids)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q_min.copy(), self._q_max.copy()

    def forward_kinematics(self, q_arm: np.ndarray) -> np.ndarray:
        """Compute EEF position for given arm joint angles.

        Args:
            q_arm: Arm joint positions (n_arm_joints,)

        Returns:
            EEF position in world frame (3,)
        """
        q_full = np.zeros(self._model.nq)
        q_full[self._arm_q_ids] = q_arm
        pin.forwardKinematics(self._model, self._data, q_full)
        pin.updateFramePlacements(self._model, self._data)
        return self._data.oMf[self._eef_id].translation.copy()

    def get_jacobian(self, q_arm: np.ndarray) -> np.ndarray:
        """Compute the 3×N position Jacobian of the EEF.

        Args:
            q_arm: Arm joint positions (n_arm_joints,)

        Returns:
            Position Jacobian (3, n_arm_joints)
        """
        q_full = np.zeros(self._model.nq)
        q_full[self._arm_q_ids] = q_arm
        pin.forwardKinematics(self._model, self._data, q_full)
        pin.computeJointJacobians(self._model, self._data, q_full)

        # Get full Jacobian, extract arm columns
        J6 = pin.getFrameJacobian(
            self._model, self._data, self._eef_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_pos = J6[:3, self._arm_q_ids]  # (3, n_arm_joints)
        return J_pos

    def solve(
        self,
        q_arm: np.ndarray,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        """Run one IK step toward target position.

        Args:
            q_arm: Current arm joint positions (n_arm_joints,)
            target_pos: Target EEF position in world frame (3,)

        Returns:
            New arm joint positions (n_arm_joints,)
        """
        current_pos = self.forward_kinematics(q_arm)
        error = target_pos - current_pos

        J = self.get_jacobian(q_arm)

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^{-1} error
        JJT = J @ J.T + self._damping**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)

        # Clip step size
        dq = np.clip(dq, -self._max_delta, self._max_delta)

        # Update and clamp to joint limits
        q_new = q_arm + dq
        q_new = np.clip(q_new, self._q_min, self._q_max)

        return q_new

    def solve_to_convergence(
        self,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        max_iter: int = 200,
        tol: float = 1e-3,
    ) -> tuple[np.ndarray, bool, float]:
        """Iterate IK until convergence or max iterations.

        Args:
            q_init: Initial arm joint positions (n_arm_joints,)
            target_pos: Target EEF position (3,)
            max_iter: Maximum iterations
            tol: Position tolerance in meters

        Returns:
            (q_final, converged, final_error)
        """
        q = q_init.copy()
        for _ in range(max_iter):
            q = self.solve(q, target_pos)
            pos = self.forward_kinematics(q)
            err = np.linalg.norm(pos - target_pos)
            if err < tol:
                return q, True, err
        return q, False, err
