import mujoco
import numpy as np
from src.sim_env import SimEnvironment


class IKSolver:
    def __init__(
        self,
        env: SimEnvironment,
        damping: float = 0.1,
        max_delta: float = 0.1,
        kp: float = 50.0,
        kd: float = 5.0,
    ) -> None:
        self._env = env
        self._damping = damping
        self._max_delta = max_delta
        self._kp = kp
        self._kd = kd
        self._jac_pos = np.zeros((3, env.model.nv))
        self._jac_rot = np.zeros((3, env.model.nv))

    def solve(self, side: str, target_pos: np.ndarray) -> np.ndarray:
        env = self._env
        if side == "left":
            body_id = env.left_eef_body
            dofadr = env.left_arm_dofadr
            qposadr = env.left_arm_qposadr
            joint_ranges = env.left_joint_ranges
        else:
            body_id = env.right_eef_body
            dofadr = env.right_arm_dofadr
            qposadr = env.right_arm_qposadr
            joint_ranges = env.right_joint_ranges

        current_pos = env.data.xpos[body_id].copy()
        mujoco.mj_jac(
            env.model, env.data, self._jac_pos, self._jac_rot, current_pos, body_id
        )
        J = self._jac_pos[:, dofadr]
        error = target_pos - current_pos

        JJT = J @ J.T + self._damping**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)
        dq = np.clip(dq, -self._max_delta, self._max_delta)

        q_current = env.data.qpos[qposadr].copy()
        q_new = q_current + dq
        q_new = np.clip(q_new, joint_ranges[:, 0], joint_ranges[:, 1])

        env.data.qpos[qposadr] = q_new
        env.forward()
        return q_new

    def compute_pd_torque(self, side: str, q_target: np.ndarray) -> np.ndarray:
        env = self._env
        q_current = env.get_joint_positions(side)
        qdot = env.get_joint_velocities(side)
        torque = self._kp * (q_target - q_current) - self._kd * qdot
        return torque
