"""MuJoCo simulation environment for SO101 arm."""

import mujoco
import numpy as np


class SimEnvironment:
    """SO101 arm simulation environment using MuJoCo.

    Wraps MuJoCo model/data with joint lookup and control helpers.
    """

    ARM_JOINTS = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]

    def __init__(self, xml_path: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002  # 500Hz physics

        # Find arm joint indices
        if "so102" in xml_path:
            self.arm_joint_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_yaw",
                "wrist_roll",
            ]
        else:
            self.arm_joint_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ]
        self.arm_joints = self._find_joints(self.arm_joint_names)
        self.arm_actuators = self._find_actuators(self.arm_joint_names)

        # Gripper joint
        self.gripper_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
        )
        self.gripper_actuator = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"
        )
        self.gripper_qposadr = self.model.jnt_qposadr[self.gripper_joint]
        self.gripper_range = self.model.jnt_range[self.gripper_joint]

        # EEF body (gripper is the last body before EEF frame)
        self.eef_body = self._find_body("gripper")

        # qpos/dof addresses
        self.arm_qposadr = self.model.jnt_qposadr[self.arm_joints]
        self.arm_dofadr = self.model.jnt_dofadr[self.arm_joints]

        # Joint ranges
        self.joint_ranges = self.model.jnt_range[self.arm_joints]

        mujoco.mj_resetData(self.model, self.data)

    def _find_joints(self, names: list[str]) -> np.ndarray:
        return np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        )

    def _find_actuators(self, names: list[str]) -> np.ndarray:
        return np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names]
        )

    def _find_body(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def get_eef_position(self) -> np.ndarray:
        return self.data.xpos[self.eef_body].copy()

    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.arm_qposadr].copy()

    def set_joint_positions(self, q: np.ndarray) -> None:
        self.data.qpos[self.arm_qposadr] = q
        mujoco.mj_forward(self.model, self.data)

    def set_control(self, q: np.ndarray) -> None:
        self.data.ctrl[self.arm_actuators] = q

    def set_gripper(self, position: float) -> None:
        """Set gripper joint position.

        Args:
            position: Joint angle in radians (0 = closed, ~1.4 = open)
        """
        self.data.ctrl[self.gripper_actuator] = np.clip(
            position, self.gripper_range[0], self.gripper_range[1]
        )

    def reset(self) -> None:
        """Reset arm to zero position and open gripper."""
        self.data.qpos[self.arm_qposadr] = 0.0
        self.data.qvel[self.arm_dofadr] = 0.0
        self.data.ctrl[self.arm_actuators] = 0.0
        # Open gripper slightly
        open_pos = self.gripper_range[1] * 0.8
        self.data.qpos[self.gripper_qposadr] = open_pos
        self.data.ctrl[self.gripper_actuator] = open_pos
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)
