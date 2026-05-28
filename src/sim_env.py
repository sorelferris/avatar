import mujoco
import numpy as np


class SimEnvironment:
    def __init__(self, xml_path: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002  # 500Hz physics
        # Find arm joint indices
        self.left_arm_joints = self._find_joints([
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
        ])
        self.right_arm_joints = self._find_joints([
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ])
        # Find actuator indices
        self.left_arm_actuators = self._find_actuators([
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
        ])
        self.right_arm_actuators = self._find_actuators([
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ])
        # Find EEF body IDs
        self.left_eef_body = self._find_body("left_wrist_roll_rubber_hand")
        self.right_eef_body = self._find_body("right_wrist_roll_rubber_hand")
        # Get joint ranges
        self.left_joint_ranges = self.model.jnt_range[self.left_arm_joints]
        self.right_joint_ranges = self.model.jnt_range[self.right_arm_joints]
        mujoco.mj_resetData(self.model, self.data)

    def _find_joints(self, names: list[str]) -> np.ndarray:
        indices = []
        for name in names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            indices.append(jnt_id)
        return np.array(indices)

    def _find_actuators(self, names: list[str]) -> np.ndarray:
        indices = []
        for name in names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            indices.append(act_id)
        return np.array(indices)

    def _find_body(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def get_eef_position(self, side: str) -> np.ndarray:
        body_id = self.left_eef_body if side == "left" else self.right_eef_body
        return self.data.xpos[body_id].copy()

    def get_joint_positions(self, side: str) -> np.ndarray:
        joints = self.left_arm_joints if side == "left" else self.right_arm_joints
        return self.data.qpos[joints].copy()

    def get_joint_velocities(self, side: str) -> np.ndarray:
        joints = self.left_arm_joints if side == "left" else self.right_arm_joints
        return self.data.qvel[joints].copy()

    def set_control(self, side: str, torques: np.ndarray) -> None:
        actuators = self.left_arm_actuators if side == "left" else self.right_arm_actuators
        self.data.ctrl[actuators] = torques

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)
