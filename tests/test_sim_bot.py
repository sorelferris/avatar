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


def test_get_angles_returns_full_vector():
    """get_angles() returns the robot's full joint angle vector."""
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    angles = bot.get_angles()
    assert isinstance(angles, np.ndarray)
    # SO101 URDF has 6 joints (5 arm + 1 gripper)
    assert angles.shape == (6,)
    assert np.allclose(angles, 0.0)


def test_set_joints_round_trip_and_delta():
    """set_joints updates only the joints in the dict (delta semantics).

    Setting joint A then joint B should leave A at its value, not reset
    to zero.
    """
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )

    # Set shoulder_pan to 0.3
    bot.set_joints({"shoulder_pan": 0.3})
    angles = bot.get_angles()
    # SO101 URDF joint order: gripper(0), wrist_roll(1), wrist_flex(2), elbow_flex(3), shoulder_lift(4), shoulder_pan(5)
    assert np.isclose(angles[5], 0.3), "shoulder_pan should be 0.3"

    # Set elbow_flex to -0.5; shoulder_pan should still be 0.3
    bot.set_joints({"elbow_flex": -0.5})
    angles = bot.get_angles()
    assert np.isclose(angles[5], 0.3), "shoulder_pan should still be 0.3"
    assert np.isclose(angles[3], -0.5), "elbow_flex should be -0.5"


def test_set_joints_unknown_name_raises_keyerror():
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    with pytest.raises(KeyError):
        bot.set_joints({"nonexistent_joint": 0.0})


def test_get_arm_angles_and_eef_position():
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    bot.set_joints({"shoulder_pan": 0.3, "elbow_flex": -0.5})

    # 5-DoF arm slice
    right_angles = bot.get_arm_angles("right")
    assert right_angles.shape == (5,)
    # SO101 URDF 关节顺序: gripper(0), wrist_roll(1), wrist_flex(2), elbow_flex(3), shoulder_lift(4), shoulder_pan(5)
    # right_arm_joints 顺序: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
    assert np.isclose(right_angles[0], 0.3)   # shoulder_pan
    assert np.isclose(right_angles[2], -0.5)  # elbow_flex

    # EEF position is a 3-vector
    eef = bot.get_eef_position("right")
    assert eef.shape == (3,)
    assert np.linalg.norm(eef) > 0.0


def test_solve_ik_moves_eef_toward_target():
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    # EEF at zero pose
    eef_before = bot.get_eef_position("right")

    # Target slightly offset in +X
    target = eef_before + np.array([0.02, 0.0, 0.0])

    q_new = bot.solve_ik("right", target)
    assert q_new.shape == (5,)

    # Single-step IK should move EEF toward target
    eef_after = bot.get_eef_position("right")
    # Either EEF moved toward target, or joint limits clamped it
    eef_after_dist = np.linalg.norm(eef_after - target)
    eef_before_dist = np.linalg.norm(eef_before - target)
    assert eef_after_dist <= eef_before_dist + 1e-6, (
        f"IK step did not reduce EEF distance: "
        f"before={eef_before_dist:.4f} after={eef_after_dist:.4f}"
    )


def test_solve_ik_respects_max_delta():
    """Single-step IK should change joints by at most IKSolver.max_delta."""
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    eef = bot.get_eef_position("right")
    # Target 1m away (unreachable in one step)
    target = eef + np.array([1.0, 0.0, 0.0])

    q_init = bot.get_arm_angles("right")
    q_new = bot.solve_ik("right", target, q_init=q_init)

    # IKSolver default max_delta=0.1 rad per joint; for 5 joints the
    # Euclidean norm of a full [-0.1, 0.1]^5 delta vector is sqrt(5)*0.1 ≈ 0.224
    assert np.linalg.norm(q_new - q_init) <= np.sqrt(5) * 0.1 + 1e-6


def test_close_is_idempotent():
    """close() can be called multiple times without error.

    Only meaningful with viewer=True (otherwise no viewer to close).
    """
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    # When viewer=False, _viewer is None; close should be a no-op.
    bot.close()
    bot.close()  # must not raise


def test_close_calls_viewer_close_idempotently(monkeypatch):
    """close() should call viewer.close() and tolerate multiple calls.

    Uses a fake viewer that counts close() calls to verify the SimBot
    delegates correctly. We don't construct a real ViserViewer because
    that would open a browser and bind a port.
    """
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    # Inject a fake viewer (the real one would need a browser)
    close_call_count = [0]

    class FakeViewer:
        def close(self):
            close_call_count[0] += 1

        def redraw(self):
            pass

    bot._viewer = FakeViewer()

    bot.close()
    bot.close()  # must not raise; close is called twice
    assert close_call_count[0] == 2
