import numpy as np
from src.sim_env import SimEnvironment
from src.ik_solver import IKSolver


def test_ik_converges_to_reachable_target():
    env = SimEnvironment("assets/g1_23dof_fixed.xml")
    solver = IKSolver(env, damping=0.1, max_delta=0.1)
    env.forward()
    current_pos = env.get_eef_position("left")
    target = current_pos + np.array([0.05, 0.0, 0.0])
    for _ in range(200):
        q = solver.solve("left", target)
    env.forward()
    final_pos = env.get_eef_position("left")
    error = np.linalg.norm(final_pos - target)
    assert error < 0.05, f"IK did not converge, error: {error:.4f}"


def test_ik_respects_joint_limits():
    env = SimEnvironment("assets/g1_23dof_fixed.xml")
    solver = IKSolver(env, damping=0.1, max_delta=0.1)
    target = np.array([1.0, 0.0, 1.0])
    for _ in range(500):
        q = solver.solve("left", target)
    joints = env.get_joint_positions("left")
    ranges = env.left_joint_ranges
    assert np.all(joints >= ranges[:, 0] - 0.01)
    assert np.all(joints <= ranges[:, 1] + 0.01)
