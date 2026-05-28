import numpy as np
from src.ik_solver import IKSolver

URDF = "assets/SO101/so101_new_calib.urdf"


def test_fk_at_zero():
    solver = IKSolver(URDF)
    pos = solver.forward_kinematics(np.zeros(5))
    assert pos.shape == (3,)
    assert np.linalg.norm(pos) > 0.1


def test_ik_converges_to_reachable_target():
    solver = IKSolver(URDF, damping=0.1, max_delta=0.1)
    q0 = np.zeros(5)
    current_pos = solver.forward_kinematics(q0)
    target = current_pos + np.array([0.02, 0.0, 0.0])

    q_final, converged, error = solver.solve_to_convergence(q0, target, max_iter=200)
    assert converged, f"IK did not converge, error: {error:.4f}"
    assert error < 0.01


def test_ik_respects_joint_limits():
    solver = IKSolver(URDF, damping=0.1, max_delta=0.1)
    q0 = np.zeros(5)
    target = np.array([1.0, 0.0, 1.0])

    q = q0.copy()
    for _ in range(500):
        q = solver.solve(q, target)

    q_min, q_max = solver.joint_limits
    assert np.all(q >= q_min - 0.01)
    assert np.all(q <= q_max + 0.01)


def test_solve_step_is_deterministic():
    solver = IKSolver(URDF)
    q = np.zeros(5)
    target = solver.forward_kinematics(q) + np.array([0.01, 0.0, 0.0])
    q1 = solver.solve(q, target)
    q2 = solver.solve(q, target)
    assert np.allclose(q1, q2)
