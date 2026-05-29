import numpy as np
from src.ik_solver import IKSolver
from src.sim_env import SimEnvironment

URDF_PATH = "assets/SO101/so102.urdf"
XML_PATH = "assets/SO101/so102.xml"


def test_so102_joint_count():
    # 1. 验证 IK 求解器检测到 6 个控制关节
    solver = IKSolver(URDF_PATH)
    assert solver.n_arm_joints == 6

    # 2. 验证 MuJoCo 模型拥有 6 个手臂关节
    env = SimEnvironment(XML_PATH)
    assert len(env.arm_joint_names) == 6


def test_so102_fk_at_zero():
    # 验证全零位下的 Forward Kinematics 正常运行且产生非零 EEF
    solver = IKSolver(URDF_PATH)
    pos = solver.forward_kinematics(np.zeros(6))
    assert pos.shape == (3,)
    assert np.linalg.norm(pos) > 0.1


def test_so102_ik_convergence():
    # 验证 6DOF 的 IK 逆运动学求解，随机给一个可达点，确保 DLS IK 完美收敛且误差极小 (< 1mm)
    solver = IKSolver(URDF_PATH, damping=0.01, max_delta=0.1)
    q_min, q_max = solver.joint_limits

    # 使用随机种子确保确定性
    rng = np.random.default_rng(42)
    # 在限位内生成一个目标姿态（不要离零位太远以防陷入局部极小值）
    q_target = rng.uniform(-0.5, 0.5, size=6)
    q_target = np.clip(q_target, q_min, q_max)

    target_pos = solver.forward_kinematics(q_target)

    # 从零位开始求解
    q0 = np.zeros(6)
    q_final, converged, error = solver.solve_to_convergence(
        q0, target_pos, max_iter=200, tol=1e-3
    )

    assert converged, f"IK did not converge, error: {error:.4f}"
    assert error < 0.001  # 误差极小 (< 1mm)


def test_so102_joint_limits():
    # 验证 IK 求解时所有 6 个关节始终遵守其在 URDF 中声明的各自限位
    solver = IKSolver(URDF_PATH, damping=0.1, max_delta=0.1)
    q0 = np.zeros(6)
    target = np.array([1.0, 1.0, 1.0])  # 不可达的远点

    q = q0.copy()
    for _ in range(500):
        q = solver.solve(q, target)

    q_min, q_max = solver.joint_limits
    assert np.all(q >= q_min - 1e-5)
    assert np.all(q <= q_max + 1e-5)
