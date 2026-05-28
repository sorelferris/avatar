# tests/test_coordinate_processor.py
import numpy as np
from src.coordinate_processor import (
    remap_camera_to_robot,
    LinearScaler,
    EMAFilter,
    clip_to_workspace,
    CoordinateProcessor,
)


def test_remap_camera_to_robot_identity():
    """Camera: X right, Y down, Z forward -> Robot: X forward, Y left, Z up"""
    cam = np.array([0.0, 0.0, 1.0])
    robot = remap_camera_to_robot(cam)
    assert np.allclose(robot, [1.0, 0.0, 0.0])


def test_remap_camera_to_robot_x():
    cam = np.array([1.0, 0.0, 0.0])
    robot = remap_camera_to_robot(cam)
    assert np.allclose(robot, [0.0, -1.0, 0.0])


def test_remap_camera_to_robot_y():
    cam = np.array([0.0, 1.0, 0.0])
    robot = remap_camera_to_robot(cam)
    assert np.allclose(robot, [0.0, 0.0, -1.0])


def test_scaler_identity_at_anchor():
    scaler = LinearScaler(hand_range=0.3, robot_range=0.4)
    anchor = np.array([0.5, 0.5, 0.5])
    scaler.set_anchor(anchor)
    result = scaler.scale(anchor)
    assert np.allclose(result, [0.0, 0.0, 0.0])


def test_scaler_proportional():
    scaler = LinearScaler(hand_range=0.3, robot_range=0.4)
    anchor = np.array([0.5, 0.5, 0.5])
    scaler.set_anchor(anchor)
    moved = np.array([0.65, 0.5, 0.5])
    result = scaler.scale(moved)
    assert np.allclose(result, [0.2, 0.0, 0.0], atol=1e-6)


def test_ema_first_value():
    ema = EMAFilter(alpha=0.3)
    result = ema.update(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(result, [1.0, 2.0, 3.0])


def test_ema_smoothing():
    ema = EMAFilter(alpha=0.3)
    ema.update(np.array([0.0, 0.0, 0.0]))
    result = ema.update(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(result, [0.3, 0.3, 0.3])


def test_ema_convergence():
    ema = EMAFilter(alpha=0.3)
    val = np.array([1.0, 1.0, 1.0])
    for _ in range(100):
        result = ema.update(val)
    assert np.allclose(result, [1.0, 1.0, 1.0], atol=1e-3)


def test_clip_inside_workspace():
    target = np.array([0.1, 0.1, 0.1])
    result = clip_to_workspace(target, max_radius=0.4)
    assert np.allclose(result, target)


def test_clip_outside_workspace():
    target = np.array([0.5, 0.5, 0.5])
    result = clip_to_workspace(target, max_radius=0.4)
    assert np.linalg.norm(result) <= 0.4 + 1e-6
    assert np.allclose(
        result / np.linalg.norm(result), target / np.linalg.norm(target), atol=1e-6
    )


def test_processor_full_pipeline():
    proc = CoordinateProcessor(
        hand_range=0.3,
        robot_range=0.4,
        ema_alpha=0.3,
        max_radius=0.36,
    )
    anchor = np.array([0.5, 0.5, 0.5])
    proc.set_anchor(anchor)
    result, detected = proc.process(anchor, detected=True)
    assert detected is True
    assert np.allclose(result, [0.0, 0.0, 0.0], atol=1e-6)


def test_processor_hand_lost():
    proc = CoordinateProcessor(
        hand_range=0.3,
        robot_range=0.4,
        ema_alpha=0.3,
        max_radius=0.36,
    )
    proc.set_anchor(np.array([0.5, 0.5, 0.5]))
    result1, _ = proc.process(np.array([0.6, 0.5, 0.5]), detected=True)
    result2, detected = proc.process(None, detected=False)
    assert detected is False
    assert np.allclose(result2, result1)  # frozen
