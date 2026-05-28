# tests/test_coordinate_processor.py
import numpy as np
from src.coordinate_processor import remap_camera_to_robot, LinearScaler


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
