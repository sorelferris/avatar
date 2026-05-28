# tests/test_coordinate_processor.py
import numpy as np
from src.coordinate_processor import remap_camera_to_robot


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
