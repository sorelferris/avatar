# src/coordinate_processor.py
import numpy as np


def remap_camera_to_robot(cam_coords: np.ndarray) -> np.ndarray:
    """Remap camera frame (X right, Y down, Z forward) to robot frame (X forward, Y left, Z up)."""
    return np.array([cam_coords[2], -cam_coords[0], -cam_coords[1]])
