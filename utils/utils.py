"""
Utility functions for the RAIT (Robot-AI Toolkit) system.
"""

import os
import sys
import numpy as np
import pyrealsense2 as rs

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import load_config

config = load_config("config/config.yaml")

camera_transformations = config['Camera']['D435I']['India']['Transformations']
camera_intrinsics = config['Camera']['D435I']['India']['Intrinsics']['Color_Intrinsics']
#----------------------------------------------------------------#
def get_valid_depth(depth_array, x, y):
    """
    Find the first non-zero depth value within a 10-pixel radius around the given point.

    Searches in increasing radius up to 10 pixels until a valid (non-zero) depth value
    is found. This helps handle cases where the target pixel has invalid depth data.

    Args:
        depth_array (numpy.ndarray): 2D array containing depth values
        x (int): Target x-coordinate in the depth array
        y (int): Target y-coordinate in the depth array

    Returns:
        tuple: (depth, x, y) where:
            - depth (float): Valid depth value or 0 if none found
            - x (int): X-coordinate of valid depth point
            - y (int): Y-coordinate of valid depth point
    """
    height, width = depth_array.shape
    if depth_array[y, x] > 0:
        return depth_array[y, x], x, y

    max_radius = 10
    for radius in range(1, max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                new_x = x + dx
                new_y = y + dy
                if 0 <= new_x < width and 0 <= new_y < height:
                    depth = depth_array[new_y, new_x]
                    if depth > 0:
                        return depth, new_x, new_y

    return 0, x, y

def deproject_pixel_to_point(depth_array, pixel_coords, intrinsics):
    """
    Deproject pixel coordinates and depth to 3D point using RealSense intrinsics.

    Converts 2D pixel coordinates and corresponding depth values into 3D points
    using the RealSense camera's intrinsic parameters. Handles invalid depth values
    by searching nearby pixels.

    Args:
        depth_array (numpy.ndarray): 2D array of depth values
        pixel_coords (tuple): (x, y) pixel coordinates to deproject
        intrinsics (dict): RealSense camera intrinsic parameters

    Returns:
        numpy.ndarray: 3D point coordinates [x, y, z] in camera space.
        Returns [0, 0, 0] if no valid depth is found or coordinates are out of bounds.
    """
    x, y = int(pixel_coords[0]), int(pixel_coords[1])
    if x < 0 or x >= depth_array.shape[1] or y < 0 or y >= depth_array.shape[0]:
        return np.array([0, 0, 0])
    depth, valid_x, valid_y = get_valid_depth(depth_array, x, y)
    if depth == 0:
        print(f"Warning: No valid depth found near pixel ({x}, {y})")
        return np.array([0, 0, 0])
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [valid_x, valid_y], depth)
    return np.array(point_3d)

def transform_coordinates(x, y, z):
    """
    Transforms coordinates from camera space to collaborative robot base frame.

    Applies a series of transformations using calibration matrices to convert
    coordinates from the camera's reference frame to the robot's base frame.

    Args:
        x (float): X-coordinate in camera space (millimeters)
        y (float): Y-coordinate in camera space (millimeters)
        z (float): Z-coordinate in camera space (millimeters)

    Returns:
        tuple: (transformed_x, transformed_y, transformed_z) in robot base frame (millimeters)
    """
    calib_matrix_x = np.array(config['Transformations']['X'])
    calib_matrix_y = np.array(config['Transformations']['Y'])
    B = np.eye(4)
    B[:3, 3] = [x / 1000, y / 1000, z / 1000]
    A = calib_matrix_y @ B @ np.linalg.inv(calib_matrix_x)
    transformed_x, transformed_y, transformed_z = A[:3, 3] * 1000
    return float(transformed_x), float(transformed_y), float(transformed_z)