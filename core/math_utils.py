"""Mathematical utilities for robot arm simulation."""

import numpy as np
from typing import Tuple, List, Union
from scipy.spatial.transform import Rotation


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in [-pi, pi]
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to rotation matrix.
    
    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
        
    Returns:
        3x3 rotation matrix
    """
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    return r.as_matrix()


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to Euler angles.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    r = Rotation.from_matrix(R)
    return tuple(r.as_euler('xyz'))


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion as [x, y, z, w]
        
    Returns:
        3x3 rotation matrix
    """
    r = Rotation.from_quat(q)
    return r.as_matrix()


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion as [x, y, z, w]
    """
    r = Rotation.from_matrix(R)
    return r.as_quat()


def homogeneous_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous transformation matrix.
    
    Args:
        R: 3x3 rotation matrix
        t: 3D translation vector
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Create Denavit-Hartenberg transformation matrix.
    
    Args:
        a: Link length
        alpha: Link twist
        d: Link offset
        theta: Joint angle
        
    Returns:
        4x4 DH transformation matrix
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to specified range.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def interpolate_angles(start_angles: np.ndarray, end_angles: np.ndarray, 
                      t: float) -> np.ndarray:
    """Interpolate between two sets of joint angles.
    
    Args:
        start_angles: Starting joint angles
        end_angles: Ending joint angles
        t: Interpolation parameter [0, 1]
        
    Returns:
        Interpolated joint angles
    """
    # Handle angle wrapping for smooth interpolation
    diff = end_angles - start_angles
    diff = np.array([normalize_angle(d) for d in diff])
    return start_angles + t * diff


def compute_jacobian(joint_positions: List[np.ndarray], 
                    joint_axes: List[np.ndarray],
                    end_effector_pos: np.ndarray) -> np.ndarray:
    """Compute Jacobian matrix for the robot arm.
    
    Args:
        joint_positions: List of joint positions in world coordinates
        joint_axes: List of joint rotation axes
        end_effector_pos: End effector position
        
    Returns:
        6xN Jacobian matrix (3 for linear, 3 for angular velocity)
    """
    n_joints = len(joint_positions)
    jacobian = np.zeros((6, n_joints))
    
    for i in range(n_joints):
        # Linear velocity component
        r = end_effector_pos - joint_positions[i]
        jacobian[:3, i] = np.cross(joint_axes[i], r)
        
        # Angular velocity component
        jacobian[3:, i] = joint_axes[i]
    
    return jacobian


def distance_point_to_line(point: np.ndarray, line_start: np.ndarray, 
                          line_end: np.ndarray) -> float:
    """Calculate distance from point to line segment.
    
    Args:
        point: 3D point
        line_start: Start of line segment
        line_end: End of line segment
        
    Returns:
        Distance from point to line segment
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return np.linalg.norm(point_vec)
    
    line_unitvec = line_vec / line_len
    proj_length = np.dot(point_vec, line_unitvec)
    
    if proj_length < 0:
        return np.linalg.norm(point_vec)
    elif proj_length > line_len:
        return np.linalg.norm(point - line_end)
    else:
        proj_point = line_start + proj_length * line_unitvec
        return np.linalg.norm(point - proj_point)
