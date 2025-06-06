"""Forward and inverse kinematics for the robot arm."""

import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING
from scipy.optimize import minimize
import warnings

from core.math_utils import (
    dh_transform, homogeneous_transform, euler_to_rotation_matrix,
    rotation_matrix_to_euler, compute_jacobian, clamp
)

if TYPE_CHECKING:
    from .robot_arm import RobotArm


class ForwardKinematics:
    """Forward kinematics solver for the robot arm."""
    
    def __init__(self, robot_arm: 'RobotArm', joint_names: List[str]):
        """Initialize forward kinematics solver.
        
        Args:
            robot_arm: Reference to the robot arm
            joint_names: List of joint names to include in kinematics
        """
        self.robot_arm = robot_arm
        self.joint_names = joint_names
        
        # DH parameters for the main arm joints
        # [a, alpha, d, theta_offset]
        self.dh_params = self._setup_dh_parameters()
        
        # Cached transforms
        self._joint_transforms = []
        self._link_transforms = []
        self._end_effector_transform = np.eye(4)
        
    def _setup_dh_parameters(self) -> List[List[float]]:
        """Setup Denavit-Hartenberg parameters for the arm.
        
        Returns:
            List of DH parameters [a, alpha, d, theta_offset] for each joint
        """
        # Simplified DH parameters for anthropomorphic arm
        # These would need to be adjusted based on exact robot geometry
        
        link_lengths = self.robot_arm.links
        upper_arm_length = link_lengths['upper_arm'].length if 'upper_arm' in link_lengths else 0.3
        forearm_length = link_lengths['forearm'].length if 'forearm' in link_lengths else 0.25
        hand_length = link_lengths['hand'].length if 'hand' in link_lengths else 0.15
        
        dh_params = [
            # [a, alpha, d, theta_offset]
            [0, 0, 0, 0],                    # shoulder_pitch (base)
            [0, np.pi/2, 0, 0],             # shoulder_yaw
            [0, -np.pi/2, 0, 0],            # shoulder_roll
            [upper_arm_length, 0, 0, 0],     # elbow_flexion
            [forearm_length, 0, 0, 0],       # wrist_pitch
            [0, np.pi/2, hand_length, 0],    # wrist_yaw (end effector)
        ]
        
        return dh_params
    
    def compute_joint_transforms(self, joint_angles: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Compute transformation matrices for each joint.
        
        Args:
            joint_angles: Joint angles (uses current if None)
            
        Returns:
            List of 4x4 transformation matrices
        """
        if joint_angles is None:
            joint_angles = self.robot_arm.get_joint_positions(self.joint_names)
        
        transforms = []
        cumulative_transform = np.eye(4)
        
        for i, (joint_name, dh_param) in enumerate(zip(self.joint_names, self.dh_params)):
            a, alpha, d, theta_offset = dh_param
            theta = joint_angles[i] + theta_offset
            
            # Compute DH transformation
            T = dh_transform(a, alpha, d, theta)
            
            # Accumulate transformation
            cumulative_transform = cumulative_transform @ T
            transforms.append(cumulative_transform.copy())
        
        self._joint_transforms = transforms
        return transforms
    
    def compute_end_effector_pose(self, joint_angles: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end effector position and orientation.
        
        Args:
            joint_angles: Joint angles (uses current if None)
            
        Returns:
            Tuple of (position, rotation_matrix)
        """
        transforms = self.compute_joint_transforms(joint_angles)
        
        if len(transforms) > 0:
            self._end_effector_transform = transforms[-1]
            position = self._end_effector_transform[:3, 3]
            rotation = self._end_effector_transform[:3, :3]
            return position, rotation
        else:
            return np.zeros(3), np.eye(3)
    
    def compute_jacobian(self, joint_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute Jacobian matrix for the current configuration.
        
        Args:
            joint_angles: Joint angles (uses current if None)
            
        Returns:
            6xN Jacobian matrix
        """
        if joint_angles is None:
            joint_angles = self.robot_arm.get_joint_positions(self.joint_names)
        
        transforms = self.compute_joint_transforms(joint_angles)
        end_effector_pos, _ = self.compute_end_effector_pose(joint_angles)
        
        # Extract joint positions and axes
        joint_positions = []
        joint_axes = []
        
        for i, transform in enumerate(transforms):
            joint_positions.append(transform[:3, 3])
            # Z-axis of the joint frame is the rotation axis
            joint_axes.append(transform[:3, 2])
        
        return compute_jacobian(joint_positions, joint_axes, end_effector_pos)
    
    def update(self) -> None:
        """Update forward kinematics with current joint positions."""
        self.compute_end_effector_pose()
    
    def get_link_positions(self, joint_angles: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Get positions of all links.
        
        Args:
            joint_angles: Joint angles (uses current if None)
            
        Returns:
            List of link positions
        """
        transforms = self.compute_joint_transforms(joint_angles)
        return [T[:3, 3] for T in transforms]


class InverseKinematics:
    """Inverse kinematics solver for the robot arm."""
    
    def __init__(self, robot_arm: 'RobotArm', joint_names: List[str]):
        """Initialize inverse kinematics solver.
        
        Args:
            robot_arm: Reference to the robot arm
            joint_names: List of joint names to include in kinematics
        """
        self.robot_arm = robot_arm
        self.joint_names = joint_names
        self.fk_solver = ForwardKinematics(robot_arm, joint_names)
        
        # Solver parameters
        self.max_iterations = 100
        self.tolerance = 1e-6
        self.step_size = 0.1
        
    def solve(self, target_position: np.ndarray, 
             target_orientation: Optional[np.ndarray] = None,
             initial_guess: Optional[np.ndarray] = None,
             method: str = "jacobian") -> Optional[np.ndarray]:
        """Solve inverse kinematics for target pose.
        
        Args:
            target_position: Target end effector position
            target_orientation: Target end effector orientation (rotation matrix)
            initial_guess: Initial joint angle guess
            method: Solution method ("jacobian", "optimization")
            
        Returns:
            Joint angles that achieve target pose, or None if no solution
        """
        if method == "jacobian":
            return self._solve_jacobian(target_position, target_orientation, initial_guess)
        elif method == "optimization":
            return self._solve_optimization(target_position, target_orientation, initial_guess)
        else:
            raise ValueError(f"Unknown IK method: {method}")
    
    def _solve_jacobian(self, target_position: np.ndarray,
                       target_orientation: Optional[np.ndarray] = None,
                       initial_guess: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Solve IK using Jacobian-based method (Newton-Raphson).
        
        Args:
            target_position: Target position
            target_orientation: Target orientation
            initial_guess: Initial joint angles
            
        Returns:
            Solution joint angles or None
        """
        # Initialize joint angles
        if initial_guess is not None:
            q = initial_guess.copy()
        else:
            q = self.robot_arm.get_joint_positions(self.joint_names)
        
        for iteration in range(self.max_iterations):
            # Compute current end effector pose
            current_pos, current_rot = self.fk_solver.compute_end_effector_pose(q)
            
            # Compute position error
            pos_error = target_position - current_pos
            
            # Compute orientation error (if target orientation provided)
            if target_orientation is not None:
                # Convert rotation matrices to axis-angle representation for error
                rot_error_matrix = target_orientation @ current_rot.T
                # Extract axis-angle (simplified)
                rot_error = rotation_matrix_to_euler(rot_error_matrix)
                error = np.concatenate([pos_error, rot_error])
            else:
                error = pos_error
            
            # Check convergence
            if np.linalg.norm(error) < self.tolerance:
                return q
            
            # Compute Jacobian
            J = self.fk_solver.compute_jacobian(q)
            
            # Use only position part if no orientation target
            if target_orientation is None:
                J = J[:3, :]
            
            # Compute pseudo-inverse
            try:
                J_pinv = np.linalg.pinv(J)
            except np.linalg.LinAlgError:
                warnings.warn("Jacobian is singular, IK may not converge")
                return None
            
            # Update joint angles
            dq = J_pinv @ error
            q = q + self.step_size * dq
            
            # Apply joint limits
            for i, joint_name in enumerate(self.joint_names):
                joint = self.robot_arm.joints[joint_name]
                q[i] = clamp(q[i], joint.limits[0], joint.limits[1])
        
        # Check final error
        final_pos, final_rot = self.fk_solver.compute_end_effector_pose(q)
        final_error = np.linalg.norm(target_position - final_pos)
        
        if final_error < self.tolerance * 10:  # Relaxed tolerance
            return q
        else:
            return None
    
    def _solve_optimization(self, target_position: np.ndarray,
                           target_orientation: Optional[np.ndarray] = None,
                           initial_guess: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Solve IK using optimization-based method.
        
        Args:
            target_position: Target position
            target_orientation: Target orientation
            initial_guess: Initial joint angles
            
        Returns:
            Solution joint angles or None
        """
        # Initialize
        if initial_guess is not None:
            x0 = initial_guess.copy()
        else:
            x0 = self.robot_arm.get_joint_positions(self.joint_names)
        
        # Define objective function
        def objective(q):
            current_pos, current_rot = self.fk_solver.compute_end_effector_pose(q)
            pos_error = np.linalg.norm(target_position - current_pos)
            
            if target_orientation is not None:
                rot_error_matrix = target_orientation @ current_rot.T
                rot_error = np.linalg.norm(rotation_matrix_to_euler(rot_error_matrix))
                return pos_error + rot_error
            else:
                return pos_error
        
        # Define constraints (joint limits)
        bounds = []
        for joint_name in self.joint_names:
            joint = self.robot_arm.joints[joint_name]
            bounds.append((joint.limits[0], joint.limits[1]))
        
        # Solve optimization
        try:
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.success and result.fun < self.tolerance * 10:
                return result.x
            else:
                return None
        except Exception:
            return None
