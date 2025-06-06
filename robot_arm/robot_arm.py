"""Main robot arm class with kinematics and control."""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import time

from .joint import Joint, JointType
from .link import Link
try:
    from .kinematics import ForwardKinematics, InverseKinematics
except ImportError:
    # Fallback if kinematics module has issues
    ForwardKinematics = None
    InverseKinematics = None
from core.config import config
from core.math_utils import (
    clamp, normalize_angle, interpolate_angles,
    homogeneous_transform, euler_to_rotation_matrix
)


class RobotArm:
    """Main robot arm class with full anthropomorphic design."""

    def __init__(self):
        """Initialize the robot arm with all joints and links."""
        self.joints = {}
        self.links = {}
        self.joint_order = []

        # Kinematics solvers
        self.fk_solver = None
        self.ik_solver = None

        # Control state
        self.control_mode = "position"  # "position", "velocity", "force"
        self.is_enabled = True

        # Initialize robot structure
        self._create_joints()
        self._create_links()
        self._setup_kinematics()

        # Current state
        self.last_update_time = time.time()

    def _create_joints(self) -> None:
        """Create all joints for the anthropomorphic arm."""
        joint_limits = config.joint_limits

        # Shoulder joints (3 DOF)
        self.joints['shoulder_pitch'] = Joint(
            'shoulder_pitch', JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),  # Y-axis
            limits=joint_limits.get('shoulder_pitch', [-1.57, 1.57])
        )

        self.joints['shoulder_yaw'] = Joint(
            'shoulder_yaw', JointType.REVOLUTE,
            axis=np.array([0, 0, 1]),  # Z-axis
            limits=joint_limits.get('shoulder_yaw', [-3.14, 3.14])
        )

        self.joints['shoulder_roll'] = Joint(
            'shoulder_roll', JointType.REVOLUTE,
            axis=np.array([1, 0, 0]),  # X-axis
            limits=joint_limits.get('shoulder_roll', [-1.57, 1.57])
        )

        # Elbow joint (1 DOF)
        self.joints['elbow_flexion'] = Joint(
            'elbow_flexion', JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),  # Y-axis
            limits=joint_limits.get('elbow_flexion', [0.0, 2.35])
        )

        # Wrist joints (2 DOF)
        self.joints['wrist_pitch'] = Joint(
            'wrist_pitch', JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),  # Y-axis
            limits=joint_limits.get('wrist_pitch', [-1.57, 1.57])
        )

        self.joints['wrist_yaw'] = Joint(
            'wrist_yaw', JointType.REVOLUTE,
            axis=np.array([0, 0, 1]),  # Z-axis
            limits=joint_limits.get('wrist_yaw', [-1.57, 1.57])
        )

        # Hand joints - 3 fingers + 1 thumb
        finger_limits = {
            'metacarpal': joint_limits.get('finger_metacarpal', [0.0, 1.57]),
            'proximal': joint_limits.get('finger_proximal', [0.0, 1.57]),
            'distal': joint_limits.get('finger_distal', [0.0, 1.57])
        }

        # Create finger joints
        for finger_idx in range(3):  # 3 fingers
            finger_name = f'finger_{finger_idx}'
            for joint_type in ['metacarpal', 'proximal', 'distal']:
                joint_name = f'{finger_name}_{joint_type}'
                self.joints[joint_name] = Joint(
                    joint_name, JointType.REVOLUTE,
                    axis=np.array([0, 1, 0]),  # Y-axis for flexion
                    limits=finger_limits[joint_type]
                )

        # Create thumb joints
        thumb_limits = {
            'metacarpal': joint_limits.get('thumb_metacarpal', [0.0, 1.57]),
            'interphalangeal': joint_limits.get('thumb_interphalangeal', [0.0, 1.57])
        }

        for joint_type in ['metacarpal', 'interphalangeal']:
            joint_name = f'thumb_{joint_type}'
            self.joints[joint_name] = Joint(
                joint_name, JointType.REVOLUTE,
                axis=np.array([0, 1, 0]),  # Y-axis
                limits=thumb_limits[joint_type]
            )

        # Define joint order for kinematics
        self.joint_order = [
            'shoulder_pitch', 'shoulder_yaw', 'shoulder_roll',
            'elbow_flexion', 'wrist_pitch', 'wrist_yaw'
        ]

        # Add finger joints to order
        for finger_idx in range(3):
            finger_name = f'finger_{finger_idx}'
            for joint_type in ['metacarpal', 'proximal', 'distal']:
                self.joint_order.append(f'{finger_name}_{joint_type}')

        # Add thumb joints
        self.joint_order.extend(['thumb_metacarpal', 'thumb_interphalangeal'])

    def _create_links(self) -> None:
        """Create all links for the robot arm."""
        link_lengths = config.link_lengths
        masses = config.masses

        # Main arm links
        self.links['upper_arm'] = Link(
            'upper_arm',
            length=link_lengths.get('upper_arm', 0.3),
            mass=masses.get('upper_arm', 2.0)
        )

        self.links['forearm'] = Link(
            'forearm',
            length=link_lengths.get('forearm', 0.25),
            mass=masses.get('forearm', 1.5)
        )

        self.links['hand'] = Link(
            'hand',
            length=link_lengths.get('hand', 0.15),
            mass=masses.get('hand', 0.5)
        )

        # Finger links
        finger_mass = masses.get('finger', 0.05)
        for finger_idx in range(3):
            finger_name = f'finger_{finger_idx}'
            for joint_type in ['metacarpal', 'proximal', 'distal']:
                link_name = f'{finger_name}_{joint_type}'
                length = link_lengths.get(f'finger_{joint_type}', 0.03)
                self.links[link_name] = Link(link_name, length, finger_mass)

        # Thumb links
        thumb_mass = masses.get('thumb', 0.04)
        for joint_type in ['metacarpal', 'interphalangeal']:
            link_name = f'thumb_{joint_type}'
            length = link_lengths.get(f'thumb_{joint_type}', 0.03)
            self.links[link_name] = Link(link_name, length, thumb_mass)

    def _setup_kinematics(self) -> None:
        """Setup forward and inverse kinematics solvers."""
        # Get main arm joints (excluding fingers for primary kinematics)
        main_joints = ['shoulder_pitch', 'shoulder_yaw', 'shoulder_roll',
                      'elbow_flexion', 'wrist_pitch', 'wrist_yaw']

        if ForwardKinematics is not None and InverseKinematics is not None:
            self.fk_solver = ForwardKinematics(self, main_joints)
            self.ik_solver = InverseKinematics(self, main_joints)
        else:
            print("Warning: Kinematics solvers not available")
            self.fk_solver = None
            self.ik_solver = None

    def get_joint_positions(self, joint_names: Optional[List[str]] = None) -> np.ndarray:
        """Get current joint positions.

        Args:
            joint_names: List of joint names (defaults to all joints)

        Returns:
            Array of joint positions
        """
        if joint_names is None:
            joint_names = self.joint_order

        return np.array([self.joints[name].position for name in joint_names])

    def set_joint_positions(self, positions: Union[np.ndarray, List[float]],
                           joint_names: Optional[List[str]] = None) -> None:
        """Set joint positions.

        Args:
            positions: Array or list of joint positions
            joint_names: List of joint names (defaults to all joints)
        """
        if joint_names is None:
            joint_names = self.joint_order

        for i, name in enumerate(joint_names):
            if i < len(positions):
                self.joints[name].position = positions[i]

    def get_joint_velocities(self, joint_names: Optional[List[str]] = None) -> np.ndarray:
        """Get current joint velocities."""
        if joint_names is None:
            joint_names = self.joint_order

        return np.array([self.joints[name].velocity for name in joint_names])

    def set_joint_velocities(self, velocities: Union[np.ndarray, List[float]],
                            joint_names: Optional[List[str]] = None) -> None:
        """Set joint velocities."""
        if joint_names is None:
            joint_names = self.joint_order

        for i, name in enumerate(joint_names):
            if i < len(velocities):
                self.joints[name].velocity = velocities[i]

    def set_joint_targets(self, targets: Union[np.ndarray, List[float]],
                         joint_names: Optional[List[str]] = None) -> None:
        """Set target joint positions.

        Args:
            targets: Array or list of target positions
            joint_names: List of joint names (defaults to all joints)
        """
        if joint_names is None:
            joint_names = self.joint_order

        for i, name in enumerate(joint_names):
            if i < len(targets):
                self.joints[name].target_position = targets[i]

    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get end effector position and orientation.

        Returns:
            Tuple of (position, rotation_matrix)
        """
        if self.fk_solver is None:
            raise RuntimeError("Forward kinematics solver not initialized")

        return self.fk_solver.compute_end_effector_pose()

    def solve_inverse_kinematics(self, target_position: np.ndarray,
                                target_orientation: Optional[np.ndarray] = None,
                                initial_guess: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Solve inverse kinematics for target pose.

        Args:
            target_position: Target end effector position
            target_orientation: Target end effector orientation (rotation matrix)
            initial_guess: Initial joint angle guess

        Returns:
            Joint angles that achieve target pose, or None if no solution
        """
        if self.ik_solver is None:
            raise RuntimeError("Inverse kinematics solver not initialized")

        return self.ik_solver.solve(target_position, target_orientation, initial_guess)

    def move_to_pose(self, target_position: np.ndarray,
                    target_orientation: Optional[np.ndarray] = None,
                    duration: float = 2.0) -> bool:
        """Move arm to target pose using inverse kinematics.

        Args:
            target_position: Target end effector position
            target_orientation: Target end effector orientation
            duration: Movement duration in seconds

        Returns:
            True if movement was successful
        """
        # Solve inverse kinematics
        target_joints = self.solve_inverse_kinematics(target_position, target_orientation)

        if target_joints is None:
            return False

        # Set targets for main arm joints
        main_joints = ['shoulder_pitch', 'shoulder_yaw', 'shoulder_roll',
                      'elbow_flexion', 'wrist_pitch', 'wrist_yaw']
        self.set_joint_targets(target_joints, main_joints)

        return True

    def move_joints_smoothly(self, target_positions: np.ndarray,
                           joint_names: Optional[List[str]] = None,
                           duration: float = 2.0, steps: int = 100) -> None:
        """Move joints smoothly to target positions.

        Args:
            target_positions: Target joint positions
            joint_names: Joint names to move
            duration: Movement duration
            steps: Number of interpolation steps
        """
        if joint_names is None:
            joint_names = self.joint_order

        start_positions = self.get_joint_positions(joint_names)

        for i in range(steps + 1):
            t = i / steps
            interpolated = interpolate_angles(start_positions, target_positions, t)
            self.set_joint_positions(interpolated, joint_names)
            time.sleep(duration / steps)

    def reset_to_home(self) -> None:
        """Reset arm to home position."""
        home_positions = np.zeros(len(self.joint_order))
        self.set_joint_positions(home_positions)
        self.set_joint_targets(home_positions)

    def update(self, dt: Optional[float] = None) -> None:
        """Update robot arm state.

        Args:
            dt: Time step (auto-calculated if None)
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_update_time
        self.last_update_time = current_time

        if not self.is_enabled:
            return

        # Update all joints
        for joint in self.joints.values():
            joint.update(dt, self.control_mode)

        # Update forward kinematics
        if self.fk_solver is not None:
            self.fk_solver.update()

    def check_self_collision(self) -> bool:
        """Check for self-collision between links.

        Returns:
            True if collision detected
        """
        # Simplified collision checking
        # In a full implementation, this would check all link pairs
        return False

    def check_workspace_limits(self, position: np.ndarray) -> bool:
        """Check if position is within workspace limits.

        Args:
            position: 3D position to check

        Returns:
            True if position is reachable
        """
        # Simple spherical workspace check
        max_reach = (self.links['upper_arm'].length +
                    self.links['forearm'].length +
                    self.links['hand'].length)

        distance = np.linalg.norm(position)
        return distance <= max_reach

    def get_joint_info(self) -> Dict[str, Dict]:
        """Get information about all joints.

        Returns:
            Dictionary with joint information
        """
        info = {}
        for name, joint in self.joints.items():
            info[name] = {
                'position': joint.position,
                'velocity': joint.velocity,
                'target_position': joint.target_position,
                'limits': joint.limits,
                'at_limit': joint.is_at_limit()
            }
        return info

    def enable(self) -> None:
        """Enable the robot arm."""
        self.is_enabled = True

    def disable(self) -> None:
        """Disable the robot arm."""
        self.is_enabled = False

    def emergency_stop(self) -> None:
        """Emergency stop - disable arm and zero velocities."""
        self.disable()
        for joint in self.joints.values():
            joint.velocity = 0.0
            joint.target_velocity = 0.0

    def __str__(self) -> str:
        """String representation of the robot arm."""
        return f"RobotArm(joints={len(self.joints)}, enabled={self.is_enabled})"
