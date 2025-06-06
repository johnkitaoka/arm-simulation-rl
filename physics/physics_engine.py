"""Physics engine using PyBullet for realistic simulation."""

import pybullet as p
import pybullet_data
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import tempfile

from core.config import config
from robot_arm.robot_arm import RobotArm


class PhysicsEngine:
    """Physics engine wrapper for PyBullet."""

    def __init__(self, gui: bool = True, gravity: Optional[List[float]] = None):
        """Initialize physics engine.

        Args:
            gui: Whether to show GUI
            gravity: Gravity vector [x, y, z]
        """
        self.gui = gui
        self.gravity = gravity or config.gravity

        # PyBullet connection
        self.physics_client = None
        self.robot_id = None
        self.ground_id = None

        # Object tracking
        self.objects = {}
        self.constraints = {}

        # Simulation parameters
        self.time_step = config.simulation_timestep
        self.real_time = False

        self._initialize_physics()

    def _initialize_physics(self) -> None:
        """Initialize PyBullet physics simulation."""
        # Connect to physics server
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Set additional search path for URDF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Configure simulation
        p.setGravity(*self.gravity)
        p.setTimeStep(self.time_step)
        p.setRealTimeSimulation(0)  # Disable real-time simulation

        # Load ground plane
        self.ground_id = p.loadURDF("plane.urdf")

        # Configure rendering
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.5]
            )

    def create_robot_urdf(self, robot_arm: RobotArm) -> str:
        """Create URDF file for the robot arm.

        Args:
            robot_arm: Robot arm instance

        Returns:
            Path to generated URDF file
        """
        # Generate URDF content
        urdf_content = self._generate_urdf(robot_arm)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_content)
            return f.name

    def _generate_urdf(self, robot_arm: RobotArm) -> str:
        """Generate URDF content for the robot arm.

        Args:
            robot_arm: Robot arm instance

        Returns:
            URDF XML content
        """
        urdf = '<?xml version="1.0"?>\n'
        urdf += '<robot name="anthropomorphic_arm">\n\n'

        # Base link
        urdf += '  <link name="base_link">\n'
        urdf += '    <visual>\n'
        urdf += '      <geometry>\n'
        urdf += '        <box size="0.1 0.1 0.05"/>\n'
        urdf += '      </geometry>\n'
        urdf += '      <material name="gray">\n'
        urdf += '        <color rgba="0.5 0.5 0.5 1"/>\n'
        urdf += '      </material>\n'
        urdf += '    </visual>\n'
        urdf += '    <collision>\n'
        urdf += '      <geometry>\n'
        urdf += '        <box size="0.1 0.1 0.05"/>\n'
        urdf += '      </geometry>\n'
        urdf += '    </collision>\n'
        urdf += '    <inertial>\n'
        urdf += '      <mass value="1.0"/>\n'
        urdf += '      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>\n'
        urdf += '    </inertial>\n'
        urdf += '  </link>\n\n'

        # All links that will be referenced by joints
        all_links = [
            ('upper_arm', 0.3, 2.0, '0.8 0.2 0.2'),
            ('forearm', 0.25, 1.5, '0.2 0.8 0.2'),
            ('hand', 0.15, 0.5, '0.2 0.2 0.8'),
            ('end_effector', 0.05, 0.1, '1.0 1.0 0.0')  # End effector link
        ]

        for link_name, length, mass, color in all_links:
            urdf += f'  <link name="{link_name}">\n'
            urdf += '    <visual>\n'
            urdf += '      <geometry>\n'
            urdf += f'        <cylinder radius="0.03" length="{length}"/>\n'
            urdf += '      </geometry>\n'
            urdf += f'      <material name="{link_name}_material">\n'
            urdf += f'        <color rgba="{color} 1"/>\n'
            urdf += '      </material>\n'
            urdf += '    </visual>\n'
            urdf += '    <collision>\n'
            urdf += '      <geometry>\n'
            urdf += f'        <cylinder radius="0.03" length="{length}"/>\n'
            urdf += '      </geometry>\n'
            urdf += '    </collision>\n'
            urdf += '    <inertial>\n'
            urdf += f'      <mass value="{mass}"/>\n'
            urdf += f'      <inertia ixx="{mass*length*length/12}" ixy="0" ixz="0" iyy="{mass*length*length/12}" iyz="0" izz="0.01"/>\n'
            urdf += '    </inertial>\n'
            urdf += '  </link>\n\n'

        # Simplified joint chain - only main arm joints that exist in robot_arm
        joint_configs = [
            ('shoulder_pitch', 'base_link', 'upper_arm', '0 0 0.05', '0 1 0'),
            ('elbow_flexion', 'upper_arm', 'forearm', '0.3 0 0', '0 1 0'),
            ('wrist_pitch', 'forearm', 'hand', '0.25 0 0', '0 1 0'),
            ('wrist_yaw', 'hand', 'end_effector', '0.15 0 0', '0 0 1')
        ]

        for joint_name, parent, child, origin, axis in joint_configs:
            if joint_name in robot_arm.joints:
                joint = robot_arm.joints[joint_name]
                urdf += f'  <joint name="{joint_name}" type="revolute">\n'
                urdf += f'    <parent link="{parent}"/>\n'
                urdf += f'    <child link="{child}"/>\n'
                urdf += f'    <origin xyz="{origin}" rpy="0 0 0"/>\n'
                urdf += f'    <axis xyz="{axis}"/>\n'
                urdf += f'    <limit lower="{joint.limits[0]}" upper="{joint.limits[1]}" effort="{joint.max_force}" velocity="{joint.max_velocity}"/>\n'
                urdf += '  </joint>\n\n'

        urdf += '</robot>\n'
        return urdf

    def load_robot(self, robot_arm: RobotArm, position: List[float] = [0, 0, 0]) -> int:
        """Load robot into physics simulation.

        Args:
            robot_arm: Robot arm instance
            position: Initial position

        Returns:
            Robot body ID
        """
        # Generate URDF file
        urdf_path = self.create_robot_urdf(robot_arm)

        try:
            # Load robot
            self.robot_id = p.loadURDF(
                urdf_path,
                basePosition=position,
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=True
            )

            # Store robot reference
            self.objects['robot'] = {
                'id': self.robot_id,
                'type': 'robot',
                'robot_arm': robot_arm
            }

            return self.robot_id

        except Exception as e:
            print(f"Error loading URDF: {e}")
            # Keep the URDF file for debugging
            print(f"URDF file saved at: {urdf_path}")
            raise

        finally:
            # Clean up temporary file only if successful
            if hasattr(self, 'robot_id') and self.robot_id is not None:
                if os.path.exists(urdf_path):
                    os.unlink(urdf_path)

    def add_object(self, name: str, shape: str, size: List[float],
                  position: List[float], color: List[float] = [1, 0, 0, 1],
                  mass: float = 1.0) -> int:
        """Add object to simulation.

        Args:
            name: Object name
            shape: Shape type ('box', 'sphere', 'cylinder')
            size: Shape dimensions
            position: Initial position
            color: RGBA color
            mass: Object mass

        Returns:
            Object body ID
        """
        # Create collision shape
        if shape == 'box':
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
        elif shape == 'sphere':
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=size[0])
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=size[0], rgbaColor=color)
        elif shape == 'cylinder':
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=size[0], height=size[1])
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=size[0], length=size[1], rgbaColor=color)
        else:
            raise ValueError(f"Unknown shape: {shape}")

        # Create multi-body
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )

        # Store object
        self.objects[name] = {
            'id': body_id,
            'type': 'object',
            'shape': shape,
            'size': size,
            'mass': mass
        }

        return body_id

    def set_joint_positions(self, joint_positions: Dict[str, float]) -> None:
        """Set robot joint positions.

        Args:
            joint_positions: Dictionary of joint names to positions
        """
        if self.robot_id is None:
            return

        # Get joint info
        num_joints = p.getNumJoints(self.robot_id)

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')

            if joint_name in joint_positions:
                p.resetJointState(self.robot_id, i, joint_positions[joint_name])

    def get_joint_states(self) -> Dict[str, Tuple[float, float]]:
        """Get current joint states.

        Returns:
            Dictionary of joint names to (position, velocity) tuples
        """
        if self.robot_id is None:
            return {}

        joint_states = {}
        num_joints = p.getNumJoints(self.robot_id)

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_state = p.getJointState(self.robot_id, i)

            joint_states[joint_name] = (joint_state[0], joint_state[1])  # position, velocity

        return joint_states

    def apply_joint_torques(self, joint_torques: Dict[str, float]) -> None:
        """Apply torques to robot joints.

        Args:
            joint_torques: Dictionary of joint names to torques
        """
        if self.robot_id is None:
            return

        num_joints = p.getNumJoints(self.robot_id)

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')

            if joint_name in joint_torques:
                p.setJointMotorControl2(
                    self.robot_id, i,
                    controlMode=p.TORQUE_CONTROL,
                    force=joint_torques[joint_name]
                )

    def step_simulation(self) -> None:
        """Step the physics simulation forward."""
        p.stepSimulation()

    def reset_simulation(self) -> None:
        """Reset the simulation."""
        p.resetSimulation()
        self._initialize_physics()
        self.objects.clear()
        self.constraints.clear()
        self.robot_id = None

    def get_contact_points(self, body_a: int, body_b: int = -1) -> List[Dict]:
        """Get contact points between bodies.

        Args:
            body_a: First body ID
            body_b: Second body ID (-1 for all bodies)

        Returns:
            List of contact point information
        """
        if body_b == -1:
            contacts = p.getContactPoints(body_a)
        else:
            contacts = p.getContactPoints(body_a, body_b)

        contact_info = []
        for contact in contacts:
            contact_info.append({
                'body_a': contact[1],
                'body_b': contact[2],
                'position': contact[5],
                'normal': contact[7],
                'distance': contact[8],
                'force': contact[9]
            })

        return contact_info

    def disconnect(self) -> None:
        """Disconnect from physics server."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
