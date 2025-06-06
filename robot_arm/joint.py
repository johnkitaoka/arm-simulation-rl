"""Joint definitions for the robot arm."""

import numpy as np
from typing import Tuple, Optional
from enum import Enum

from core.math_utils import clamp, normalize_angle


class JointType(Enum):
    """Types of joints in the robot arm."""
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"


class Joint:
    """Represents a single joint in the robot arm."""
    
    def __init__(self, name: str, joint_type: JointType, 
                 axis: np.ndarray, limits: Tuple[float, float],
                 max_velocity: float = 2.0, max_force: float = 100.0):
        """Initialize a joint.
        
        Args:
            name: Joint name
            joint_type: Type of joint
            axis: Joint rotation/translation axis
            limits: Joint limits (min, max)
            max_velocity: Maximum joint velocity
            max_force: Maximum joint force/torque
        """
        self.name = name
        self.type = joint_type
        self.axis = np.array(axis, dtype=np.float32)
        self.limits = limits
        self.max_velocity = max_velocity
        self.max_force = max_force
        
        # Current state
        self._position = 0.0
        self._velocity = 0.0
        self._acceleration = 0.0
        self._force = 0.0
        
        # Target state
        self._target_position = 0.0
        self._target_velocity = 0.0
    
    @property
    def position(self) -> float:
        """Get current joint position."""
        return self._position
    
    @position.setter
    def position(self, value: float) -> None:
        """Set joint position with limits checking."""
        if self.type == JointType.REVOLUTE:
            # For revolute joints, normalize angle
            self._position = normalize_angle(value)
            # Then clamp to limits
            self._position = clamp(self._position, self.limits[0], self.limits[1])
        else:
            self._position = clamp(value, self.limits[0], self.limits[1])
    
    @property
    def velocity(self) -> float:
        """Get current joint velocity."""
        return self._velocity
    
    @velocity.setter
    def velocity(self, value: float) -> None:
        """Set joint velocity with limits checking."""
        self._velocity = clamp(value, -self.max_velocity, self.max_velocity)
    
    @property
    def acceleration(self) -> float:
        """Get current joint acceleration."""
        return self._acceleration
    
    @acceleration.setter
    def acceleration(self, value: float) -> None:
        """Set joint acceleration."""
        self._acceleration = value
    
    @property
    def force(self) -> float:
        """Get current joint force/torque."""
        return self._force
    
    @force.setter
    def force(self, value: float) -> None:
        """Set joint force/torque with limits checking."""
        self._force = clamp(value, -self.max_force, self.max_force)
    
    @property
    def target_position(self) -> float:
        """Get target joint position."""
        return self._target_position
    
    @target_position.setter
    def target_position(self, value: float) -> None:
        """Set target joint position."""
        if self.type == JointType.REVOLUTE:
            self._target_position = normalize_angle(value)
            self._target_position = clamp(self._target_position, 
                                        self.limits[0], self.limits[1])
        else:
            self._target_position = clamp(value, self.limits[0], self.limits[1])
    
    @property
    def target_velocity(self) -> float:
        """Get target joint velocity."""
        return self._target_velocity
    
    @target_velocity.setter
    def target_velocity(self, value: float) -> None:
        """Set target joint velocity."""
        self._target_velocity = clamp(value, -self.max_velocity, self.max_velocity)
    
    def is_at_limit(self) -> bool:
        """Check if joint is at its limits."""
        return (abs(self._position - self.limits[0]) < 1e-6 or 
                abs(self._position - self.limits[1]) < 1e-6)
    
    def distance_to_limit(self) -> float:
        """Get minimum distance to joint limits."""
        return min(abs(self._position - self.limits[0]),
                  abs(self._position - self.limits[1]))
    
    def reset(self) -> None:
        """Reset joint to initial state."""
        self._position = 0.0
        self._velocity = 0.0
        self._acceleration = 0.0
        self._force = 0.0
        self._target_position = 0.0
        self._target_velocity = 0.0
    
    def update(self, dt: float, control_mode: str = "position") -> None:
        """Update joint state based on control mode.
        
        Args:
            dt: Time step
            control_mode: Control mode ("position", "velocity", "force")
        """
        if control_mode == "position":
            # Simple PD control for position
            kp = 50.0  # Proportional gain
            kd = 5.0   # Derivative gain
            
            error = self._target_position - self._position
            error_dot = self._target_velocity - self._velocity
            
            self._force = kp * error + kd * error_dot
            self._force = clamp(self._force, -self.max_force, self.max_force)
            
            # Simple integration (should be replaced with proper physics)
            self._acceleration = self._force / 1.0  # Assume unit mass
            self._velocity += self._acceleration * dt
            self._velocity = clamp(self._velocity, -self.max_velocity, self.max_velocity)
            self.position = self._position + self._velocity * dt
            
        elif control_mode == "velocity":
            self._velocity = self._target_velocity
            self.position = self._position + self._velocity * dt
            
        elif control_mode == "force":
            self._force = self._target_position  # In this mode, target_position is force
            # Physics integration would go here
            pass
    
    def __str__(self) -> str:
        """String representation of the joint."""
        return (f"Joint({self.name}, type={self.type.value}, "
                f"pos={self._position:.3f}, vel={self._velocity:.3f})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
