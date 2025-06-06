"""Link definitions for the robot arm."""

import numpy as np
from typing import Optional, List, Tuple

from core.math_utils import homogeneous_transform


class Link:
    """Represents a link in the robot arm."""
    
    def __init__(self, name: str, length: float, mass: float,
                 local_com: Optional[np.ndarray] = None,
                 inertia: Optional[np.ndarray] = None,
                 visual_mesh: Optional[str] = None,
                 collision_mesh: Optional[str] = None):
        """Initialize a link.
        
        Args:
            name: Link name
            length: Link length
            mass: Link mass
            local_com: Center of mass in local coordinates
            inertia: 3x3 inertia tensor
            visual_mesh: Path to visual mesh file
            collision_mesh: Path to collision mesh file
        """
        self.name = name
        self.length = length
        self.mass = mass
        
        # Default center of mass at link center
        self.local_com = local_com if local_com is not None else np.array([length/2, 0, 0])
        
        # Default inertia for a uniform rod
        if inertia is not None:
            self.inertia = inertia
        else:
            # Inertia of a uniform rod about its center
            Ixx = mass * length**2 / 12
            self.inertia = np.diag([Ixx, Ixx, 0])
        
        self.visual_mesh = visual_mesh
        self.collision_mesh = collision_mesh
        
        # Transform matrices
        self._local_transform = np.eye(4)
        self._world_transform = np.eye(4)
        
        # Collision geometry (simplified as cylinders/boxes)
        self.collision_shapes = []
        self._add_default_collision_shape()
    
    def _add_default_collision_shape(self) -> None:
        """Add default collision shape for the link."""
        # Default: cylinder along x-axis
        self.collision_shapes.append({
            'type': 'cylinder',
            'radius': 0.02,  # 2cm radius
            'height': self.length,
            'position': np.array([self.length/2, 0, 0]),
            'orientation': np.array([0, np.pi/2, 0])  # Rotate to align with x-axis
        })
    
    def add_collision_shape(self, shape_type: str, **kwargs) -> None:
        """Add a collision shape to the link.
        
        Args:
            shape_type: Type of shape ('box', 'cylinder', 'sphere')
            **kwargs: Shape-specific parameters
        """
        shape = {'type': shape_type}
        shape.update(kwargs)
        self.collision_shapes.append(shape)
    
    def set_local_transform(self, position: np.ndarray, 
                           rotation_matrix: np.ndarray) -> None:
        """Set the local transformation matrix.
        
        Args:
            position: 3D position vector
            rotation_matrix: 3x3 rotation matrix
        """
        self._local_transform = homogeneous_transform(rotation_matrix, position)
    
    def set_world_transform(self, transform: np.ndarray) -> None:
        """Set the world transformation matrix.
        
        Args:
            transform: 4x4 transformation matrix
        """
        self._world_transform = transform.copy()
    
    @property
    def local_transform(self) -> np.ndarray:
        """Get local transformation matrix."""
        return self._local_transform.copy()
    
    @property
    def world_transform(self) -> np.ndarray:
        """Get world transformation matrix."""
        return self._world_transform.copy()
    
    @property
    def world_position(self) -> np.ndarray:
        """Get world position of the link."""
        return self._world_transform[:3, 3]
    
    @property
    def world_orientation(self) -> np.ndarray:
        """Get world orientation matrix of the link."""
        return self._world_transform[:3, :3]
    
    @property
    def world_com(self) -> np.ndarray:
        """Get world position of center of mass."""
        local_com_homogeneous = np.append(self.local_com, 1)
        world_com_homogeneous = self._world_transform @ local_com_homogeneous
        return world_com_homogeneous[:3]
    
    def get_collision_shapes_world(self) -> List[dict]:
        """Get collision shapes in world coordinates.
        
        Returns:
            List of collision shapes with world coordinates
        """
        world_shapes = []
        
        for shape in self.collision_shapes:
            world_shape = shape.copy()
            
            # Transform position to world coordinates
            if 'position' in shape:
                local_pos_homogeneous = np.append(shape['position'], 1)
                world_pos_homogeneous = self._world_transform @ local_pos_homogeneous
                world_shape['position'] = world_pos_homogeneous[:3]
            
            # Transform orientation to world coordinates
            if 'orientation' in shape:
                # Convert orientation to rotation matrix, transform, then back
                from core.math_utils import euler_to_rotation_matrix, rotation_matrix_to_euler
                local_rot = euler_to_rotation_matrix(*shape['orientation'])
                world_rot = self.world_orientation @ local_rot
                world_shape['orientation'] = rotation_matrix_to_euler(world_rot)
            
            world_shapes.append(world_shape)
        
        return world_shapes
    
    def check_collision_with_point(self, point: np.ndarray, 
                                  margin: float = 0.0) -> bool:
        """Check if a point collides with this link.
        
        Args:
            point: 3D point to check
            margin: Safety margin
            
        Returns:
            True if collision detected
        """
        world_shapes = self.get_collision_shapes_world()
        
        for shape in world_shapes:
            if shape['type'] == 'sphere':
                distance = np.linalg.norm(point - shape['position'])
                if distance <= shape['radius'] + margin:
                    return True
                    
            elif shape['type'] == 'cylinder':
                # Simplified cylinder collision (treat as sphere for now)
                distance = np.linalg.norm(point - shape['position'])
                if distance <= shape['radius'] + margin:
                    return True
                    
            elif shape['type'] == 'box':
                # Simplified box collision (treat as sphere for now)
                if 'size' in shape:
                    max_extent = max(shape['size']) / 2
                    distance = np.linalg.norm(point - shape['position'])
                    if distance <= max_extent + margin:
                        return True
        
        return False
    
    def __str__(self) -> str:
        """String representation of the link."""
        return f"Link({self.name}, length={self.length:.3f}, mass={self.mass:.3f})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
