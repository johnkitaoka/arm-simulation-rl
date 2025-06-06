"""Configuration management for robot arm simulation."""

import yaml
import os
from typing import Dict, Any, List, Tuple


class Config:
    """Configuration manager for the robot arm simulation."""
    
    def __init__(self, config_path: str = "config/robot_config.yaml"):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'robot_arm.base_position')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Optional path to save to (defaults to original config_path)
        """
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
    
    # Convenience properties for commonly used configurations
    @property
    def joint_limits(self) -> Dict[str, List[float]]:
        """Get joint limits configuration."""
        return self.get('robot_arm.joint_limits', {})
    
    @property
    def link_lengths(self) -> Dict[str, float]:
        """Get link lengths configuration."""
        return self.get('robot_arm.link_lengths', {})
    
    @property
    def masses(self) -> Dict[str, float]:
        """Get mass properties configuration."""
        return self.get('robot_arm.masses', {})
    
    @property
    def simulation_timestep(self) -> float:
        """Get simulation time step."""
        return self.get('simulation.time_step', 0.01)
    
    @property
    def gravity(self) -> List[float]:
        """Get gravity vector."""
        return self.get('simulation.gravity', [0.0, 0.0, -9.81])
    
    @property
    def window_size(self) -> Tuple[int, int]:
        """Get visualization window size."""
        size = self.get('visualization.window_size', [1200, 800])
        return tuple(size)
    
    @property
    def camera_config(self) -> Dict[str, Any]:
        """Get camera configuration."""
        return self.get('visualization.camera', {})
    
    @property
    def colors(self) -> Dict[str, List[float]]:
        """Get color configuration."""
        return self.get('visualization.colors', {})
    
    @property
    def ml_config(self) -> Dict[str, Any]:
        """Get machine learning configuration."""
        return self.get('ml', {})


# Global configuration instance
config = Config()
