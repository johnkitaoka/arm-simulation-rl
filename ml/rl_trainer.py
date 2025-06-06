"""Reinforcement Learning trainer for robot arm control."""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import os
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import time

from core.config import config
from robot_arm.robot_arm import RobotArm
from physics.physics_engine import PhysicsEngine


class RobotArmEnv(gym.Env):
    """Gymnasium environment for robot arm training."""
    
    def __init__(self, robot_arm: RobotArm, physics_engine: Optional[PhysicsEngine] = None,
                 render_mode: Optional[str] = None):
        """Initialize the environment.
        
        Args:
            robot_arm: Robot arm instance
            physics_engine: Physics engine (optional)
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.robot_arm = robot_arm
        self.physics_engine = physics_engine
        self.render_mode = render_mode
        
        # Get main arm joints (excluding fingers for now)
        self.main_joints = ['shoulder_pitch', 'shoulder_yaw', 'shoulder_roll',
                           'elbow_flexion', 'wrist_pitch', 'wrist_yaw']
        
        # Action space: joint velocities or positions
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(len(self.main_joints),), 
            dtype=np.float32
        )
        
        # Observation space: joint positions, velocities, end effector pose, target
        obs_dim = (
            len(self.main_joints) * 2 +  # joint positions and velocities
            6 +  # end effector position and orientation (euler)
            3    # target position
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Environment state
        self.target_position = np.array([0.3, 0.0, 0.3])
        self.max_episode_steps = 500
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Reward configuration
        self.reward_config = config.ml_config.get('rewards', {})
        
        # Reset to initial state
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Observation and info
        """
        super().reset(seed=seed)
        
        # Reset robot to home position with small random perturbation
        if seed is not None:
            np.random.seed(seed)
        
        home_positions = np.zeros(len(self.main_joints))
        noise = np.random.normal(0, 0.1, len(self.main_joints))
        initial_positions = home_positions + noise
        
        self.robot_arm.set_joint_positions(initial_positions, self.main_joints)
        self.robot_arm.set_joint_velocities(np.zeros(len(self.main_joints)), self.main_joints)
        
        # Set random target position within workspace
        self._generate_random_target()
        
        # Reset episode state
        self.current_step = 0
        self.episode_reward = 0.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Observation, reward, terminated, truncated, info
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Convert action to joint targets (position control)
        current_positions = self.robot_arm.get_joint_positions(self.main_joints)
        max_delta = 0.1  # Maximum change per step
        target_positions = current_positions + action * max_delta
        
        # Apply joint limits
        for i, joint_name in enumerate(self.main_joints):
            joint = self.robot_arm.joints[joint_name]
            target_positions[i] = np.clip(target_positions[i], 
                                        joint.limits[0], joint.limits[1])
        
        # Set targets and update robot
        self.robot_arm.set_joint_targets(target_positions, self.main_joints)
        self.robot_arm.update()
        
        # Update physics if available
        if self.physics_engine:
            self.physics_engine.step_simulation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        self.current_step += 1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Observation vector
        """
        # Joint positions and velocities
        positions = self.robot_arm.get_joint_positions(self.main_joints)
        velocities = self.robot_arm.get_joint_velocities(self.main_joints)
        
        # End effector pose
        try:
            ee_pos, ee_rot = self.robot_arm.get_end_effector_pose()
            # Convert rotation matrix to euler angles
            from core.math_utils import rotation_matrix_to_euler
            ee_euler = np.array(rotation_matrix_to_euler(ee_rot))
        except:
            ee_pos = np.zeros(3)
            ee_euler = np.zeros(3)
        
        # Combine all observations
        observation = np.concatenate([
            positions,
            velocities,
            ee_pos,
            ee_euler,
            self.target_position
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for current state and action.
        
        Args:
            action: Action taken
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        try:
            # Distance to target reward
            ee_pos, _ = self.robot_arm.get_end_effector_pose()
            distance = np.linalg.norm(ee_pos - self.target_position)
            
            # Exponential reward for reaching target
            reach_reward = self.reward_config.get('reach_target', 100.0)
            reward += reach_reward * np.exp(-10 * distance)
            
            # Penalty for being far from target
            if distance > 0.5:
                reward -= distance * 10
            
            # Smooth movement reward (penalize large actions)
            smooth_reward = self.reward_config.get('smooth_movement', 10.0)
            action_penalty = np.sum(np.square(action))
            reward += smooth_reward * np.exp(-action_penalty)
            
            # Energy efficiency (penalize high velocities)
            velocities = self.robot_arm.get_joint_velocities(self.main_joints)
            energy_reward = self.reward_config.get('energy_efficiency', 5.0)
            velocity_penalty = np.sum(np.square(velocities))
            reward += energy_reward * np.exp(-velocity_penalty)
            
            # Collision penalty
            if self.robot_arm.check_self_collision():
                collision_penalty = self.reward_config.get('collision_penalty', -50.0)
                reward += collision_penalty
            
            # Joint limit penalty
            for joint_name in self.main_joints:
                joint = self.robot_arm.joints[joint_name]
                if joint.is_at_limit():
                    reward -= 10.0
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            reward = -1.0
        
        return reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate.
        
        Returns:
            True if episode should end
        """
        try:
            # Check if target is reached
            ee_pos, _ = self.robot_arm.get_end_effector_pose()
            distance = np.linalg.norm(ee_pos - self.target_position)
            
            if distance < 0.05:  # 5cm tolerance
                return True
            
            # Check for collisions
            if self.robot_arm.check_self_collision():
                return True
            
            # Check if end effector is too far from workspace
            if np.linalg.norm(ee_pos) > 1.0:
                return True
                
        except Exception:
            return True
        
        return False
    
    def _generate_random_target(self) -> None:
        """Generate a random target position within workspace."""
        # Generate target within reachable workspace
        max_reach = 0.7  # Conservative estimate
        
        # Spherical coordinates
        r = np.random.uniform(0.2, max_reach)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        
        # Convert to cartesian
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi) + 0.1  # Offset from ground
        
        self.target_position = np.array([x, y, z])
    
    def _get_info(self) -> Dict:
        """Get additional info about the environment state.
        
        Returns:
            Info dictionary
        """
        try:
            ee_pos, _ = self.robot_arm.get_end_effector_pose()
            distance = np.linalg.norm(ee_pos - self.target_position)
        except:
            distance = float('inf')
        
        return {
            'distance_to_target': distance,
            'episode_reward': self.episode_reward,
            'current_step': self.current_step,
            'target_position': self.target_position.copy()
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment.
        
        Returns:
            Rendered image if applicable
        """
        if self.render_mode == "human":
            # This would integrate with the visualization system
            pass
        return None


class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """Called at each step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        if len(self.locals.get('episode_rewards', [])) > 0:
            self.episode_rewards.extend(self.locals['episode_rewards'])
            self.episode_lengths.extend(self.locals['episode_lengths'])


class RLTrainer:
    """Reinforcement learning trainer for robot arm."""
    
    def __init__(self, robot_arm: RobotArm, physics_engine: Optional[PhysicsEngine] = None):
        """Initialize the trainer.
        
        Args:
            robot_arm: Robot arm instance
            physics_engine: Physics engine (optional)
        """
        self.robot_arm = robot_arm
        self.physics_engine = physics_engine
        self.env = None
        self.model = None
        
        # Training configuration
        self.training_config = config.ml_config.get('training', {})
        self.model_save_path = config.ml_config.get('model_save_path', 'models/')
        
        # Create environment
        self._create_environment()
    
    def _create_environment(self) -> None:
        """Create the training environment."""
        self.env = RobotArmEnv(self.robot_arm, self.physics_engine)
    
    def train(self, algorithm: str = "PPO", total_timesteps: int = 100000,
              save_path: Optional[str] = None) -> None:
        """Train the robot arm using reinforcement learning.
        
        Args:
            algorithm: RL algorithm to use ("PPO", "SAC", "TD3")
            total_timesteps: Total training timesteps
            save_path: Path to save the trained model
        """
        print(f"Starting training with {algorithm} for {total_timesteps} timesteps")
        
        # Create model
        if algorithm == "PPO":
            self.model = PPO("MlpPolicy", self.env, verbose=1,
                           learning_rate=self.training_config.get('learning_rate', 3e-4))
        elif algorithm == "SAC":
            self.model = SAC("MlpPolicy", self.env, verbose=1,
                           learning_rate=self.training_config.get('learning_rate', 3e-4))
        elif algorithm == "TD3":
            self.model = TD3("MlpPolicy", self.env, verbose=1,
                           learning_rate=self.training_config.get('learning_rate', 3e-4))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Create callback
        callback = TrainingCallback()
        
        # Train the model
        start_time = time.time()
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save model
        if save_path is None:
            save_path = os.path.join(self.model_save_path, f"{algorithm}_robot_arm.zip")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str, algorithm: str = "PPO") -> None:
        """Load a trained model.
        
        Args:
            model_path: Path to the saved model
            algorithm: Algorithm used for training
        """
        if algorithm == "PPO":
            self.model = PPO.load(model_path, env=self.env)
        elif algorithm == "SAC":
            self.model = SAC.load(model_path, env=self.env)
        elif algorithm == "TD3":
            self.model = TD3.load(model_path, env=self.env)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        print(f"Model loaded from {model_path}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained model.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded for evaluation")
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    if info['distance_to_target'] < 0.05:
                        success_count += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
                  f"Length = {episode_length}, "
                  f"Final distance = {info['distance_to_target']:.3f}")
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': success_count / num_episodes
        }
        
        print(f"\nEvaluation Results:")
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Episode Length: {metrics['mean_length']:.1f}")
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        
        return metrics
