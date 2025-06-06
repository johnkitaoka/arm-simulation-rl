#!/usr/bin/env python3
"""Basic usage examples for the robot arm simulation."""

import sys
import os
import time
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_arm.robot_arm import RobotArm
from physics.physics_engine import PhysicsEngine
from ml.nlp_processor import CommandParser
from ml.rl_trainer import RLTrainer, RobotArmEnv


def example_basic_robot_control():
    """Example 1: Basic robot arm control."""
    print("Example 1: Basic Robot Arm Control")
    print("-" * 40)
    
    # Create robot arm
    robot = RobotArm()
    print(f"Created robot with {len(robot.joints)} joints")
    
    # Get current joint positions
    positions = robot.get_joint_positions()
    print(f"Initial joint positions: {positions[:6]}")  # Show first 6 joints
    
    # Set some target positions
    target_positions = np.array([0.5, 0.3, -0.2, 1.0, 0.4, -0.1])
    main_joints = ['shoulder_pitch', 'shoulder_yaw', 'shoulder_roll',
                   'elbow_flexion', 'wrist_pitch', 'wrist_yaw']
    
    robot.set_joint_targets(target_positions, main_joints)
    print(f"Set target positions: {target_positions}")
    
    # Update robot for a few steps
    for i in range(10):
        robot.update(dt=0.1)
        if i % 3 == 0:
            current_pos = robot.get_joint_positions(main_joints)
            print(f"Step {i}: {current_pos}")
    
    # Get end effector pose
    try:
        ee_pos, ee_rot = robot.get_end_effector_pose()
        print(f"End effector position: {ee_pos}")
    except Exception as e:
        print(f"Could not get end effector pose: {e}")
    
    # Reset to home
    robot.reset_to_home()
    print("Reset to home position")


def example_inverse_kinematics():
    """Example 2: Inverse kinematics."""
    print("\nExample 2: Inverse Kinematics")
    print("-" * 40)
    
    robot = RobotArm()
    
    # Define target positions
    target_positions = [
        [0.3, 0.0, 0.3],   # Forward
        [0.2, 0.2, 0.4],   # Right and up
        [0.2, -0.2, 0.4],  # Left and up
        [0.4, 0.0, 0.2],   # Forward and down
    ]
    
    for i, target_pos in enumerate(target_positions):
        print(f"\nTarget {i+1}: {target_pos}")
        
        # Solve inverse kinematics
        target_joints = robot.solve_inverse_kinematics(np.array(target_pos))
        
        if target_joints is not None:
            print(f"IK solution found: {target_joints}")
            
            # Move to target
            success = robot.move_to_pose(np.array(target_pos))
            print(f"Movement {'successful' if success else 'failed'}")
            
            # Verify position
            try:
                actual_pos, _ = robot.get_end_effector_pose()
                error = np.linalg.norm(actual_pos - np.array(target_pos))
                print(f"Position error: {error:.4f}m")
            except:
                print("Could not verify position")
        else:
            print("No IK solution found")


def example_natural_language_commands():
    """Example 3: Natural language command processing."""
    print("\nExample 3: Natural Language Commands")
    print("-" * 40)
    
    robot = RobotArm()
    parser = CommandParser()
    
    # Test commands
    commands = [
        "move forward",
        "wave hello",
        "point at target",
        "move to position 0.3, 0.0, 0.4",
        "reset to home",
        "stop the robot"
    ]
    
    for command in commands:
        print(f"\nCommand: '{command}'")
        
        # Parse command
        parsed = parser.parse_command(command)
        print(f"Parsed action: {parsed['action']}")
        print(f"Confidence: {parsed['confidence']:.2f}")
        
        # Convert to robot action
        action = parser.command_to_robot_action(parsed)
        print(f"Robot action: {action['type']}")
        
        if action['success']:
            print("✓ Command understood and ready for execution")
        else:
            print("✗ Command not understood")


def example_physics_simulation():
    """Example 4: Physics simulation."""
    print("\nExample 4: Physics Simulation")
    print("-" * 40)
    
    try:
        # Create robot and physics engine
        robot = RobotArm()
        physics = PhysicsEngine(gui=False)  # Headless mode
        
        print("Created physics engine")
        
        # Load robot into physics
        robot_id = physics.load_robot(robot)
        print(f"Loaded robot into physics (ID: {robot_id})")
        
        # Add some objects to the scene
        ball_id = physics.add_object("ball", "sphere", [0.05], [0.5, 0.0, 0.3], [1, 0, 0, 1])
        cube_id = physics.add_object("cube", "box", [0.05, 0.05, 0.05], [0.3, 0.2, 0.1], [0, 1, 0, 1])
        
        print(f"Added ball (ID: {ball_id}) and cube (ID: {cube_id})")
        
        # Run simulation for a few steps
        for i in range(100):
            physics.step_simulation()
            
            if i % 20 == 0:
                joint_states = physics.get_joint_states()
                print(f"Step {i}: {len(joint_states)} joints simulated")
        
        # Check for contacts
        contacts = physics.get_contact_points(robot_id)
        print(f"Found {len(contacts)} contact points")
        
        physics.disconnect()
        print("Physics simulation completed")
        
    except Exception as e:
        print(f"Physics simulation failed: {e}")


def example_reinforcement_learning():
    """Example 5: Reinforcement learning environment."""
    print("\nExample 5: Reinforcement Learning")
    print("-" * 40)
    
    try:
        # Create robot and RL environment
        robot = RobotArm()
        env = RobotArmEnv(robot)
        
        print(f"Created RL environment")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Run a few episodes
        for episode in range(3):
            print(f"\nEpisode {episode + 1}")
            
            obs, info = env.reset()
            print(f"Initial observation shape: {obs.shape}")
            print(f"Target position: {info['target_position']}")
            
            episode_reward = 0
            for step in range(50):
                # Random action
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                
                if step % 10 == 0:
                    print(f"  Step {step}: reward={reward:.3f}, distance={info['distance_to_target']:.3f}")
                
                if terminated or truncated:
                    break
            
            print(f"Episode reward: {episode_reward:.2f}")
            print(f"Final distance to target: {info['distance_to_target']:.3f}")
        
    except Exception as e:
        print(f"RL environment test failed: {e}")


def example_training_setup():
    """Example 6: Setting up training (without actually training)."""
    print("\nExample 6: Training Setup")
    print("-" * 40)
    
    try:
        # Create robot and trainer
        robot = RobotArm()
        trainer = RLTrainer(robot)
        
        print("Created RL trainer")
        print("Training configuration:")
        print(f"  Environment: {type(trainer.env).__name__}")
        print(f"  Action space: {trainer.env.action_space}")
        print(f"  Observation space: {trainer.env.observation_space}")
        
        # Note: We don't actually train here as it would take too long
        print("\nTo start training, you would call:")
        print("  trainer.train('PPO', timesteps=100000)")
        print("\nOr use the command line:")
        print("  python main.py --train PPO --timesteps 100000")
        
    except Exception as e:
        print(f"Training setup failed: {e}")


def main():
    """Run all examples."""
    print("Robot Arm Simulation - Basic Usage Examples")
    print("=" * 50)
    
    examples = [
        example_basic_robot_control,
        example_inverse_kinematics,
        example_natural_language_commands,
        example_physics_simulation,
        example_reinforcement_learning,
        example_training_setup,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\nExample {i} failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(examples):
            print("\n" + "=" * 50)
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNext steps:")
    print("1. Run the full simulation: python main.py")
    print("2. Try natural language commands: python main.py --command 'wave hello'")
    print("3. Run the movement demo: python main.py --demo")
    print("4. Start training: python main.py --train PPO")


if __name__ == "__main__":
    main()
