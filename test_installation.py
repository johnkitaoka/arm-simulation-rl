#!/usr/bin/env python3
"""Test script to verify robot arm simulation installation."""

import sys
import os
import traceback

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import numpy as np
        print("✓ NumPy")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False

    try:
        import scipy
        print("✓ SciPy")
    except ImportError as e:
        print(f"✗ SciPy: {e}")
        return False

    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML: {e}")
        return False

    try:
        import pybullet
        print("✓ PyBullet")
    except ImportError as e:
        print(f"✗ PyBullet: {e}")
        return False

    try:
        import torch
        print("✓ PyTorch")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False

    try:
        import transformers
        print("✓ Transformers")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        return False

    try:
        import stable_baselines3
        print("✓ Stable-Baselines3")
    except ImportError as e:
        print(f"✗ Stable-Baselines3: {e}")
        return False

    try:
        import gymnasium
        print("✓ Gymnasium")
    except ImportError as e:
        print(f"✗ Gymnasium: {e}")
        return False

    try:
        import cv2
        print("✓ OpenCV")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False

    try:
        import tkinter
        print("✓ Tkinter")
    except ImportError as e:
        print(f"✗ Tkinter: {e}")
        return False

    # Optional OpenGL imports (may fail on headless systems)
    try:
        import glfw
        print("✓ GLFW")
    except ImportError as e:
        print(f"⚠ GLFW (optional): {e}")

    try:
        import OpenGL.GL
        import OpenGL.GLU
        print("✓ PyOpenGL")
    except ImportError as e:
        print(f"⚠ PyOpenGL (optional): {e}")

    return True

def test_core_modules():
    """Test that core project modules can be imported."""
    print("\nTesting core modules...")

    try:
        from core.config import config
        print("✓ Core config")
    except ImportError as e:
        print(f"✗ Core config: {e}")
        return False

    try:
        from core.math_utils import normalize_angle
        print("✓ Core math utils")
    except ImportError as e:
        print(f"✗ Core math utils: {e}")
        return False

    try:
        from robot_arm.joint import Joint, JointType
        print("✓ Robot arm joint")
    except ImportError as e:
        print(f"✗ Robot arm joint: {e}")
        return False

    try:
        from robot_arm.link import Link
        print("✓ Robot arm link")
    except ImportError as e:
        print(f"✗ Robot arm link: {e}")
        return False

    try:
        from robot_arm.robot_arm import RobotArm
        print("✓ Robot arm main")
    except ImportError as e:
        print(f"✗ Robot arm main: {e}")
        return False

    try:
        from physics.physics_engine import PhysicsEngine
        print("✓ Physics engine")
    except ImportError as e:
        print(f"✗ Physics engine: {e}")
        return False

    try:
        from ml.nlp_processor import CommandParser
        print("✓ NLP processor")
    except ImportError as e:
        print(f"✗ NLP processor: {e}")
        return False

    try:
        from ml.rl_trainer import RLTrainer
        print("✓ RL trainer")
    except ImportError as e:
        print(f"✗ RL trainer: {e}")
        return False

    try:
        from ui.control_panel import ControlPanel
        print("✓ UI control panel")
    except ImportError as e:
        print(f"✗ UI control panel: {e}")
        return False

    return True

def test_robot_arm_creation():
    """Test creating a robot arm instance."""
    print("\nTesting robot arm creation...")

    try:
        from robot_arm.robot_arm import RobotArm

        robot = RobotArm()
        print(f"✓ Robot arm created with {len(robot.joints)} joints")

        # Test basic functionality
        positions = robot.get_joint_positions()
        print(f"✓ Got joint positions: {len(positions)} values")

        robot.reset_to_home()
        print("✓ Reset to home position")

        return True

    except Exception as e:
        print(f"✗ Robot arm creation failed: {e}")
        traceback.print_exc()
        return False

def test_physics_engine():
    """Test physics engine creation."""
    print("\nTesting physics engine...")

    try:
        from physics.physics_engine import PhysicsEngine
        from robot_arm.robot_arm import RobotArm

        # Create headless physics engine
        physics = PhysicsEngine(gui=False)
        print("✓ Physics engine created")

        # Create robot and load into physics
        robot = RobotArm()
        robot_id = physics.load_robot(robot)
        print(f"✓ Robot loaded into physics (ID: {robot_id})")

        # Test simulation step
        physics.step_simulation()
        print("✓ Physics simulation step")

        physics.disconnect()
        print("✓ Physics engine disconnected")

        return True

    except Exception as e:
        print(f"✗ Physics engine test failed: {e}")
        traceback.print_exc()
        return False

def test_nlp_processor():
    """Test natural language processing."""
    print("\nTesting NLP processor...")

    try:
        from ml.nlp_processor import CommandParser

        parser = CommandParser()
        print("✓ Command parser created")

        # Test command parsing
        test_commands = [
            "move forward",
            "wave hello",
            "point at target",
            "reset to home"
        ]

        for command in test_commands:
            parsed = parser.parse_command(command)
            action = parser.command_to_robot_action(parsed)
            print(f"✓ Parsed '{command}' -> {action['type']}")

        return True

    except Exception as e:
        print(f"✗ NLP processor test failed: {e}")
        traceback.print_exc()
        return False

def test_rl_environment():
    """Test reinforcement learning environment."""
    print("\nTesting RL environment...")

    try:
        from ml.rl_trainer import RobotArmEnv
        from robot_arm.robot_arm import RobotArm

        robot = RobotArm()
        env = RobotArmEnv(robot)
        print("✓ RL environment created")

        # Test environment reset
        obs, info = env.reset()
        print(f"✓ Environment reset, observation shape: {obs.shape}")

        # Test environment step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step, reward: {reward:.3f}")

        return True

    except Exception as e:
        print(f"✗ RL environment test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        from core.config import config

        # Test accessing configuration values
        joint_limits = config.joint_limits
        print(f"✓ Joint limits loaded: {len(joint_limits)} joints")

        link_lengths = config.link_lengths
        print(f"✓ Link lengths loaded: {len(link_lengths)} links")

        window_size = config.window_size
        print(f"✓ Window size: {window_size}")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Robot Arm Simulation - Installation Test")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Core Modules Test", test_core_modules),
        ("Configuration Test", test_configuration),
        ("Robot Arm Creation Test", test_robot_arm_creation),
        ("Physics Engine Test", test_physics_engine),
        ("NLP Processor Test", test_nlp_processor),
        ("RL Environment Test", test_rl_environment),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))

        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\nYou can now run the simulation with:")
        print("  python main.py")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\nTry installing missing dependencies with:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
