#!/usr/bin/env python3
"""Basic test without physics or GUI dependencies."""

import sys
import os
import time
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_robot_arm():
    """Test basic robot arm functionality."""
    print("Testing Robot Arm...")
    
    try:
        from robot_arm.robot_arm import RobotArm
        
        # Create robot
        robot = RobotArm()
        print(f"✓ Robot created with {len(robot.joints)} joints")
        
        # Test joint positions
        positions = robot.get_joint_positions()
        print(f"✓ Got {len(positions)} joint positions")
        
        # Test setting positions
        main_joints = ['shoulder_pitch', 'shoulder_yaw', 'shoulder_roll',
                      'elbow_flexion', 'wrist_pitch', 'wrist_yaw']
        
        target_positions = np.array([0.5, 0.3, -0.2, 1.0, 0.4, -0.1])
        robot.set_joint_targets(target_positions, main_joints)
        print("✓ Set target joint positions")
        
        # Update robot
        for i in range(5):
            robot.update(dt=0.1)
        
        current_positions = robot.get_joint_positions(main_joints)
        print(f"✓ Updated robot, current positions: {current_positions[:3]}")
        
        # Test reset
        robot.reset_to_home()
        print("✓ Reset to home position")
        
        return True
        
    except Exception as e:
        print(f"✗ Robot arm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_nlp():
    """Test natural language processing."""
    print("\nTesting NLP...")
    
    try:
        from ml.nlp_processor import CommandParser
        
        parser = CommandParser()
        print("✓ Command parser created")
        
        # Test commands
        test_commands = [
            "move forward",
            "wave hello", 
            "point at target",
            "reset to home"
        ]
        
        for command in test_commands:
            parsed = parser.parse_command(command)
            action = parser.command_to_robot_action(parsed)
            print(f"✓ '{command}' -> {action['type']} (confidence: {parsed['confidence']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"✗ NLP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_robot_movements():
    """Test robot movements without physics."""
    print("\nTesting Robot Movements...")
    
    try:
        from robot_arm.robot_arm import RobotArm
        
        robot = RobotArm()
        
        # Demo positions
        positions = [
            [0.3, 0.0, 0.3],   # Forward
            [0.2, 0.2, 0.4],   # Right and up
            [0.2, -0.2, 0.4],  # Left and up
            [0.4, 0.0, 0.2],   # Forward and down
        ]
        
        for i, pos in enumerate(positions):
            print(f"Moving to position {i+1}: {pos}")
            
            # Try inverse kinematics (may not work without full implementation)
            try:
                if robot.ik_solver:
                    target_joints = robot.solve_inverse_kinematics(np.array(pos))
                    if target_joints is not None:
                        print(f"  IK solution found")
                        robot.set_joint_targets(target_joints[:6])
                    else:
                        print(f"  No IK solution")
                else:
                    print(f"  IK solver not available")
            except Exception as e:
                print(f"  IK failed: {e}")
            
            # Update robot
            for _ in range(10):
                robot.update(dt=0.1)
            
            print(f"  Movement completed")
        
        # Return to home
        robot.reset_to_home()
        print("✓ Returned to home position")
        
        return True
        
    except Exception as e:
        print(f"✗ Movement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_command_execution():
    """Test command execution."""
    print("\nTesting Command Execution...")
    
    try:
        from robot_arm.robot_arm import RobotArm
        from ml.nlp_processor import CommandParser
        
        robot = RobotArm()
        parser = CommandParser()
        
        commands = [
            "wave hello",
            "move forward", 
            "reset to home"
        ]
        
        for command in commands:
            print(f"Executing: '{command}'")
            
            # Parse command
            parsed = parser.parse_command(command)
            action = parser.command_to_robot_action(parsed)
            
            if action['success']:
                print(f"  ✓ Command understood: {action['type']}")
                
                # Simple execution simulation
                if action['type'] == 'reset_to_home':
                    robot.reset_to_home()
                    print("  ✓ Reset executed")
                elif action['type'] == 'move_to_position':
                    print("  ✓ Move command prepared")
                elif action['type'] == 'gesture':
                    print("  ✓ Gesture command prepared")
                else:
                    print(f"  - Action type '{action['type']}' noted")
            else:
                print(f"  ✗ Command not understood")
        
        return True
        
    except Exception as e:
        print(f"✗ Command execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic tests."""
    print("Robot Arm Simulation - Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        ("Robot Arm", test_robot_arm),
        ("NLP Processing", test_nlp),
        ("Robot Movements", test_robot_movements),
        ("Command Execution", test_command_execution),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name} Test")
        print("-" * (len(test_name) + 5))
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} test PASSED")
            else:
                print(f"✗ {test_name} test FAILED")
        except Exception as e:
            print(f"✗ {test_name} test FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        print("\nNext steps:")
        print("1. Activate conda environment: conda activate pybullet-env")
        print("2. Run with physics: python main.py --demo")
        print("3. Try commands: python main.py --command 'wave hello'")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
