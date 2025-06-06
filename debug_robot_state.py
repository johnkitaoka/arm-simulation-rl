#!/usr/bin/env python3
"""Debug script to check robot state data."""

import requests
import json
import time

def get_robot_state_via_api():
    """Get robot state by triggering a command and checking response."""
    try:
        # First, move a joint to trigger state change
        response = requests.post('http://localhost:8080/api/joint/shoulder_pitch', 
                               json={'position': 0.2}, timeout=5)
        print(f"Joint move response: {response.json()}")
        
        # The robot state is sent via WebSocket, but we can't easily capture that here
        # Instead, let's create a simple endpoint to get the current state
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_forward_kinematics():
    """Test the forward kinematics calculation locally."""
    print("Testing forward kinematics calculation...")
    
    # Simulate the MockFKSolver calculation
    import math
    
    # Mock joint positions
    joints = {
        'shoulder_pitch': 0.2,
        'shoulder_yaw': 0.0,
        'elbow': 0.3,
        'wrist_pitch': 0.0
    }
    
    link_lengths = [0.1, 0.15, 0.12, 0.08]
    
    # Scale down for stability (same as in the code)
    shoulder_pitch = joints['shoulder_pitch'] * 0.5
    shoulder_yaw = joints['shoulder_yaw'] * 0.5
    elbow = joints['elbow'] * 0.5
    wrist_pitch = joints['wrist_pitch'] * 0.3
    
    positions = []
    
    # Start at base
    x, y, z = 0.0, 0.0, 0.0
    
    # Base to shoulder (vertical)
    z += link_lengths[0]
    positions.append([x, y, z])
    print(f"Shoulder position: [{x:.3f}, {y:.3f}, {z:.3f}]")
    
    # Shoulder to elbow
    upper_arm_length = link_lengths[1]
    x += upper_arm_length * math.sin(shoulder_pitch) * math.cos(shoulder_yaw)
    y += upper_arm_length * math.sin(shoulder_pitch) * math.sin(shoulder_yaw)
    z += upper_arm_length * math.cos(shoulder_pitch)
    positions.append([x, y, z])
    print(f"Elbow position: [{x:.3f}, {y:.3f}, {z:.3f}]")
    
    # Elbow to wrist
    forearm_length = link_lengths[2]
    elbow_direction_x = math.sin(shoulder_pitch + elbow) * math.cos(shoulder_yaw)
    elbow_direction_y = math.sin(shoulder_pitch + elbow) * math.sin(shoulder_yaw)
    elbow_direction_z = math.cos(shoulder_pitch + elbow)
    
    x += forearm_length * elbow_direction_x
    y += forearm_length * elbow_direction_y
    z += forearm_length * elbow_direction_z
    positions.append([x, y, z])
    print(f"Wrist position: [{x:.3f}, {y:.3f}, {z:.3f}]")
    
    # Wrist to end effector
    hand_length = link_lengths[3]
    wrist_direction_x = math.sin(shoulder_pitch + elbow + wrist_pitch) * math.cos(shoulder_yaw)
    wrist_direction_y = math.sin(shoulder_pitch + elbow + wrist_pitch) * math.sin(shoulder_yaw)
    wrist_direction_z = math.cos(shoulder_pitch + elbow + wrist_pitch)
    
    x += hand_length * wrist_direction_x
    y += hand_length * wrist_direction_y
    z += hand_length * wrist_direction_z
    positions.append([x, y, z])
    print(f"End effector position: [{x:.3f}, {y:.3f}, {z:.3f}]")
    
    print(f"\nAll positions: {positions}")
    
    # Check if positions are reasonable
    total_reach = sum(link_lengths)
    end_distance = math.sqrt(x*x + y*y + z*z)
    print(f"\nTotal arm reach: {total_reach:.3f}")
    print(f"End effector distance from base: {end_distance:.3f}")
    
    if end_distance <= total_reach:
        print("✅ Positions look reasonable")
    else:
        print("❌ End effector is beyond reach - calculation error")
    
    return positions

def main():
    """Main debug function."""
    print("🔍 Robot State Debugging")
    print("=" * 40)
    
    # Test local forward kinematics
    test_forward_kinematics()
    
    print("\n" + "=" * 40)
    print("🌐 Testing API connection...")
    
    # Test API
    if get_robot_state_via_api():
        print("✅ API connection working")
    else:
        print("❌ API connection failed")
    
    print("\n💡 To see real-time robot state:")
    print("   1. Open browser to http://localhost:8080")
    print("   2. Open browser developer tools (F12)")
    print("   3. Look at Console tab for 'Received robot state' messages")
    print("   4. Move joint sliders and watch the console output")

if __name__ == "__main__":
    main()
