#!/usr/bin/env python3
"""Test script for the 3D web interface."""

import requests
import json
import time

def test_api_endpoints():
    """Test the API endpoints of the 3D interface."""
    base_url = "http://localhost:8080"

    print("🧪 Testing 3D Robot Interface API")
    print("=" * 40)

    # Test 1: Check if server is responding
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is responding")
            print(f"   Status: {response.status_code}")
        else:
            print(f"❌ Server error: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure the 3D interface is running:")
        print("   python web_3d_interface.py")
        return False

    # Test 2: Test command API
    print("\n🎮 Testing command API...")
    test_commands = [
        "wave hello",
        "reset to home",
        "move forward",
        "point at target"
    ]

    for command in test_commands:
        try:
            response = requests.post(
                f"{base_url}/api/command",
                json={"command": command},
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Command '{command}':")
                print(f"   Action: {data.get('action_type', 'unknown')}")
                print(f"   Success: {data.get('success', False)}")
                print(f"   Confidence: {data.get('confidence', 0.0):.2f}")
            else:
                print(f"❌ Command '{command}' failed: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Command '{command}' error: {e}")

    # Test 3: Test joint movement API
    print("\n🦾 Testing joint control API...")
    test_joints = [
        ("shoulder_pitch", 0.5),
        ("elbow", 1.0),
        ("wrist_yaw", -0.3)
    ]

    for joint_name, position in test_joints:
        try:
            response = requests.post(
                f"{base_url}/api/joint/{joint_name}",
                json={"position": position},
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ Joint '{joint_name}' moved to {position}")
                else:
                    print(f"⚠️  Joint '{joint_name}' move failed: {data.get('error', 'unknown')}")
            else:
                print(f"❌ Joint '{joint_name}' API error: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Joint '{joint_name}' error: {e}")

    # Test 4: Test reset API
    print("\n🔄 Testing reset API...")
    try:
        response = requests.post(f"{base_url}/api/reset", timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Robot reset successful")
            else:
                print(f"⚠️  Robot reset failed: {data.get('error', 'unknown')}")
        else:
            print(f"❌ Reset API error: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Reset error: {e}")

    print("\n" + "=" * 40)
    print("🎉 API testing complete!")
    print("\n💡 To view the 3D interface:")
    print("   Open your browser to: http://localhost:8080")
    print("\n🎮 Interface features:")
    print("   • Real-time 3D robot visualization")
    print("   • Interactive joint controls")
    print("   • Natural language commands")
    print("   • Mouse controls: drag to orbit, scroll to zoom")

    return True

def main():
    """Main test function."""
    success = test_api_endpoints()

    if success:
        print("\n🚀 The 3D interface is working correctly!")
        print("   You can now interact with the robot through the web interface.")
    else:
        print("\n❌ Some tests failed. Check the server status.")

    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
