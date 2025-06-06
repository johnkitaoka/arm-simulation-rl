#!/usr/bin/env python3
"""Demo script for the 3D robot interface - shows automated movements."""

import requests
import time
import json

def send_command(command, base_url="http://localhost:8080"):
    """Send a command to the robot interface."""
    try:
        response = requests.post(
            f"{base_url}/api/command",
            json={"command": command},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ '{command}' -> {data.get('action_type', 'unknown')} (success: {data.get('success', False)})")
            return data.get('success', False)
        else:
            print(f"❌ Command failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return False

def move_joint(joint_name, position, base_url="http://localhost:8080"):
    """Move a specific joint to a position."""
    try:
        response = requests.post(
            f"{base_url}/api/joint/{joint_name}",
            json={"position": position},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Moved {joint_name} to {position:.2f}")
                return True
            else:
                print(f"⚠️  Failed to move {joint_name}: {data.get('error', 'unknown')}")
                return False
        else:
            print(f"❌ Joint move failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return False

def demo_sequence():
    """Run a demonstration sequence."""
    print("🤖 3D Robot Interface Demo")
    print("=" * 40)
    print("🌐 Make sure the 3D interface is open in your browser:")
    print("   http://localhost:8080")
    print("\n🎬 Starting demonstration sequence...")
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding. Please start the 3D interface first:")
            print("   python web_3d_interface.py")
            return False
    except:
        print("❌ Cannot connect to server. Please start the 3D interface first:")
        print("   python web_3d_interface.py")
        return False
    
    print("✅ Server is running")
    
    # Demo sequence
    demos = [
        ("🏠 Resetting to home position", lambda: send_command("reset to home")),
        ("⏱️  Waiting 2 seconds", lambda: time.sleep(2)),
        
        ("🦾 Moving shoulder joints", lambda: move_joint("shoulder_pitch", 0.5)),
        ("⏱️  Waiting 1 second", lambda: time.sleep(1)),
        ("🦾 Moving shoulder yaw", lambda: move_joint("shoulder_yaw", 0.8)),
        ("⏱️  Waiting 1 second", lambda: time.sleep(1)),
        
        ("💪 Flexing elbow", lambda: move_joint("elbow", 1.2)),
        ("⏱️  Waiting 1 second", lambda: time.sleep(1)),
        
        ("🤚 Moving wrist", lambda: move_joint("wrist_pitch", -0.5)),
        ("⏱️  Waiting 1 second", lambda: time.sleep(1)),
        ("🤚 Rotating wrist", lambda: move_joint("wrist_yaw", 1.0)),
        ("⏱️  Waiting 2 seconds", lambda: time.sleep(2)),
        
        ("👋 Wave gesture", lambda: send_command("wave hello")),
        ("⏱️  Waiting 3 seconds", lambda: time.sleep(3)),
        
        ("🏠 Returning home", lambda: send_command("reset to home")),
        ("⏱️  Waiting 2 seconds", lambda: time.sleep(2)),
        
        ("🎯 Pointing gesture", lambda: send_command("point at target")),
        ("⏱️  Waiting 2 seconds", lambda: time.sleep(2)),
        
        ("🏠 Final reset", lambda: send_command("reset to home")),
    ]
    
    print(f"\n🎭 Running {len(demos)} demo steps...")
    print("👀 Watch the 3D visualization in your browser!")
    print()
    
    for i, (description, action) in enumerate(demos, 1):
        print(f"[{i:2d}/{len(demos)}] {description}")
        try:
            action()
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Small delay between actions for better visualization
        if not description.startswith("⏱️"):
            time.sleep(0.5)
    
    print("\n🎉 Demo sequence complete!")
    print("\n💡 Try these commands in the web interface:")
    print("   • 'wave hello'")
    print("   • 'reset to home'") 
    print("   • 'move forward'")
    print("   • Use the joint sliders to control individual joints")
    print("   • Drag with mouse to orbit the camera")
    print("   • Scroll to zoom in/out")
    
    return True

def interactive_mode():
    """Interactive command mode."""
    print("\n🎮 Interactive Mode")
    print("=" * 20)
    print("Enter commands to send to the robot (or 'quit' to exit):")
    print("Examples: 'wave hello', 'reset to home', 'move forward'")
    print()
    
    while True:
        try:
            command = input("🤖 Command: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not command:
                continue
                
            send_command(command)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demo function."""
    print("🚀 3D Robot Interface Demo Launcher")
    print("=" * 50)
    
    try:
        choice = input("Choose mode:\n1. Automated demo\n2. Interactive commands\n\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            demo_sequence()
        elif choice == "2":
            interactive_mode()
        else:
            print("Running automated demo...")
            demo_sequence()
            
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
