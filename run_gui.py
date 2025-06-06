#!/usr/bin/env python3
"""Safe GUI launcher for robot arm simulation."""

import sys
import os
import platform

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_gui_compatibility():
    """Check if GUI can run on this system."""
    print("🔍 Checking GUI compatibility...")
    
    # Check platform
    system = platform.system()
    print(f"Platform: {system}")
    
    if system == "Darwin":
        print("⚠️  macOS detected - GUI may have compatibility issues")
        
        # Check macOS version
        try:
            import subprocess
            result = subprocess.run(['sw_vers', '-productVersion'], 
                                  capture_output=True, text=True)
            macos_version = result.stdout.strip()
            print(f"macOS version: {macos_version}")
            
            # Parse version
            major_version = int(macos_version.split('.')[0])
            if major_version >= 12:  # macOS Monterey and later
                print("⚠️  macOS 12+ detected - Tkinter may crash")
                return False
        except:
            print("⚠️  Could not determine macOS version")
    
    # Test Tkinter import
    try:
        import tkinter as tk
        print("✅ Tkinter import successful")
        
        # Test basic Tkinter functionality
        root = tk.Tk()
        root.withdraw()  # Hide the window
        root.destroy()
        print("✅ Tkinter basic test passed")
        return True
        
    except Exception as e:
        print(f"❌ Tkinter test failed: {e}")
        return False

def run_gui_safe():
    """Run GUI with safety checks."""
    print("🤖 Robot Arm Simulation - GUI Launcher")
    print("=" * 50)
    
    # Check compatibility
    if not check_gui_compatibility():
        print("\n❌ GUI compatibility check failed!")
        print("\n🔧 Alternative options:")
        print("1. Force-enable GUI (may crash):")
        print("   FORCE_GUI=true python main.py")
        print("\n2. Use command line interface:")
        print("   python main.py --demo")
        print("   python main.py --command 'wave hello'")
        print("\n3. Use web interface (if available):")
        print("   python web_interface.py")
        
        # Ask user if they want to proceed anyway
        try:
            response = input("\nDo you want to try running the GUI anyway? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                return False
        except KeyboardInterrupt:
            print("\nExiting...")
            return False
    
    print("\n🚀 Starting GUI...")
    
    try:
        from robot_arm.robot_arm import RobotArm
        from ui.control_panel import ControlPanel
        
        # Create robot arm
        print("Creating robot arm...")
        robot = RobotArm()
        
        # Create and run GUI
        print("Launching control panel...")
        control_panel = ControlPanel(robot)
        control_panel.run()
        
        return True
        
    except Exception as e:
        print(f"\n❌ GUI failed to start: {e}")
        print("\n🔧 Try these alternatives:")
        print("1. Command line demo: python main.py --demo")
        print("2. Natural language commands: python main.py --command 'wave hello'")
        print("3. Basic test: python test_basic.py")
        
        import traceback
        print(f"\nFull error details:")
        traceback.print_exc()
        
        return False

def main():
    """Main entry point."""
    success = run_gui_safe()
    
    if not success:
        print("\n" + "=" * 50)
        print("GUI could not be started. Using command line interface:")
        print("\nAvailable commands:")
        print("• python main.py --demo                    # Movement demonstration")
        print("• python main.py --command 'wave hello'    # Natural language command")
        print("• python main.py --train PPO               # Train ML model")
        print("• python test_basic.py                     # Test basic functionality")
        
        # Offer to run demo instead
        try:
            response = input("\nWould you like to run the demo instead? (Y/n): ")
            if response.lower() not in ['n', 'no']:
                print("\n🚀 Running demo...")
                os.system("python main.py --demo")
        except KeyboardInterrupt:
            print("\nExiting...")
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
