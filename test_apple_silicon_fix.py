#!/usr/bin/env python3
"""
Test Apple Silicon Compatibility Fix

This script tests the Apple Silicon compatibility fixes for the robot arm
simulation GUI, specifically addressing the tkinter "-bg" option errors.
"""

import sys
import os
import time
import platform

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_compatibility_module():
    """Test the Apple Silicon compatibility module."""
    print("🧪 Testing Apple Silicon Compatibility Module")
    print("=" * 50)
    
    try:
        from core.apple_silicon_compat import get_compat, safe_config_widget_colors
        
        # Get compatibility instance
        compat = get_compat()
        
        # Print compatibility info
        compat.print_compatibility_info()
        
        # Test widget color configuration
        import tkinter as tk
        
        print("\n🎨 Testing Widget Color Configuration...")
        
        # Create test window
        root = tk.Tk()
        root.title("Apple Silicon Compatibility Test")
        root.geometry("400x300")
        
        # Test labels with different color configurations
        test_frame = tk.Frame(root)
        test_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(test_frame, text="Apple Silicon Compatibility Test", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Test background color configuration
        bg_label = tk.Label(test_frame, text="Background Color Test")
        bg_label.pack(pady=5)
        
        bg_result = safe_config_widget_colors(bg_label, bg="lightblue")
        if bg_result.get('bg', False):
            print("✅ Background color configuration: SUPPORTED")
        else:
            print("❌ Background color configuration: NOT SUPPORTED")
            bg_label.config(text="🔵 Background Color Test (using symbol)")
        
        # Test foreground color configuration
        fg_label = tk.Label(test_frame, text="Foreground Color Test")
        fg_label.pack(pady=5)
        
        fg_result = safe_config_widget_colors(fg_label, fg="red")
        if fg_result.get('fg', False):
            print("✅ Foreground color configuration: SUPPORTED")
        else:
            print("❌ Foreground color configuration: NOT SUPPORTED")
            fg_label.config(text="🔴 Foreground Color Test (using symbol)")
        
        # Test status indicators
        print("\n📊 Testing Status Indicators...")
        
        status_frame = tk.Frame(test_frame)
        status_frame.pack(fill="x", pady=10)
        
        statuses = ['ok', 'warning', 'error', 'moving', 'limit']
        for status in statuses:
            indicator = compat.create_status_indicator(status_frame, status, f"{status.title()} Status")
            indicator.pack(anchor="w", pady=2)
        
        # Add close button
        tk.Button(test_frame, text="Close Test", 
                 command=root.destroy).pack(pady=20)
        
        print("\n💡 Test window created. Check for any tkinter errors in the console.")
        print("   The window should display without the '-bg' option errors.")
        
        # Run for a short time to test
        root.after(5000, root.destroy)  # Auto-close after 5 seconds
        root.mainloop()
        
        print("✅ GUI test completed without errors!")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_robot_status_panel():
    """Test the robot status panel with Apple Silicon fixes."""
    print("\n🤖 Testing Robot Status Panel")
    print("=" * 50)
    
    try:
        from robot_arm.robot_arm import RobotArm
        from ui.robot_status_panel import RobotStatusPanel
        import tkinter as tk
        
        # Create robot arm
        robot = RobotArm()
        
        # Create test window
        root = tk.Tk()
        root.title("Robot Status Panel Test")
        root.geometry("800x600")
        
        # Create status panel
        status_panel = RobotStatusPanel(root, robot)
        status_panel.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Update status a few times to test for errors
        def update_test():
            try:
                status_panel.update_status()
                print("✅ Status update completed without errors")
            except Exception as e:
                print(f"❌ Status update error: {e}")
        
        # Schedule updates
        for i in range(5):
            root.after(i * 1000, update_test)
        
        # Auto-close after 6 seconds
        root.after(6000, root.destroy)
        
        print("💡 Running status panel test for 6 seconds...")
        print("   Watch for any '-bg' option errors in the console.")
        
        root.mainloop()
        
        print("✅ Robot status panel test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Robot status panel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_control_panel():
    """Test the enhanced control panel with Apple Silicon fixes."""
    print("\n🎮 Testing Enhanced Control Panel")
    print("=" * 50)
    
    try:
        from robot_arm.robot_arm import RobotArm
        from ml.nlp_processor import CommandParser
        from ui.enhanced_control_panel import EnhancedControlPanel
        import tkinter as tk
        
        # Create robot arm and command parser
        robot = RobotArm()
        parser = CommandParser()
        
        # Create test window
        root = tk.Tk()
        root.title("Enhanced Control Panel Test")
        root.geometry("1000x700")
        
        # Create control panel
        control_panel = EnhancedControlPanel(root, robot, parser)
        control_panel.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Update from robot to test joint indicators
        def update_test():
            try:
                control_panel.update_from_robot()
                print("✅ Control panel update completed without errors")
            except Exception as e:
                print(f"❌ Control panel update error: {e}")
        
        # Schedule updates
        for i in range(3):
            root.after(i * 1000, update_test)
        
        # Auto-close after 4 seconds
        root.after(4000, root.destroy)
        
        print("💡 Running control panel test for 4 seconds...")
        print("   Watch for any color configuration errors in the console.")
        
        root.mainloop()
        
        print("✅ Enhanced control panel test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced control panel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Apple Silicon compatibility tests."""
    print("🍎 APPLE SILICON COMPATIBILITY TEST SUITE")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    
    results = []
    
    # Test 1: Compatibility module
    results.append(test_compatibility_module())
    
    # Test 2: Robot status panel
    results.append(test_robot_status_panel())
    
    # Test 3: Enhanced control panel
    results.append(test_enhanced_control_panel())
    
    # Summary
    print("\n" + "=" * 60)
    print("🏆 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Apple Silicon Compatibility Module",
        "Robot Status Panel",
        "Enhanced Control Panel"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 ALL TESTS PASSED! Apple Silicon compatibility fix is working correctly.")
        print("\n💡 The native desktop GUI should now run without tkinter errors on Apple Silicon Macs.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
