#!/usr/bin/env python3
"""
Test script for the Native Desktop GUI

This script tests all components of the native desktop GUI to ensure
they can be imported and initialized properly.
"""

import sys
import os
import traceback

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test importing all GUI components."""
    print("🧪 Testing GUI component imports...")
    
    tests = [
        ("tkinter", "Tkinter GUI framework"),
        ("numpy", "NumPy mathematical library"),
        ("robot_arm.robot_arm", "Robot arm core"),
        ("ml.nlp_processor", "NLP command processor"),
        ("ui.enhanced_control_panel", "Enhanced control panel"),
        ("ui.robot_status_panel", "Robot status panel"),
        ("ui.visualization_window", "3D visualization window"),
        ("native_desktop_gui", "Main GUI application"),
    ]
    
    results = []
    
    for module, description in tests:
        try:
            __import__(module)
            print(f"  ✅ {description}")
            results.append((module, True, None))
        except ImportError as e:
            print(f"  ❌ {description}: {e}")
            results.append((module, False, str(e)))
        except Exception as e:
            print(f"  ⚠️  {description}: {e}")
            results.append((module, False, str(e)))
    
    return results


def test_robot_creation():
    """Test creating a robot arm instance."""
    print("\n🤖 Testing robot arm creation...")
    
    try:
        from robot_arm.robot_arm import RobotArm
        robot = RobotArm()
        print(f"  ✅ Robot arm created with {len(robot.joints)} joints")
        
        # Test basic robot operations
        joint_positions = robot.get_joint_positions()
        print(f"  ✅ Joint positions retrieved: {len(joint_positions)} values")
        
        joint_info = robot.get_joint_info()
        print(f"  ✅ Joint info retrieved for {len(joint_info)} joints")
        
        return True, robot
        
    except Exception as e:
        print(f"  ❌ Robot creation failed: {e}")
        traceback.print_exc()
        return False, None


def test_nlp_processor():
    """Test NLP command processor."""
    print("\n💬 Testing NLP command processor...")
    
    try:
        from ml.nlp_processor import CommandParser
        parser = CommandParser()
        print("  ✅ Command parser created")
        
        # Test parsing a simple command
        test_command = "wave hello"
        parsed = parser.parse_command(test_command)
        print(f"  ✅ Command parsed: '{test_command}' -> {parsed.get('intent', 'unknown')}")
        
        return True, parser
        
    except Exception as e:
        print(f"  ❌ NLP processor failed: {e}")
        traceback.print_exc()
        return False, None


def test_gui_components(robot, parser):
    """Test GUI component creation without showing windows."""
    print("\n🖼️  Testing GUI component creation...")
    
    try:
        import tkinter as tk
        
        # Create a test root window (hidden)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Test enhanced control panel
        try:
            from ui.enhanced_control_panel import EnhancedControlPanel
            control_panel = EnhancedControlPanel(root, robot, parser)
            print("  ✅ Enhanced control panel created")
        except Exception as e:
            print(f"  ❌ Enhanced control panel failed: {e}")
        
        # Test robot status panel
        try:
            from ui.robot_status_panel import RobotStatusPanel
            status_panel = RobotStatusPanel(root, robot)
            print("  ✅ Robot status panel created")
        except Exception as e:
            print(f"  ❌ Robot status panel failed: {e}")
        
        # Test visualization window (without actually creating OpenGL context)
        try:
            from ui.visualization_window import create_visualization_window
            viz_window = create_visualization_window(robot)
            print("  ✅ Visualization window component created")
        except Exception as e:
            print(f"  ❌ Visualization window failed: {e}")
        
        # Clean up
        root.destroy()
        
        return True
        
    except Exception as e:
        print(f"  ❌ GUI component testing failed: {e}")
        traceback.print_exc()
        return False


def test_main_application():
    """Test main application creation (without running)."""
    print("\n🚀 Testing main application creation...")
    
    try:
        from native_desktop_gui import NativeDesktopGUI
        
        # Create application instance (but don't run it)
        app = NativeDesktopGUI()
        print("  ✅ Main application instance created")
        
        # Test robot system initialization
        if app.initialize_robot_system():
            print("  ✅ Robot system initialized")
        else:
            print("  ❌ Robot system initialization failed")
            return False
        
        print("  ✅ Main application ready to run")
        return True
        
    except Exception as e:
        print(f"  ❌ Main application creation failed: {e}")
        traceback.print_exc()
        return False


def check_optional_features():
    """Check availability of optional features."""
    print("\n🔍 Checking optional features...")
    
    features = [
        ("OpenGL.GL", "3D visualization"),
        ("glfw", "3D window management"),
        ("scipy", "Advanced mathematics"),
        ("transformers", "Advanced NLP"),
        ("torch", "Machine learning"),
    ]
    
    available = []
    missing = []
    
    for module, feature in features:
        try:
            __import__(module)
            print(f"  ✅ {feature} available")
            available.append(feature)
        except ImportError:
            print(f"  ⚠️  {feature} not available")
            missing.append(feature)
    
    return available, missing


def main():
    """Main test function."""
    print("=" * 60)
    print("🧪 Native Desktop GUI Test Suite")
    print("=" * 60)
    
    # Test imports
    import_results = test_imports()
    
    # Count successful imports
    successful_imports = sum(1 for _, success, _ in import_results if success)
    total_imports = len(import_results)
    
    print(f"\n📊 Import Results: {successful_imports}/{total_imports} successful")
    
    if successful_imports < total_imports:
        print("\n❌ Some imports failed. Checking what's missing...")
        for module, success, error in import_results:
            if not success:
                print(f"  • {module}: {error}")
    
    # Test robot creation
    robot_ok, robot = test_robot_creation()
    
    # Test NLP processor
    nlp_ok, parser = test_nlp_processor()
    
    # Test GUI components if robot and parser are available
    gui_ok = False
    if robot_ok and nlp_ok:
        gui_ok = test_gui_components(robot, parser)
    else:
        print("\n⚠️  Skipping GUI component tests (robot or NLP not available)")
    
    # Test main application
    app_ok = False
    if robot_ok and nlp_ok:
        app_ok = test_main_application()
    else:
        print("\n⚠️  Skipping main application test (dependencies not available)")
    
    # Check optional features
    available_features, missing_features = check_optional_features()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    print(f"✅ Imports: {successful_imports}/{total_imports}")
    print(f"✅ Robot Creation: {'Yes' if robot_ok else 'No'}")
    print(f"✅ NLP Processor: {'Yes' if nlp_ok else 'No'}")
    print(f"✅ GUI Components: {'Yes' if gui_ok else 'No'}")
    print(f"✅ Main Application: {'Yes' if app_ok else 'No'}")
    
    if available_features:
        print(f"\n🌟 Available Features: {', '.join(available_features)}")
    
    if missing_features:
        print(f"\n⚠️  Missing Features: {', '.join(missing_features)}")
    
    # Overall assessment
    core_ok = robot_ok and nlp_ok and gui_ok and app_ok
    
    if core_ok:
        print("\n🎉 ALL CORE TESTS PASSED!")
        print("   The native desktop GUI should work correctly.")
        print("\n🚀 To launch the application:")
        print("   python run_native_gui.py")
        
        if missing_features:
            print(f"\n💡 To enable missing features, install:")
            for module, feature in [
                ("PyOpenGL glfw", "3D visualization"),
                ("scipy", "Advanced mathematics"),
                ("transformers torch", "Advanced NLP"),
            ]:
                if any(f in missing_features for f in feature.split()):
                    print(f"   pip install {module}")
        
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("   The native desktop GUI may not work correctly.")
        print("\n🔧 Troubleshooting:")
        
        if not robot_ok:
            print("   • Check robot arm dependencies")
        if not nlp_ok:
            print("   • Check NLP processor dependencies")
        if not gui_ok:
            print("   • Check tkinter installation")
        if not app_ok:
            print("   • Check main application dependencies")
        
        print("\n🔄 Alternative options:")
        print("   python web_3d_interface.py    # Web-based interface")
        print("   python main.py --demo         # Command line demo")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
