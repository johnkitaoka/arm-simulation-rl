#!/usr/bin/env python3
"""
Launcher for the Native Desktop GUI

This script provides a safe launcher for the native desktop GUI with:
- System compatibility checks
- Dependency verification
- Fallback options if GUI cannot start
- User-friendly error messages and alternatives
"""

import sys
import os
import platform
import subprocess

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   This application requires Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_required_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking required dependencies...")
    
    required_deps = [
        ('numpy', 'NumPy'),
        ('tkinter', 'Tkinter'),
    ]
    
    missing_deps = []
    
    for module, name in required_deps:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - REQUIRED")
            missing_deps.append(name)
    
    return len(missing_deps) == 0, missing_deps


def check_optional_dependencies():
    """Check optional dependencies for enhanced features."""
    print("🔍 Checking optional dependencies...")
    
    optional_deps = [
        ('OpenGL.GL', 'PyOpenGL', '3D visualization'),
        ('glfw', 'GLFW', '3D window management'),
        ('scipy', 'SciPy', 'Advanced mathematics'),
        ('transformers', 'Transformers', 'Natural language processing'),
    ]
    
    available_features = []
    missing_features = []
    
    for module, name, feature in optional_deps:
        try:
            __import__(module)
            print(f"  ✅ {name} - {feature} available")
            available_features.append(feature)
        except ImportError:
            print(f"  ⚠️  {name} - {feature} disabled")
            missing_features.append((name, feature))
    
    return available_features, missing_features


def check_system_compatibility():
    """Check system-specific compatibility issues."""
    print("🖥️  Checking system compatibility...")
    
    system = platform.system()
    print(f"  Platform: {system}")
    
    warnings = []
    
    if system == "Darwin":  # macOS
        print("  🍎 macOS detected")
        
        try:
            # Check macOS version
            result = subprocess.run(['sw_vers', '-productVersion'], 
                                  capture_output=True, text=True)
            macos_version = result.stdout.strip()
            major_version = int(macos_version.split('.')[0])
            
            print(f"  macOS version: {macos_version}")
            
            if major_version >= 12:
                warnings.append("macOS 12+ may have tkinter compatibility issues")
            
            # Check for Apple Silicon
            result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
            arch = result.stdout.strip()
            
            if arch == "arm64":
                print("  🔧 Apple Silicon detected")
                warnings.append("Some OpenGL features may not work on Apple Silicon")
            
        except Exception as e:
            warnings.append(f"Could not determine macOS details: {e}")
    
    elif system == "Linux":
        print("  🐧 Linux detected")
        # Check for display server
        if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
            warnings.append("No display server detected - GUI may not work")
    
    elif system == "Windows":
        print("  🪟 Windows detected")
        # Windows generally has good tkinter support
    
    if warnings:
        print("  ⚠️  Compatibility warnings:")
        for warning in warnings:
            print(f"     • {warning}")
    else:
        print("  ✅ No compatibility issues detected")
    
    return warnings


def install_missing_dependencies(missing_deps):
    """Attempt to install missing dependencies."""
    if not missing_deps:
        return True
    
    print(f"\n📦 Attempting to install missing dependencies: {', '.join(missing_deps)}")
    
    # Try different package managers
    package_managers = [
        (['uv', 'pip', 'install'], 'uv'),
        ([sys.executable, '-m', 'pip', 'install'], 'pip'),
    ]
    
    for cmd_base, manager_name in package_managers:
        try:
            # Test if package manager is available
            test_cmd = cmd_base[:-1] + ['--version']
            subprocess.run(test_cmd, check=True, capture_output=True)
            
            print(f"  Using {manager_name}...")
            
            # Install dependencies
            install_cmd = cmd_base + missing_deps
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            
            print(f"  ✅ Dependencies installed successfully with {manager_name}")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  ❌ {manager_name} not available or failed")
            continue
    
    print("  ❌ Could not install dependencies automatically")
    return False


def show_installation_help():
    """Show manual installation instructions."""
    print("\n" + "=" * 60)
    print("📋 MANUAL INSTALLATION INSTRUCTIONS")
    print("=" * 60)
    
    print("\n1. Install required dependencies:")
    print("   pip install numpy")
    print("   # tkinter is usually included with Python")
    
    print("\n2. Install optional dependencies for full features:")
    print("   pip install PyOpenGL PyOpenGL-accelerate glfw")
    print("   pip install scipy transformers torch")
    
    print("\n3. For Apple Silicon Macs:")
    print("   conda install -c conda-forge pybullet")
    print("   # Some OpenGL features may not work")
    
    print("\n4. Alternative interfaces if GUI fails:")
    print("   python web_3d_interface.py    # Web-based 3D interface")
    print("   python main.py --demo         # Command line demo")
    print("   python main.py --command 'wave hello'  # Single commands")


def main():
    """Main launcher function."""
    print("=" * 60)
    print("🤖 Robot Arm Simulation - Native Desktop GUI Launcher")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check system compatibility
    warnings = check_system_compatibility()
    
    # Check dependencies
    deps_ok, missing_deps = check_required_dependencies()
    available_features, missing_features = check_optional_dependencies()
    
    # Handle missing required dependencies
    if not deps_ok:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_deps)}")
        
        if input("\nAttempt automatic installation? (Y/n): ").lower() not in ['n', 'no']:
            if install_missing_dependencies(missing_deps):
                print("✅ Dependencies installed - please restart the launcher")
                return 0
            else:
                show_installation_help()
                return 1
        else:
            show_installation_help()
            return 1
    
    # Show feature availability
    if available_features:
        print(f"\n✅ Available features: {', '.join(available_features)}")
    
    if missing_features:
        print("\n⚠️  Disabled features:")
        for name, feature in missing_features:
            print(f"   • {feature} (install {name})")
    
    # Show warnings if any
    if warnings:
        print("\n⚠️  System warnings:")
        for warning in warnings:
            print(f"   • {warning}")
        
        if input("\nContinue anyway? (Y/n): ").lower() in ['n', 'no']:
            print("\n🔧 Alternative options:")
            print("1. Web interface: python web_3d_interface.py")
            print("2. Command line: python main.py --demo")
            return 0
    
    # Launch the native GUI
    print("\n🚀 Launching Native Desktop GUI...")
    
    try:
        from native_desktop_gui import main as gui_main
        return gui_main()
        
    except ImportError as e:
        print(f"❌ Failed to import GUI application: {e}")
        print("\n🔧 Please ensure all files are present:")
        print("   • native_desktop_gui.py")
        print("   • ui/enhanced_control_panel.py")
        print("   • ui/robot_status_panel.py")
        print("   • ui/visualization_window.py")
        return 1
        
    except Exception as e:
        print(f"❌ Failed to start GUI application: {e}")
        print("\n🔧 Alternative options:")
        print("1. Web interface: python web_3d_interface.py")
        print("2. Command line: python main.py --demo")
        print("3. Basic test: python test_basic.py")
        
        import traceback
        print(f"\nFull error details:")
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
