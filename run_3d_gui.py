#!/usr/bin/env python3
"""Launcher for the 3D web-based robot arm interface."""

import sys
import os
import subprocess
import webbrowser
import time
import platform

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")

    missing_deps = []

    try:
        import flask
        print("✅ Flask found")
    except ImportError:
        missing_deps.append("flask")

    try:
        import flask_socketio
        print("✅ Flask-SocketIO found")
    except ImportError:
        missing_deps.append("flask-socketio")

    try:
        import numpy
        print("✅ NumPy found")
    except ImportError:
        missing_deps.append("numpy")

    # Check if robot simulation components are available
    try:
        from robot_arm.robot_arm import RobotArm
        print("✅ Robot arm simulation found")
    except ImportError as e:
        print(f"⚠️  Robot arm simulation not fully available: {e}")
        print("   This may be due to missing PyBullet or other physics dependencies")
        print("   The 3D interface will still work but with limited functionality")

    try:
        from ml.nlp_processor import CommandParser
        print("✅ NLP command processor found")
    except ImportError as e:
        print(f"⚠️  NLP processor not fully available: {e}")
        print("   Natural language commands may not work")

    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("\n🔧 To install missing dependencies:")
        print(f"   uv pip install {' '.join(missing_deps)}")
        return False

    print("✅ All core dependencies found")
    return True

def install_dependencies():
    """Install missing dependencies using uv."""
    print("📦 Installing dependencies...")

    try:
        # Check if uv is available
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv package manager found")

        # Install dependencies
        deps = ["flask>=2.2.0", "flask-socketio>=5.3.0", "python-socketio>=5.7.0"]
        cmd = ["uv", "pip", "install"] + deps

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("📝 Please install manually:")
        print("   uv pip install flask flask-socketio python-socketio")
        return False
    except FileNotFoundError:
        print("❌ uv package manager not found")
        print("📝 Please install dependencies manually:")
        print("   pip install flask flask-socketio python-socketio")
        return False

def check_port_available(port=8080):
    """Check if the port is available."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def main():
    """Main launcher function."""
    print("🤖 Robot Arm 3D Visualization Launcher")
    print("=" * 50)

    # Check system
    system = platform.system()
    print(f"Platform: {system}")

    if system == "Darwin":
        print("✅ macOS detected - Web-based 3D interface is optimal for this platform")

    # Check dependencies
    if not check_dependencies():
        print("\n🔧 Attempting to install missing dependencies...")
        if not install_dependencies():
            print("\n❌ Could not install dependencies automatically")
            print("Please install them manually and try again")
            return 1

        # Re-check after installation
        print("\n🔍 Re-checking dependencies...")
        if not check_dependencies():
            print("❌ Dependencies still missing after installation")
            return 1

    # Check port availability
    port = 8080
    if not check_port_available(port):
        print(f"⚠️  Port {port} is already in use")
        print("   The interface may not start properly")
        print("   Please close any other applications using this port")

        response = input("Continue anyway? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            return 1

    print(f"\n🚀 Starting 3D Robot Arm Interface on port {port}...")
    print(f"🌐 Interface will be available at: http://localhost:{port}")
    print("📱 The interface will open automatically in your browser")
    print("\n💡 Features:")
    print("   • Real-time 3D robot visualization")
    print("   • Interactive joint controls")
    print("   • Natural language commands")
    print("   • Mouse controls: drag to orbit, scroll to zoom")
    print("\n⌨️  Press Ctrl+C to stop the interface")
    print("=" * 50)

    try:
        # Import and run the 3D interface
        from web_3d_interface import main as run_3d_interface
        run_3d_interface()

    except KeyboardInterrupt:
        print("\n\n🛑 Interface stopped by user")
        return 0
    except ImportError as e:
        print(f"\n❌ Failed to import 3D interface: {e}")
        print("   Make sure all files are in place and dependencies are installed")
        return 1
    except Exception as e:
        print(f"\n❌ Error starting interface: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check that all dependencies are installed")
        print("   2. Make sure port 5000 is available")
        print("   3. Try running: python web_3d_interface.py directly")
        return 1

if __name__ == "__main__":
    sys.exit(main())
