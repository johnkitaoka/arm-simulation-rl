#!/usr/bin/env python3
"""Install dependencies for the 3D web interface."""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {' '.join(cmd)}")
        return False

def main():
    """Install dependencies for 3D interface."""
    print("🤖 Installing 3D Robot Interface Dependencies")
    print("=" * 50)

    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        print("   Please run this script from the robot-simulation directory")
        return 1

    # Try uv first (preferred for this project)
    print("🔍 Checking for uv package manager...")
    if run_command(["uv", "--version"], "Checking uv"):
        print("✅ Using uv package manager")

        # Install web interface dependencies
        web_deps = [
            "flask>=2.2.0",
            "flask-socketio>=5.3.0",
            "python-socketio>=5.7.0"
        ]

        if run_command(["uv", "pip", "install"] + web_deps, "Installing web interface dependencies"):
            print("✅ Web interface dependencies installed")
        else:
            print("❌ Failed to install web interface dependencies")
            return 1

        # Install other requirements if needed
        if run_command(["uv", "pip", "install", "-r", "requirements.txt"], "Installing other requirements"):
            print("✅ All requirements installed")
        else:
            print("⚠️  Some requirements may have failed to install")
            print("   This is normal for PyBullet on Apple Silicon")
            print("   The 3D interface should still work")

    else:
        print("⚠️  uv not found, falling back to pip")

        # Try regular pip
        web_deps = [
            "flask>=2.2.0",
            "flask-socketio>=5.3.0",
            "python-socketio>=5.7.0"
        ]

        if run_command([sys.executable, "-m", "pip", "install"] + web_deps, "Installing web interface dependencies with pip"):
            print("✅ Web interface dependencies installed")
        else:
            print("❌ Failed to install web interface dependencies")
            return 1

    print("\n" + "=" * 50)
    print("🎉 Installation complete!")
    print("\n🚀 To start the 3D interface:")
    print("   python run_3d_gui.py")
    print("\n💡 Or run directly:")
    print("   python web_3d_interface.py")
    print("\n🌐 The interface will be available at: http://localhost:8080")

    return 0

if __name__ == "__main__":
    sys.exit(main())
