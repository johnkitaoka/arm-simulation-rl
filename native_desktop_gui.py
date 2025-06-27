#!/usr/bin/env python3
"""
Native Python Desktop GUI for Robot Arm Simulation

This application provides a comprehensive native desktop interface that replaces
the web-based GUI while maintaining all functionality including:
- Real-time 3D anthropomorphic arm visualization
- Interactive joint controls with real-time feedback
- Natural language command processing
- Comprehensive status monitoring and logging
- Cross-platform compatibility with fallback options
"""

import sys
import os
import platform
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import Optional, Dict, Any, List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import robot arm components
from robot_arm.robot_arm import RobotArm
from ml.nlp_processor import CommandParser
from core.config import config

# Import GUI components (will be created)
from ui.enhanced_control_panel import EnhancedControlPanel
from ui.robot_status_panel import RobotStatusPanel

# Import visualization window with fallback
try:
    from ui.visualization_window import VisualizationWindow
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  3D visualization not available: {e}")
    VisualizationWindow = None
    VISUALIZATION_AVAILABLE = False


class NativeDesktopGUI:
    """Main native desktop GUI application for robot arm simulation."""
    
    def __init__(self):
        """Initialize the native desktop GUI application."""
        self.robot_arm = None
        self.command_parser = None
        self.main_window = None
        self.visualization_window = None
        self.control_panel = None
        self.status_panel = None
        
        # Threading and update control
        self.update_thread = None
        self.running = False
        self.update_rate = 30  # Hz
        
        # GUI state
        self.gui_initialized = False
        self.visualization_enabled = True
        
        print("🤖 Initializing Native Desktop GUI for Robot Arm Simulation")
        self._check_system_compatibility()
        
    def _check_system_compatibility(self):
        """Check system compatibility and set appropriate options."""
        system = platform.system()
        print(f"Platform: {system}")
        
        if system == "Darwin":  # macOS
            print("⚠️  macOS detected - checking compatibility...")
            try:
                # Check macOS version for tkinter compatibility
                import subprocess
                result = subprocess.run(['sw_vers', '-productVersion'], 
                                      capture_output=True, text=True)
                macos_version = result.stdout.strip()
                major_version = int(macos_version.split('.')[0])
                
                if major_version >= 12:
                    print("⚠️  macOS 12+ detected - may have GUI compatibility issues")
                    print("   Will attempt to use compatibility mode")
                    
            except Exception as e:
                print(f"⚠️  Could not determine macOS version: {e}")
        
        # Check OpenGL availability for 3D visualization
        try:
            import OpenGL.GL
            import glfw
            print("✅ OpenGL and GLFW available - 3D visualization enabled")
            self.visualization_enabled = True
        except ImportError as e:
            print(f"⚠️  OpenGL/GLFW not available: {e}")
            print("   3D visualization will be disabled, using 2D fallback")
            self.visualization_enabled = False
    
    def initialize_robot_system(self):
        """Initialize the robot arm and related systems."""
        print("🔧 Initializing robot system...")
        
        try:
            # Create robot arm
            print("  Creating robot arm...")
            self.robot_arm = RobotArm()
            print(f"  ✅ Robot arm created with {len(self.robot_arm.joints)} joints")
            
            # Initialize command parser
            print("  Loading NLP command parser...")
            self.command_parser = CommandParser()
            print("  ✅ Command parser loaded")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to initialize robot system: {e}")
            messagebox.showerror("Initialization Error", 
                               f"Failed to initialize robot system:\n{str(e)}")
            return False
    
    def create_main_window(self):
        """Create the main application window."""
        print("🖼️  Creating main application window...")
        
        self.main_window = tk.Tk()
        self.main_window.title("Robot Arm Desktop Control - Native GUI")
        self.main_window.geometry("1200x800")
        
        # Set window icon and properties
        self.main_window.resizable(True, True)
        self.main_window.minsize(800, 600)
        
        # Handle window close event
        self.main_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create main layout
        self._create_main_layout()
        
        print("  ✅ Main window created")
    
    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.main_window)
        self.main_window.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Robot State", command=self.save_robot_state)
        file_menu.add_command(label="Load Robot State", command=self.load_robot_state)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Robot menu
        robot_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Robot", menu=robot_menu)
        robot_menu.add_command(label="Reset to Home", command=self.reset_robot)
        robot_menu.add_command(label="Emergency Stop", command=self.emergency_stop)
        robot_menu.add_separator()
        robot_menu.add_command(label="Enable Robot", command=self.enable_robot)
        robot_menu.add_command(label="Disable Robot", command=self.disable_robot)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show 3D Visualization", command=self.show_3d_visualization)
        view_menu.add_command(label="Refresh All", command=self.refresh_all_displays)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Controls Help", command=self.show_controls_help)
    
    def _create_main_layout(self):
        """Create the main window layout."""
        # Create main paned window for resizable layout
        main_paned = ttk.PanedWindow(self.main_window, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel for status and information
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # Create control panel in left frame
        self.control_panel = EnhancedControlPanel(left_frame, self.robot_arm, self.command_parser)
        self.control_panel.pack(fill=tk.BOTH, expand=True)
        
        # Create status panel in right frame
        self.status_panel = RobotStatusPanel(right_frame, self.robot_arm)
        self.status_panel.pack(fill=tk.BOTH, expand=True)
    
    def create_3d_visualization(self):
        """Create the 3D visualization window if enabled."""
        if not self.visualization_enabled or not VISUALIZATION_AVAILABLE:
            print("⚠️  3D visualization disabled - skipping")
            return

        print("🎮 Creating 3D visualization window...")

        try:
            self.visualization_window = VisualizationWindow(self.robot_arm)
            print("  ✅ 3D visualization window created")
        except Exception as e:
            print(f"  ❌ Failed to create 3D visualization: {e}")
            self.visualization_enabled = False
            messagebox.showwarning("3D Visualization",
                                 f"3D visualization could not be initialized:\n{str(e)}\n\n"
                                 "The application will continue with 2D controls only.")
    
    def start_update_system(self):
        """Start the real-time update system."""
        print("🔄 Starting real-time update system...")
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        print(f"  ✅ Update system started at {self.update_rate} Hz")
    
    def _update_loop(self):
        """Main update loop running in separate thread."""
        dt = 1.0 / self.update_rate
        
        while self.running:
            try:
                start_time = time.time()
                
                # Update robot arm
                if self.robot_arm:
                    self.robot_arm.update(dt)
                
                # Update GUI components (thread-safe)
                if self.gui_initialized:
                    self.main_window.after_idle(self._update_gui_components)
                
                # Update 3D visualization
                if self.visualization_window and self.visualization_enabled:
                    self.visualization_window.update_visualization()
                
                # Maintain update rate
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"❌ Error in update loop: {e}")
                time.sleep(0.1)  # Prevent rapid error loops
    
    def _update_gui_components(self):
        """Update GUI components (called from main thread)."""
        try:
            if self.control_panel:
                self.control_panel.update_from_robot()
            
            if self.status_panel:
                self.status_panel.update_status()
                
        except Exception as e:
            print(f"❌ Error updating GUI components: {e}")
    
    # Menu command implementations
    def save_robot_state(self):
        """Save current robot state to file."""
        # Implementation will be added in enhanced control panel
        if self.control_panel:
            self.control_panel.save_robot_state()
    
    def load_robot_state(self):
        """Load robot state from file."""
        if self.control_panel:
            self.control_panel.load_robot_state()
    
    def reset_robot(self):
        """Reset robot to home position."""
        if self.robot_arm:
            self.robot_arm.reset_to_home()
    
    def emergency_stop(self):
        """Emergency stop the robot."""
        if self.robot_arm:
            self.robot_arm.emergency_stop()
    
    def enable_robot(self):
        """Enable the robot."""
        if self.robot_arm:
            self.robot_arm.enable()
    
    def disable_robot(self):
        """Disable the robot."""
        if self.robot_arm:
            self.robot_arm.disable()
    
    def show_3d_visualization(self):
        """Show or create 3D visualization window."""
        if not self.visualization_enabled or not VISUALIZATION_AVAILABLE:
            messagebox.showinfo("3D Visualization",
                              "3D visualization is not available on this system.\n\n"
                              "To enable 3D visualization, install:\n"
                              "pip install PyOpenGL PyOpenGL-accelerate glfw")
            return

        if not self.visualization_window:
            self.create_3d_visualization()
        elif self.visualization_window:
            self.visualization_window.show()
    
    def refresh_all_displays(self):
        """Refresh all display components."""
        self._update_gui_components()
        if self.visualization_window:
            self.visualization_window.refresh()
    
    def show_about(self):
        """Show about dialog."""
        about_text = """Robot Arm Desktop Control - Native GUI

A comprehensive native Python desktop application for robot arm simulation and control.

Features:
• Real-time 3D anthropomorphic arm visualization
• Interactive joint controls with real-time feedback  
• Natural language command processing
• Comprehensive status monitoring and logging
• Cross-platform compatibility

Version: 1.0.0
Built with Python, tkinter, and OpenGL"""
        
        messagebox.showinfo("About", about_text)
    
    def show_controls_help(self):
        """Show controls help dialog."""
        help_text = """Robot Arm Controls Help

Joint Control:
• Use sliders to control individual joint positions
• Real-time feedback shows current vs target positions
• Joint limits are enforced automatically

Natural Language Commands:
• Type commands like "wave hello", "move forward"
• Use predefined command buttons for common actions
• Command history shows previous commands and results

3D Visualization:
• Drag with mouse to orbit camera around robot
• Scroll wheel to zoom in/out
• Right-click and drag to pan view

Keyboard Shortcuts:
• Ctrl+R: Reset robot to home position
• Ctrl+E: Emergency stop
• Ctrl+S: Save robot state
• Ctrl+O: Load robot state"""
        
        messagebox.showinfo("Controls Help", help_text)
    
    def on_closing(self):
        """Handle application closing."""
        print("🔄 Shutting down application...")
        
        # Stop update system
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        # Close 3D visualization
        if self.visualization_window:
            self.visualization_window.close()
        
        # Disable robot
        if self.robot_arm:
            self.robot_arm.disable()
        
        # Destroy main window
        if self.main_window:
            self.main_window.destroy()
        
        print("✅ Application shutdown complete")
    
    def run(self):
        """Run the native desktop GUI application."""
        print("🚀 Starting Native Desktop GUI Application...")
        
        try:
            # Initialize robot system
            if not self.initialize_robot_system():
                return False
            
            # Create main window
            self.create_main_window()
            
            # Create 3D visualization
            self.create_3d_visualization()
            
            # Start update system
            self.start_update_system()
            
            # Mark GUI as initialized
            self.gui_initialized = True
            
            print("✅ Native Desktop GUI Application started successfully!")
            print("📱 Features available:")
            print("   • Real-time robot arm control and monitoring")
            print("   • Interactive joint position controls")
            print("   • Natural language command processing")
            if self.visualization_enabled:
                print("   • 3D anthropomorphic arm visualization")
            print("   • Comprehensive status displays and logging")
            print("\n🎮 Use the GUI to control the robot arm!")
            
            # Start main GUI loop
            self.main_window.mainloop()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start GUI application: {e}")
            messagebox.showerror("Startup Error", 
                               f"Failed to start the application:\n{str(e)}")
            return False


def main():
    """Main entry point for the native desktop GUI."""
    print("=" * 60)
    print("🤖 Robot Arm Simulation - Native Desktop GUI")
    print("=" * 60)
    
    # Create and run the application
    app = NativeDesktopGUI()
    success = app.run()
    
    if not success:
        print("\n❌ Application failed to start")
        print("\n🔧 Alternative options:")
        print("1. Use web interface: python web_3d_interface.py")
        print("2. Use command line: python main.py --demo")
        print("3. Check system compatibility: python test_installation.py")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
