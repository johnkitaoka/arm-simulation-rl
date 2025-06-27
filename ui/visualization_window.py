"""
3D Visualization Window for Native Desktop GUI

This module provides a 3D visualization window that integrates with the existing
OpenGL renderer to display the anthropomorphic robot arm in real-time.

Features:
- Real-time 3D robot arm visualization
- Interactive camera controls (orbit, zoom, pan)
- Anthropomorphic arm rendering with joint details
- Coordinate frame displays
- End effector position tracking
- Smooth animation and updates
"""

import sys
import threading
import time
import numpy as np
from typing import Optional, Dict, Any, List

try:
    import glfw
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("⚠️  OpenGL/GLFW not available - 3D visualization disabled")

from robot_arm.robot_arm import RobotArm
from visualization.renderer import Renderer3D
from core.config import config


class VisualizationWindow:
    """3D visualization window for robot arm display."""
    
    def __init__(self, robot_arm: RobotArm):
        """Initialize the 3D visualization window.
        
        Args:
            robot_arm: Robot arm instance to visualize
        """
        self.robot_arm = robot_arm
        self.window = None
        self.renderer = None
        self.running = False
        self.render_thread = None
        
        # Window properties
        self.window_width = 800
        self.window_height = 600
        self.window_title = "Robot Arm 3D Visualization"
        
        # Rendering state
        self.last_render_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        
        # Camera state
        self.camera_distance = 2.0
        self.camera_azimuth = 45.0
        self.camera_elevation = 30.0
        self.camera_target = [0.0, 0.0, 0.5]
        
        # Mouse interaction state
        self.mouse_last_x = 0.0
        self.mouse_last_y = 0.0
        self.mouse_left_pressed = False
        self.mouse_right_pressed = False
        self.mouse_middle_pressed = False
        
        # Visualization options
        self.show_coordinate_frames = True
        self.show_joint_spheres = True
        self.show_grid = True
        self.show_workspace = False
        
        if OPENGL_AVAILABLE:
            self._initialize_window()
        else:
            raise RuntimeError("OpenGL/GLFW not available - cannot create 3D visualization")
    
    def _initialize_window(self):
        """Initialize the GLFW window and OpenGL context."""
        print("🎮 Initializing 3D visualization window...")
        
        try:
            # Initialize GLFW
            if not glfw.init():
                raise RuntimeError("Failed to initialize GLFW")
            
            # Configure window hints
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
            glfw.window_hint(glfw.SAMPLES, 4)  # Anti-aliasing
            
            # Create window
            self.window = glfw.create_window(
                self.window_width, self.window_height, 
                self.window_title, None, None
            )
            
            if not self.window:
                glfw.terminate()
                raise RuntimeError("Failed to create GLFW window")
            
            # Make context current
            glfw.make_context_current(self.window)
            
            # Set callbacks
            glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)
            glfw.set_cursor_pos_callback(self.window, self._mouse_callback)
            glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            glfw.set_key_callback(self.window, self._key_callback)
            
            # Enable V-Sync
            glfw.swap_interval(1)
            
            # Initialize renderer
            self.renderer = Renderer3D(self.window_width, self.window_height)
            
            print("  ✅ 3D visualization window initialized")
            
        except Exception as e:
            print(f"  ❌ Failed to initialize 3D visualization: {e}")
            if self.window:
                glfw.destroy_window(self.window)
            glfw.terminate()
            raise
    
    def start_rendering(self):
        """Start the rendering loop in a separate thread."""
        if not OPENGL_AVAILABLE or not self.window:
            return
        
        print("🔄 Starting 3D rendering loop...")
        self.running = True
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()
        print("  ✅ 3D rendering started")
    
    def _render_loop(self):
        """Main rendering loop."""
        while self.running and not glfw.window_should_close(self.window):
            try:
                current_time = time.time()
                dt = current_time - self.last_render_time
                self.last_render_time = current_time
                
                # Update camera
                self._update_camera()
                
                # Render frame
                self._render_frame()
                
                # Swap buffers and poll events
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                
                # Calculate FPS
                self.frame_count += 1
                if self.frame_count % 60 == 0:  # Update FPS every 60 frames
                    self.fps = 60.0 / (current_time - (self.last_render_time - 60 * dt))
                
                # Limit frame rate to ~60 FPS
                time.sleep(max(0, 1.0/60.0 - dt))
                
            except Exception as e:
                print(f"❌ Rendering error: {e}")
                time.sleep(0.1)
        
        print("🔄 3D rendering loop stopped")
    
    def _render_frame(self):
        """Render a single frame."""
        if not self.renderer:
            return
        
        try:
            # Render the robot arm and scene
            self.renderer.render_frame(self.robot_arm)
            
            # Render additional visualization elements
            self._render_additional_elements()
            
        except Exception as e:
            print(f"Frame rendering error: {e}")
    
    def _render_additional_elements(self):
        """Render additional visualization elements."""
        try:
            # Render workspace boundaries if enabled
            if self.show_workspace:
                self._render_workspace()
            
            # Render FPS counter
            self._render_fps_counter()
            
        except Exception as e:
            print(f"Additional elements rendering error: {e}")
    
    def _render_workspace(self):
        """Render workspace boundaries."""
        # Simple workspace visualization as a wireframe cube
        workspace_size = 1.0
        
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_LINES)
        
        # Draw wireframe cube
        vertices = [
            [-workspace_size, -workspace_size, 0],
            [workspace_size, -workspace_size, 0],
            [workspace_size, workspace_size, 0],
            [-workspace_size, workspace_size, 0],
            [-workspace_size, -workspace_size, workspace_size],
            [workspace_size, -workspace_size, workspace_size],
            [workspace_size, workspace_size, workspace_size],
            [-workspace_size, workspace_size, workspace_size]
        ]
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        for edge in edges:
            glVertex3fv(vertices[edge[0]])
            glVertex3fv(vertices[edge[1]])
        
        glEnd()
    
    def _render_fps_counter(self):
        """Render FPS counter (simplified)."""
        # This would typically use text rendering, which is complex in OpenGL
        # For now, we'll just print to console occasionally
        if self.frame_count % 300 == 0:  # Every 5 seconds at 60 FPS
            print(f"3D Visualization FPS: {self.fps:.1f}")
    
    def _update_camera(self):
        """Update camera position based on current state."""
        if not self.renderer:
            return
        
        # Convert spherical coordinates to Cartesian
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)
        
        x = self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = self.camera_distance * np.sin(elevation_rad)
        
        camera_pos = [
            x + self.camera_target[0],
            y + self.camera_target[1],
            z + self.camera_target[2]
        ]
        
        # Update renderer camera
        self.renderer.camera.position = camera_pos
        self.renderer.camera.target = self.camera_target.copy()
    
    # GLFW Callbacks
    def _framebuffer_size_callback(self, window, width, height):
        """Handle window resize."""
        self.window_width = width
        self.window_height = height
        
        if self.renderer:
            self.renderer.resize(width, height)
        
        glViewport(0, 0, width, height)
    
    def _mouse_callback(self, window, xpos, ypos):
        """Handle mouse movement."""
        dx = xpos - self.mouse_last_x
        dy = ypos - self.mouse_last_y
        
        if self.mouse_left_pressed:
            # Orbit camera
            self.camera_azimuth += dx * 0.5
            self.camera_elevation = np.clip(self.camera_elevation - dy * 0.5, -89, 89)
        
        elif self.mouse_right_pressed:
            # Pan camera
            pan_speed = 0.01
            azimuth_rad = np.radians(self.camera_azimuth)
            
            # Calculate pan directions
            right = [-np.sin(azimuth_rad), np.cos(azimuth_rad), 0]
            up = [0, 0, 1]
            
            for i in range(3):
                self.camera_target[i] += (right[i] * dx + up[i] * dy) * pan_speed
        
        self.mouse_last_x = xpos
        self.mouse_last_y = ypos
    
    def _mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button events."""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_left_pressed = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_right_pressed = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.mouse_middle_pressed = (action == glfw.PRESS)
        
        # Store mouse position when button is pressed
        if action == glfw.PRESS:
            self.mouse_last_x, self.mouse_last_y = glfw.get_cursor_pos(window)
    
    def _scroll_callback(self, window, xoffset, yoffset):
        """Handle mouse scroll (zoom)."""
        zoom_speed = 0.1
        self.camera_distance = max(0.5, self.camera_distance - yoffset * zoom_speed)
    
    def _key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input."""
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_R:
                # Reset camera
                self.camera_distance = 2.0
                self.camera_azimuth = 45.0
                self.camera_elevation = 30.0
                self.camera_target = [0.0, 0.0, 0.5]
            elif key == glfw.KEY_G:
                # Toggle grid
                self.show_grid = not self.show_grid
                if self.renderer:
                    self.renderer.show_grid = self.show_grid
            elif key == glfw.KEY_F:
                # Toggle coordinate frames
                self.show_coordinate_frames = not self.show_coordinate_frames
                if self.renderer:
                    self.renderer.show_coordinate_frames = self.show_coordinate_frames
            elif key == glfw.KEY_J:
                # Toggle joint spheres
                self.show_joint_spheres = not self.show_joint_spheres
                if self.renderer:
                    self.renderer.show_joint_spheres = self.show_joint_spheres
            elif key == glfw.KEY_W:
                # Toggle workspace
                self.show_workspace = not self.show_workspace
    
    # Public interface methods
    def update_visualization(self):
        """Update the visualization (called from main thread)."""
        # This method is called from the main update loop
        # The actual rendering happens in the render thread
        pass
    
    def show(self):
        """Show the visualization window."""
        if self.window:
            glfw.show_window(self.window)
            if not self.running:
                self.start_rendering()
    
    def hide(self):
        """Hide the visualization window."""
        if self.window:
            glfw.hide_window(self.window)
    
    def refresh(self):
        """Force refresh the visualization."""
        # The rendering loop handles continuous updates
        pass
    
    def close(self):
        """Close the visualization window."""
        print("🔄 Closing 3D visualization window...")
        
        self.running = False
        
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=1.0)
        
        if self.window:
            glfw.destroy_window(self.window)
            self.window = None
        
        if OPENGL_AVAILABLE:
            glfw.terminate()
        
        print("  ✅ 3D visualization window closed")
    
    def is_open(self) -> bool:
        """Check if the visualization window is open."""
        return self.window is not None and not glfw.window_should_close(self.window)
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get current camera information."""
        return {
            'distance': self.camera_distance,
            'azimuth': self.camera_azimuth,
            'elevation': self.camera_elevation,
            'target': self.camera_target.copy(),
            'fps': self.fps
        }
    
    def set_camera_position(self, distance: float, azimuth: float, elevation: float):
        """Set camera position."""
        self.camera_distance = max(0.1, distance)
        self.camera_azimuth = azimuth % 360
        self.camera_elevation = np.clip(elevation, -89, 89)
    
    def set_visualization_options(self, **options):
        """Set visualization options."""
        if 'show_coordinate_frames' in options:
            self.show_coordinate_frames = options['show_coordinate_frames']
            if self.renderer:
                self.renderer.show_coordinate_frames = self.show_coordinate_frames
        
        if 'show_joint_spheres' in options:
            self.show_joint_spheres = options['show_joint_spheres']
            if self.renderer:
                self.renderer.show_joint_spheres = self.show_joint_spheres
        
        if 'show_grid' in options:
            self.show_grid = options['show_grid']
            if self.renderer:
                self.renderer.show_grid = self.show_grid
        
        if 'show_workspace' in options:
            self.show_workspace = options['show_workspace']


# Fallback class for when OpenGL is not available
class FallbackVisualizationWindow:
    """Fallback visualization window when OpenGL is not available."""
    
    def __init__(self, robot_arm: RobotArm):
        self.robot_arm = robot_arm
        print("⚠️  Using fallback visualization (OpenGL not available)")
    
    def start_rendering(self):
        pass
    
    def update_visualization(self):
        pass
    
    def show(self):
        print("3D visualization not available - OpenGL/GLFW not installed")
    
    def hide(self):
        pass
    
    def refresh(self):
        pass
    
    def close(self):
        pass
    
    def is_open(self) -> bool:
        return False
    
    def get_camera_info(self) -> Dict[str, Any]:
        return {}
    
    def set_camera_position(self, distance: float, azimuth: float, elevation: float):
        pass
    
    def set_visualization_options(self, **options):
        pass


# Factory function to create appropriate visualization window
def create_visualization_window(robot_arm: RobotArm) -> 'VisualizationWindow':
    """Create a visualization window, with fallback if OpenGL is not available."""
    if OPENGL_AVAILABLE:
        try:
            return VisualizationWindow(robot_arm)
        except Exception as e:
            print(f"Failed to create OpenGL visualization: {e}")
            return FallbackVisualizationWindow(robot_arm)
    else:
        return FallbackVisualizationWindow(robot_arm)
