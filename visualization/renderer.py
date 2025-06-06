"""3D renderer for robot arm visualization using OpenGL."""

import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
import math
from typing import Tuple, List, Optional, Dict, Any

from core.config import config
from robot_arm.robot_arm import RobotArm


class Camera:
    """3D camera for visualization."""

    def __init__(self, position: List[float] = [2.0, 2.0, 1.5],
                 target: List[float] = [0.0, 0.0, 0.5],
                 up: List[float] = [0.0, 0.0, 1.0]):
        """Initialize camera.

        Args:
            position: Camera position
            target: Look-at target
            up: Up vector
        """
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)

        # Camera parameters
        self.fov = config.camera_config.get('fov', 45.0)
        self.near_plane = config.camera_config.get('near_plane', 0.1)
        self.far_plane = config.camera_config.get('far_plane', 100.0)

        # Mouse interaction
        self.mouse_sensitivity = 0.005
        self.zoom_sensitivity = 0.1
        self.pan_sensitivity = 0.01

        # Spherical coordinates for orbit camera
        self.distance = np.linalg.norm(self.position - self.target)
        self.azimuth = math.atan2(self.position[1] - self.target[1],
                                 self.position[0] - self.target[0])
        self.elevation = math.asin((self.position[2] - self.target[2]) / self.distance)

    def update_position_from_spherical(self) -> None:
        """Update camera position from spherical coordinates."""
        x = self.target[0] + self.distance * math.cos(self.elevation) * math.cos(self.azimuth)
        y = self.target[1] + self.distance * math.cos(self.elevation) * math.sin(self.azimuth)
        z = self.target[2] + self.distance * math.sin(self.elevation)
        self.position = np.array([x, y, z], dtype=np.float32)

    def orbit(self, delta_azimuth: float, delta_elevation: float) -> None:
        """Orbit camera around target.

        Args:
            delta_azimuth: Change in azimuth angle
            delta_elevation: Change in elevation angle
        """
        self.azimuth += delta_azimuth
        self.elevation += delta_elevation

        # Clamp elevation to avoid gimbal lock
        self.elevation = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.elevation))

        self.update_position_from_spherical()

    def zoom(self, delta: float) -> None:
        """Zoom camera in/out.

        Args:
            delta: Zoom delta (positive = zoom in)
        """
        self.distance *= (1.0 - delta * self.zoom_sensitivity)
        self.distance = max(0.1, min(50.0, self.distance))
        self.update_position_from_spherical()

    def pan(self, delta_x: float, delta_y: float) -> None:
        """Pan camera target.

        Args:
            delta_x: Horizontal pan delta
            delta_y: Vertical pan delta
        """
        # Calculate camera right and up vectors
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Pan target
        self.target += right * delta_x * self.pan_sensitivity * self.distance
        self.target += up * delta_y * self.pan_sensitivity * self.distance

        self.update_position_from_spherical()

    def apply_view_matrix(self) -> None:
        """Apply camera view matrix to OpenGL."""
        gluLookAt(
            self.position[0], self.position[1], self.position[2],
            self.target[0], self.target[1], self.target[2],
            self.up[0], self.up[1], self.up[2]
        )


class Renderer:
    """OpenGL renderer for robot arm visualization."""

    def __init__(self, width: int = 1200, height: int = 800, title: str = "Robot Arm Simulation"):
        """Initialize renderer.

        Args:
            width: Window width
            height: Window height
            title: Window title
        """
        self.width = width
        self.height = height
        self.title = title

        # GLFW window
        self.window = None

        # Camera
        camera_config = config.camera_config
        self.camera = Camera(
            position=camera_config.get('initial_position', [2.0, 2.0, 1.5]),
            target=camera_config.get('initial_target', [0.0, 0.0, 0.5])
        )

        # Mouse state
        self.mouse_last_x = 0.0
        self.mouse_last_y = 0.0
        self.mouse_left_pressed = False
        self.mouse_right_pressed = False
        self.mouse_middle_pressed = False

        # Rendering options
        self.show_coordinate_frames = True
        self.show_joint_spheres = True
        self.show_wireframe = False

        # Colors
        self.colors = config.colors

        self._initialize_glfw()
        self._initialize_opengl()

    def _initialize_glfw(self) -> None:
        """Initialize GLFW and create window."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Configure GLFW
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)  # 4x MSAA

        # Create window
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.set_window_user_pointer(self.window, self)

        # Set callbacks
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        # Enable V-Sync
        glfw.swap_interval(1)

    def _initialize_opengl(self) -> None:
        """Initialize OpenGL settings."""
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Enable multisampling
        glEnable(GL_MULTISAMPLE)

        # Set clear color
        glClearColor(0.2, 0.2, 0.2, 1.0)

        # Enable lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        # Set light properties
        light_position = [2.0, 2.0, 3.0, 1.0]
        light_ambient = [0.3, 0.3, 0.3, 1.0]
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        light_specular = [1.0, 1.0, 1.0, 1.0]

        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

        # Enable color material
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def _framebuffer_size_callback(self, window, width: int, height: int) -> None:
        """Handle window resize."""
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)

    def _mouse_button_callback(self, window, button: int, action: int, mods: int) -> None:
        """Handle mouse button events."""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_left_pressed = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_right_pressed = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.mouse_middle_pressed = (action == glfw.PRESS)

    def _cursor_pos_callback(self, window, xpos: float, ypos: float) -> None:
        """Handle mouse movement."""
        dx = xpos - self.mouse_last_x
        dy = ypos - self.mouse_last_y

        if self.mouse_left_pressed:
            # Orbit camera
            self.camera.orbit(-dx * self.camera.mouse_sensitivity,
                            dy * self.camera.mouse_sensitivity)
        elif self.mouse_right_pressed:
            # Pan camera
            self.camera.pan(dx, -dy)

        self.mouse_last_x = xpos
        self.mouse_last_y = ypos

    def _scroll_callback(self, window, xoffset: float, yoffset: float) -> None:
        """Handle mouse scroll (zoom)."""
        self.camera.zoom(yoffset)

    def _key_callback(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        """Handle keyboard input."""
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_F:
                self.show_coordinate_frames = not self.show_coordinate_frames
            elif key == glfw.KEY_J:
                self.show_joint_spheres = not self.show_joint_spheres
            elif key == glfw.KEY_W:
                self.show_wireframe = not self.show_wireframe
                if self.show_wireframe:
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                else:
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def should_close(self) -> bool:
        """Check if window should close."""
        return glfw.window_should_close(self.window)

    def poll_events(self) -> None:
        """Poll for events."""
        glfw.poll_events()

    def swap_buffers(self) -> None:
        """Swap front and back buffers."""
        glfw.swap_buffers(self.window)

    def clear(self) -> None:
        """Clear the screen."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def setup_projection(self) -> None:
        """Setup projection matrix."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        aspect_ratio = self.width / self.height if self.height > 0 else 1.0
        gluPerspective(self.camera.fov, aspect_ratio,
                      self.camera.near_plane, self.camera.far_plane)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.camera.apply_view_matrix()

    def draw_coordinate_frame(self, size: float = 0.1) -> None:
        """Draw coordinate frame axes.

        Args:
            size: Size of the axes
        """
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)

        glBegin(GL_LINES)

        # X-axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(size, 0.0, 0.0)

        # Y-axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, size, 0.0)

        # Z-axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, size)

        glEnd()

        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def draw_sphere(self, position: List[float], radius: float,
                   color: List[float] = [1.0, 0.0, 0.0]) -> None:
        """Draw a sphere.

        Args:
            position: Sphere center position
            radius: Sphere radius
            color: RGB color
        """
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])
        glColor3f(color[0], color[1], color[2])

        # Use GLU quadric for sphere
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, 16, 16)
        gluDeleteQuadric(quadric)

        glPopMatrix()

    def draw_cylinder(self, start: List[float], end: List[float],
                     radius: float, color: List[float] = [0.5, 0.5, 0.5]) -> None:
        """Draw a cylinder between two points.

        Args:
            start: Start position
            end: End position
            radius: Cylinder radius
            color: RGB color
        """
        glPushMatrix()

        # Calculate cylinder transformation
        start_vec = np.array(start)
        end_vec = np.array(end)
        direction = end_vec - start_vec
        length = np.linalg.norm(direction)

        if length < 1e-6:
            glPopMatrix()
            return

        direction = direction / length

        # Move to start position
        glTranslatef(start[0], start[1], start[2])

        # Rotate to align with direction
        # Default cylinder is along Z-axis, we need to rotate to align with direction
        z_axis = np.array([0, 0, 1])
        if np.dot(direction, z_axis) < 0.999:  # Not already aligned
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = math.acos(np.dot(z_axis, direction))
            glRotatef(math.degrees(rotation_angle),
                     rotation_axis[0], rotation_axis[1], rotation_axis[2])
        elif np.dot(direction, z_axis) < -0.999:  # Opposite direction
            glRotatef(180, 1, 0, 0)

        glColor3f(color[0], color[1], color[2])

        # Draw cylinder
        quadric = gluNewQuadric()
        gluCylinder(quadric, radius, radius, length, 16, 1)
        gluDeleteQuadric(quadric)

        glPopMatrix()

    def draw_robot_arm(self, robot_arm: RobotArm) -> None:
        """Draw the robot arm.

        Args:
            robot_arm: Robot arm to draw
        """
        if robot_arm.fk_solver is None:
            return

        # Update forward kinematics
        robot_arm.fk_solver.update()

        # Get link positions
        link_positions = robot_arm.fk_solver.get_link_positions()

        if len(link_positions) == 0:
            return

        # Draw base
        base_color = self.colors.get('base', [0.3, 0.3, 0.3])
        self.draw_sphere([0, 0, 0], 0.05, base_color)

        # Draw main arm links
        link_colors = [
            self.colors.get('upper_arm', [0.8, 0.2, 0.2]),
            self.colors.get('forearm', [0.2, 0.8, 0.2]),
            self.colors.get('hand', [0.2, 0.2, 0.8])
        ]

        # Draw links as cylinders
        prev_pos = [0, 0, 0]
        for i, (pos, color) in enumerate(zip(link_positions[:3], link_colors)):
            if i < len(link_positions):
                self.draw_cylinder(prev_pos, pos.tolist(), 0.02, color)
                prev_pos = pos.tolist()

        # Draw joints as spheres
        if self.show_joint_spheres:
            joint_color = self.colors.get('joints', [1.0, 0.0, 0.0])
            for pos in link_positions:
                self.draw_sphere(pos.tolist(), 0.015, joint_color)

        # Draw coordinate frames at joints
        if self.show_coordinate_frames:
            transforms = robot_arm.fk_solver.compute_joint_transforms()
            for transform in transforms:
                glPushMatrix()

                # Apply transformation
                transform_gl = transform.T.flatten()  # OpenGL uses column-major
                glMultMatrixf(transform_gl)

                # Draw coordinate frame
                self.draw_coordinate_frame(0.05)

                glPopMatrix()

        # Draw end effector
        if len(link_positions) > 0:
            end_pos = link_positions[-1]
            self.draw_sphere(end_pos.tolist(), 0.02, [1.0, 1.0, 0.0])  # Yellow

    def draw_grid(self, size: float = 2.0, spacing: float = 0.1) -> None:
        """Draw a grid on the ground plane.

        Args:
            size: Grid size
            spacing: Grid line spacing
        """
        glDisable(GL_LIGHTING)
        glColor3f(0.4, 0.4, 0.4)
        glLineWidth(1.0)

        glBegin(GL_LINES)

        # Draw grid lines
        num_lines = int(size / spacing)
        for i in range(-num_lines, num_lines + 1):
            x = i * spacing
            # Lines parallel to Y-axis
            glVertex3f(x, -size, 0)
            glVertex3f(x, size, 0)
            # Lines parallel to X-axis
            glVertex3f(-size, x, 0)
            glVertex3f(size, x, 0)

        glEnd()
        glEnable(GL_LIGHTING)

    def draw_text_2d(self, text: str, x: int, y: int) -> None:
        """Draw 2D text overlay (simplified version).

        Args:
            text: Text to draw
            x: X position in pixels
            y: Y position in pixels
        """
        # This is a simplified version - in a full implementation,
        # you would use a proper text rendering library like FreeType
        pass

    def render_frame(self, robot_arm: RobotArm,
                    additional_objects: Optional[List[Dict]] = None) -> None:
        """Render a complete frame.

        Args:
            robot_arm: Robot arm to render
            additional_objects: Additional objects to render
        """
        self.clear()
        self.setup_projection()

        # Draw grid
        self.draw_grid()

        # Draw world coordinate frame
        if self.show_coordinate_frames:
            self.draw_coordinate_frame(0.2)

        # Draw robot arm
        self.draw_robot_arm(robot_arm)

        # Draw additional objects
        if additional_objects:
            for obj in additional_objects:
                if obj['type'] == 'sphere':
                    self.draw_sphere(obj['position'], obj['radius'],
                                   obj.get('color', [1.0, 0.0, 0.0]))
                elif obj['type'] == 'cylinder':
                    self.draw_cylinder(obj['start'], obj['end'], obj['radius'],
                                     obj.get('color', [0.5, 0.5, 0.5]))

        self.swap_buffers()

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
