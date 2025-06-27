"""
Robot Status Panel for Native Desktop GUI

This module provides comprehensive status monitoring including:
- Real-time end effector pose (position + orientation)
- Joint status information (positions, velocities, forces)
- System status and error messages
- Performance metrics and monitoring
- Movement status and control mode indicators
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Dict, List, Optional, Any
import time
import math

from robot_arm.robot_arm import RobotArm
from core.apple_silicon_compat import get_compat, safe_config_widget_colors, create_status_indicator
from core.config import config


class EndEffectorStatusFrame(ttk.LabelFrame):
    """Frame showing end effector pose and status."""
    
    def __init__(self, parent, robot_arm: RobotArm):
        super().__init__(parent, text="🎯 End Effector Status", padding=10)
        self.robot_arm = robot_arm
        
        self.position_labels = {}
        self.orientation_labels = {}
        self.velocity_labels = {}
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create end effector status widgets."""
        # Position section
        pos_frame = ttk.LabelFrame(self, text="Position (m)", padding=5)
        pos_frame.pack(fill="x", pady=5)
        
        pos_grid = ttk.Frame(pos_frame)
        pos_grid.pack(fill="x")
        
        for i, axis in enumerate(['X', 'Y', 'Z']):
            ttk.Label(pos_grid, text=f"{axis}:", width=3).grid(row=0, column=i*2, sticky="w")
            label = ttk.Label(pos_grid, text="0.000", width=8, font=("Courier", 10),
                             relief="sunken", anchor="center")
            label.grid(row=0, column=i*2+1, padx=(0, 10), sticky="ew")
            self.position_labels[axis.lower()] = label
        
        # Configure grid weights
        for i in range(6):
            pos_grid.columnconfigure(i, weight=1)
        
        # Orientation section
        orient_frame = ttk.LabelFrame(self, text="Orientation (rad)", padding=5)
        orient_frame.pack(fill="x", pady=5)
        
        orient_grid = ttk.Frame(orient_frame)
        orient_grid.pack(fill="x")
        
        for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
            ttk.Label(orient_grid, text=f"{axis}:", width=6).grid(row=0, column=i*2, sticky="w")
            label = ttk.Label(orient_grid, text="0.000", width=8, font=("Courier", 10),
                             relief="sunken", anchor="center")
            label.grid(row=0, column=i*2+1, padx=(0, 10), sticky="ew")
            self.orientation_labels[axis.lower()] = label
        
        # Configure grid weights
        for i in range(6):
            orient_grid.columnconfigure(i, weight=1)
        
        # Velocity section
        vel_frame = ttk.LabelFrame(self, text="End Effector Velocity", padding=5)
        vel_frame.pack(fill="x", pady=5)
        
        vel_grid = ttk.Frame(vel_frame)
        vel_grid.pack(fill="x")
        
        ttk.Label(vel_grid, text="Linear:", width=8).grid(row=0, column=0, sticky="w")
        self.linear_vel_label = ttk.Label(vel_grid, text="0.000 m/s", width=12, 
                                         font=("Courier", 10), relief="sunken", anchor="center")
        self.linear_vel_label.grid(row=0, column=1, padx=5, sticky="ew")
        
        ttk.Label(vel_grid, text="Angular:", width=8).grid(row=0, column=2, sticky="w")
        self.angular_vel_label = ttk.Label(vel_grid, text="0.000 rad/s", width=12, 
                                          font=("Courier", 10), relief="sunken", anchor="center")
        self.angular_vel_label.grid(row=0, column=3, padx=5, sticky="ew")
        
        # Configure grid weights
        for i in range(4):
            vel_grid.columnconfigure(i, weight=1)
    
    def update_status(self):
        """Update end effector status display."""
        try:
            # Get current end effector pose
            position, rotation_matrix = self.robot_arm.get_end_effector_pose()

            # Get compatibility instance
            compat = get_compat()

            # Update position labels
            for i, axis in enumerate(['x', 'y', 'z']):
                value = position[i] if len(position) > i else 0.0
                self.position_labels[axis].config(text=f"{value:.3f}")

                # Color coding based on workspace limits (Apple Silicon compatible)
                if abs(value) > 1.0:  # Example workspace limit
                    # Try to set background color, fall back to text indicator
                    if not safe_config_widget_colors(self.position_labels[axis], bg="lightcoral")['bg']:
                        # Use text indicator when colors not supported
                        self.position_labels[axis].config(text=f"⚠️ {value:.3f}")
                else:
                    if not safe_config_widget_colors(self.position_labels[axis], bg="lightgreen")['bg']:
                        # Use text indicator when colors not supported
                        self.position_labels[axis].config(text=f"✅ {value:.3f}")
            
            # Convert rotation matrix to Euler angles
            if rotation_matrix is not None and rotation_matrix.shape == (3, 3):
                euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
                
                for i, axis in enumerate(['roll', 'pitch', 'yaw']):
                    angle = euler_angles[i] if len(euler_angles) > i else 0.0
                    self.orientation_labels[axis].config(text=f"{angle:.3f}")
            
            # Calculate and display velocities (simplified)
            joint_velocities = self.robot_arm.get_joint_velocities()
            linear_vel = np.linalg.norm(joint_velocities[:3]) if len(joint_velocities) >= 3 else 0.0
            angular_vel = np.linalg.norm(joint_velocities[3:]) if len(joint_velocities) > 3 else 0.0
            
            self.linear_vel_label.config(text=f"{linear_vel:.3f} m/s")
            self.angular_vel_label.config(text=f"{angular_vel:.3f} rad/s")
            
        except Exception as e:
            print(f"Error updating end effector status: {e}")
    
    def _rotation_matrix_to_euler(self, R):
        """Convert rotation matrix to Euler angles (ZYX convention)."""
        try:
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            
            singular = sy < 1e-6
            
            if not singular:
                x = math.atan2(R[2, 1], R[2, 2])
                y = math.atan2(-R[2, 0], sy)
                z = math.atan2(R[1, 0], R[0, 0])
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0
            
            return [x, y, z]  # roll, pitch, yaw
        except:
            return [0.0, 0.0, 0.0]


class SystemStatusFrame(ttk.LabelFrame):
    """Frame showing system status and performance metrics."""
    
    def __init__(self, parent, robot_arm: RobotArm):
        super().__init__(parent, text="🔧 System Status", padding=10)
        self.robot_arm = robot_arm
        
        self.last_update_time = time.time()
        self.update_count = 0
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create system status widgets."""
        # Robot status
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", pady=5)
        
        ttk.Label(status_frame, text="Robot Status:", width=15).pack(side="left")
        self.robot_status_label = ttk.Label(status_frame, text="Unknown", 
                                           font=("Arial", 10, "bold"))
        self.robot_status_label.pack(side="left", padx=5)
        
        # Control mode
        mode_frame = ttk.Frame(self)
        mode_frame.pack(fill="x", pady=2)
        
        ttk.Label(mode_frame, text="Control Mode:", width=15).pack(side="left")
        self.control_mode_label = ttk.Label(mode_frame, text="Position", 
                                           font=("Courier", 10))
        self.control_mode_label.pack(side="left", padx=5)
        
        # Performance metrics
        perf_frame = ttk.LabelFrame(self, text="Performance", padding=5)
        perf_frame.pack(fill="x", pady=5)
        
        # Update rate
        rate_frame = ttk.Frame(perf_frame)
        rate_frame.pack(fill="x", pady=2)
        
        ttk.Label(rate_frame, text="Update Rate:", width=12).pack(side="left")
        self.update_rate_label = ttk.Label(rate_frame, text="0.0 Hz", 
                                          font=("Courier", 10))
        self.update_rate_label.pack(side="left", padx=5)
        
        # Joint limits status
        limits_frame = ttk.Frame(perf_frame)
        limits_frame.pack(fill="x", pady=2)
        
        ttk.Label(limits_frame, text="Joints at Limits:", width=15).pack(side="left")
        self.limits_label = ttk.Label(limits_frame, text="0", font=("Courier", 10))
        self.limits_label.pack(side="left", padx=5)
        
        # Error status
        error_frame = ttk.LabelFrame(self, text="Status Messages", padding=5)
        error_frame.pack(fill="both", expand=True, pady=5)
        
        # Status text area with scrollbar
        text_frame = ttk.Frame(error_frame)
        text_frame.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.status_text = tk.Text(text_frame, height=6, width=40, 
                                  yscrollcommand=scrollbar.set,
                                  font=("Courier", 9), state="disabled")
        self.status_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.status_text.yview)
        
        # Add initial status message
        self._add_status_message("System initialized", "INFO")
    
    def update_status(self):
        """Update system status display."""
        try:
            # Update robot status with Apple Silicon compatibility
            if self.robot_arm.is_enabled:
                if not safe_config_widget_colors(self.robot_status_label, fg="green")['fg']:
                    self.robot_status_label.config(text="✅ Enabled")
                else:
                    self.robot_status_label.config(text="Enabled")
            else:
                if not safe_config_widget_colors(self.robot_status_label, fg="red")['fg']:
                    self.robot_status_label.config(text="❌ Disabled")
                else:
                    self.robot_status_label.config(text="Disabled")

            # Update control mode
            self.control_mode_label.config(text=self.robot_arm.control_mode.title())

            # Calculate update rate
            current_time = time.time()
            self.update_count += 1

            if current_time - self.last_update_time >= 1.0:  # Update every second
                update_rate = self.update_count / (current_time - self.last_update_time)
                self.update_rate_label.config(text=f"{update_rate:.1f} Hz")
                self.last_update_time = current_time
                self.update_count = 0

            # Check joint limits with Apple Silicon compatibility
            joint_info = self.robot_arm.get_joint_info()
            joints_at_limits = sum(1 for info in joint_info.values() if info['at_limit'])

            if joints_at_limits > 0:
                if not safe_config_widget_colors(self.limits_label, fg="red")['fg']:
                    self.limits_label.config(text=f"⚠️ {joints_at_limits}")
                else:
                    self.limits_label.config(text=str(joints_at_limits))
            else:
                if not safe_config_widget_colors(self.limits_label, fg="green")['fg']:
                    self.limits_label.config(text=f"✅ {joints_at_limits}")
                else:
                    self.limits_label.config(text=str(joints_at_limits))
            
        except Exception as e:
            self._add_status_message(f"Status update error: {e}", "ERROR")
    
    def _add_status_message(self, message: str, level: str = "INFO"):
        """Add a status message to the status text area."""
        try:
            timestamp = time.strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {level}: {message}\n"
            
            self.status_text.config(state="normal")
            self.status_text.insert(tk.END, formatted_message)
            
            # Color coding
            if level == "ERROR":
                self.status_text.tag_add("error", "end-2l", "end-1l")
                self.status_text.tag_config("error", foreground="red")
            elif level == "WARNING":
                self.status_text.tag_add("warning", "end-2l", "end-1l")
                self.status_text.tag_config("warning", foreground="orange")
            elif level == "SUCCESS":
                self.status_text.tag_add("success", "end-2l", "end-1l")
                self.status_text.tag_config("success", foreground="green")
            
            self.status_text.see(tk.END)
            self.status_text.config(state="disabled")
            
            # Limit text length
            lines = self.status_text.get("1.0", tk.END).split('\n')
            if len(lines) > 100:  # Keep last 100 lines
                self.status_text.config(state="normal")
                self.status_text.delete("1.0", f"{len(lines)-100}.0")
                self.status_text.config(state="disabled")
                
        except Exception as e:
            print(f"Error adding status message: {e}")


class JointStatusFrame(ttk.LabelFrame):
    """Frame showing detailed joint status information."""
    
    def __init__(self, parent, robot_arm: RobotArm):
        super().__init__(parent, text="⚙️ Joint Status Summary", padding=10)
        self.robot_arm = robot_arm
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create joint status summary widgets."""
        # Create treeview for joint status
        columns = ("Joint", "Position", "Target", "Velocity", "Status")
        self.joint_tree = ttk.Treeview(self, columns=columns, show="headings", height=8)
        
        # Configure columns
        self.joint_tree.heading("Joint", text="Joint")
        self.joint_tree.heading("Position", text="Position")
        self.joint_tree.heading("Target", text="Target")
        self.joint_tree.heading("Velocity", text="Velocity")
        self.joint_tree.heading("Status", text="Status")
        
        self.joint_tree.column("Joint", width=120)
        self.joint_tree.column("Position", width=80)
        self.joint_tree.column("Target", width=80)
        self.joint_tree.column("Velocity", width=80)
        self.joint_tree.column("Status", width=80)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.joint_tree.yview)
        self.joint_tree.configure(yscrollcommand=scrollbar.set)
        
        self.joint_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Initialize joint entries
        self._initialize_joint_entries()
    
    def _initialize_joint_entries(self):
        """Initialize joint entries in the treeview."""
        for joint_name in self.robot_arm.joints.keys():
            self.joint_tree.insert("", "end", iid=joint_name, values=(
                joint_name.replace('_', ' ').title(),
                "0.000", "0.000", "0.000", "OK"
            ))
    
    def update_status(self):
        """Update joint status display."""
        try:
            joint_info = self.robot_arm.get_joint_info()
            
            for joint_name, info in joint_info.items():
                if self.joint_tree.exists(joint_name):
                    position = info['position']
                    target = info['target_position']
                    velocity = info['velocity']
                    at_limit = info['at_limit']
                    
                    # Determine status with Apple Silicon compatibility
                    compat = get_compat()
                    if at_limit:
                        if compat.supports_bg_option:
                            status = "LIMIT"
                            tags = ("limit",)
                        else:
                            status = "🚫 LIMIT"
                            tags = ()
                    elif abs(velocity) > 0.01:
                        if compat.supports_bg_option:
                            status = "MOVING"
                            tags = ("moving",)
                        else:
                            status = "🔄 MOVING"
                            tags = ()
                    else:
                        if compat.supports_bg_option:
                            status = "OK"
                            tags = ("ok",)
                        else:
                            status = "✅ OK"
                            tags = ()

                    # Update values
                    self.joint_tree.item(joint_name, values=(
                        joint_name.replace('_', ' ').title(),
                        f"{position:.3f}",
                        f"{target:.3f}",
                        f"{velocity:.3f}",
                        status
                    ), tags=tags)
            
            # Configure tags for color coding with Apple Silicon compatibility
            compat = get_compat()
            if compat.supports_bg_option:
                self.joint_tree.tag_configure("limit", background="lightcoral")
                self.joint_tree.tag_configure("moving", background="lightyellow")
                self.joint_tree.tag_configure("ok", background="lightgreen")
            else:
                # Use alternative indicators when colors not supported
                # TreeView will use text-based status indicators
                pass
            
        except Exception as e:
            print(f"Error updating joint status: {e}")


class RobotStatusPanel(ttk.Frame):
    """Main robot status panel combining all status displays."""
    
    def __init__(self, parent, robot_arm: RobotArm):
        super().__init__(parent)
        self.robot_arm = robot_arm
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the robot status panel widgets."""
        # Create notebook for different status views
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        
        # End effector status tab
        ee_frame = ttk.Frame(notebook)
        notebook.add(ee_frame, text="🎯 End Effector")
        self.ee_status = EndEffectorStatusFrame(ee_frame, self.robot_arm)
        self.ee_status.pack(fill="both", expand=True, padx=5, pady=5)
        
        # System status tab
        sys_frame = ttk.Frame(notebook)
        notebook.add(sys_frame, text="🔧 System")
        self.system_status = SystemStatusFrame(sys_frame, self.robot_arm)
        self.system_status.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Joint status tab
        joint_frame = ttk.Frame(notebook)
        notebook.add(joint_frame, text="⚙️ Joints")
        self.joint_status = JointStatusFrame(joint_frame, self.robot_arm)
        self.joint_status.pack(fill="both", expand=True, padx=5, pady=5)
    
    def update_status(self):
        """Update all status displays."""
        try:
            if hasattr(self, 'ee_status'):
                self.ee_status.update_status()
            
            if hasattr(self, 'system_status'):
                self.system_status.update_status()
            
            if hasattr(self, 'joint_status'):
                self.joint_status.update_status()
                
        except Exception as e:
            print(f"Error updating robot status panel: {e}")
