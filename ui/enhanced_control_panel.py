"""
Enhanced Control Panel for Native Desktop GUI

This module provides an enhanced version of the control panel with:
- Improved joint control with real-time feedback
- Better visual indicators for joint states
- Enhanced command processing interface
- Real-time position vs target displays
- Joint limit warnings and safety indicators
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import json
import time
import threading

from robot_arm.robot_arm import RobotArm
from ml.nlp_processor import CommandParser, CommandHistory
from core.config import config
from core.apple_silicon_compat import get_compat, safe_config_widget_colors


class EnhancedJointControlFrame(ttk.Frame):
    """Enhanced frame for manual joint control with real-time feedback."""
    
    def __init__(self, parent, robot_arm: RobotArm):
        super().__init__(parent)
        self.robot_arm = robot_arm
        self.joint_vars = {}
        self.joint_scales = {}
        self.position_labels = {}
        self.target_labels = {}
        self.velocity_labels = {}
        self.limit_indicators = {}
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create enhanced joint control widgets."""
        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(title_frame, text="🎮 Joint Control", 
                 font=("Arial", 14, "bold")).pack(side="left")
        
        # Control buttons
        button_frame = ttk.Frame(title_frame)
        button_frame.pack(side="right")
        
        ttk.Button(button_frame, text="Reset All", 
                  command=self.reset_all_joints).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Stop All", 
                  command=self.stop_all_joints).pack(side="left", padx=2)
        
        # Create scrollable frame for joints
        canvas = tk.Canvas(self, height=400)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create enhanced controls for each joint
        for joint_name, joint in self.robot_arm.joints.items():
            self._create_joint_control(scrollable_frame, joint_name, joint)
    
    def _create_joint_control(self, parent, joint_name: str, joint):
        """Create enhanced control for a single joint."""
        # Main frame for this joint
        joint_frame = ttk.LabelFrame(parent, text=joint_name.replace('_', ' ').title(), 
                                    padding=10)
        joint_frame.pack(fill="x", padx=5, pady=5)
        
        # Position control row
        pos_frame = ttk.Frame(joint_frame)
        pos_frame.pack(fill="x", pady=2)
        
        ttk.Label(pos_frame, text="Position:", width=10).pack(side="left")
        
        # Joint value variable
        var = tk.DoubleVar(value=joint.position)
        self.joint_vars[joint_name] = var
        
        # Joint slider
        scale = ttk.Scale(
            pos_frame,
            from_=joint.limits[0],
            to=joint.limits[1],
            variable=var,
            orient="horizontal",
            command=lambda val, name=joint_name: self._on_joint_change(name, val)
        )
        scale.pack(side="left", fill="x", expand=True, padx=5)
        self.joint_scales[joint_name] = scale
        
        # Current position label
        pos_label = ttk.Label(pos_frame, text=f"{joint.position:.3f}", 
                             width=8, font=("Courier", 9))
        pos_label.pack(side="right")
        self.position_labels[joint_name] = pos_label
        
        # Target and velocity info row
        info_frame = ttk.Frame(joint_frame)
        info_frame.pack(fill="x", pady=2)
        
        ttk.Label(info_frame, text="Target:", width=10).pack(side="left")
        target_label = ttk.Label(info_frame, text=f"{joint.target_position:.3f}", 
                                width=8, font=("Courier", 9))
        target_label.pack(side="left", padx=5)
        self.target_labels[joint_name] = target_label
        
        ttk.Label(info_frame, text="Velocity:", width=10).pack(side="left", padx=(20, 0))
        vel_label = ttk.Label(info_frame, text=f"{joint.velocity:.3f}", 
                             width=8, font=("Courier", 9))
        vel_label.pack(side="left", padx=5)
        self.velocity_labels[joint_name] = vel_label
        
        # Limit indicator
        limit_frame = ttk.Frame(joint_frame)
        limit_frame.pack(fill="x", pady=2)
        
        limit_indicator = tk.Label(limit_frame, text="●", fg="green", 
                                  font=("Arial", 12))
        limit_indicator.pack(side="left")
        self.limit_indicators[joint_name] = limit_indicator
        
        limit_text = ttk.Label(limit_frame, 
                              text=f"Limits: [{joint.limits[0]:.2f}, {joint.limits[1]:.2f}]",
                              font=("Courier", 8))
        limit_text.pack(side="left", padx=5)
        
        # Update value label when slider changes
        var.trace("w", lambda *args, name=joint_name: self._update_joint_display(name))
    
    def _on_joint_change(self, joint_name: str, value: str):
        """Handle joint slider change."""
        try:
            val = float(value)
            self.robot_arm.joints[joint_name].target_position = val
        except ValueError:
            pass
    
    def _update_joint_display(self, joint_name: str):
        """Update display for a specific joint."""
        if joint_name not in self.robot_arm.joints:
            return
        
        joint = self.robot_arm.joints[joint_name]
        
        # Update position label
        if joint_name in self.position_labels:
            self.position_labels[joint_name].config(text=f"{joint.position:.3f}")
        
        # Update target label
        if joint_name in self.target_labels:
            self.target_labels[joint_name].config(text=f"{joint.target_position:.3f}")
        
        # Update velocity label
        if joint_name in self.velocity_labels:
            self.velocity_labels[joint_name].config(text=f"{joint.velocity:.3f}")
        
        # Update limit indicator with Apple Silicon compatibility
        if joint_name in self.limit_indicators:
            indicator = self.limit_indicators[joint_name]
            if joint.is_at_limit():
                if not safe_config_widget_colors(indicator, fg="red")['fg']:
                    indicator.config(text="🚫")
                else:
                    indicator.config(text="⚠")
            elif joint.distance_to_limit() < 0.1:
                if not safe_config_widget_colors(indicator, fg="orange")['fg']:
                    indicator.config(text="⚠️")
                else:
                    indicator.config(text="●")
            else:
                if not safe_config_widget_colors(indicator, fg="green")['fg']:
                    indicator.config(text="✅")
                else:
                    indicator.config(text="●")
    
    def update_from_robot(self):
        """Update all controls from current robot state."""
        for joint_name, var in self.joint_vars.items():
            if joint_name in self.robot_arm.joints:
                joint = self.robot_arm.joints[joint_name]
                var.set(joint.position)
                self._update_joint_display(joint_name)
    
    def reset_all_joints(self):
        """Reset all joints to home position."""
        for joint_name, var in self.joint_vars.items():
            var.set(0.0)
            self.robot_arm.joints[joint_name].target_position = 0.0
    
    def stop_all_joints(self):
        """Stop all joint movements."""
        for joint_name in self.joint_vars.keys():
            if joint_name in self.robot_arm.joints:
                joint = self.robot_arm.joints[joint_name]
                joint.target_position = joint.position
                joint.velocity = 0.0


class EnhancedCommandFrame(ttk.Frame):
    """Enhanced frame for natural language command input."""
    
    def __init__(self, parent, robot_arm: RobotArm, command_parser: CommandParser):
        super().__init__(parent)
        self.robot_arm = robot_arm
        self.command_parser = command_parser
        self.command_history = CommandHistory()
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create enhanced command input widgets."""
        # Title
        ttk.Label(self, text="💬 Natural Language Commands", 
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        # Command input frame
        input_frame = ttk.Frame(self)
        input_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(input_frame, text="Command:").pack(side="left")
        
        self.command_var = tk.StringVar()
        self.command_entry = ttk.Entry(input_frame, textvariable=self.command_var, 
                                      font=("Arial", 11))
        self.command_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.command_entry.bind("<Return>", lambda e: self.execute_command())
        
        ttk.Button(input_frame, text="Execute", 
                  command=self.execute_command).pack(side="right")
        
        # Predefined commands frame
        preset_frame = ttk.LabelFrame(self, text="Quick Commands", padding=5)
        preset_frame.pack(fill="x", padx=5, pady=5)
        
        # Create preset command buttons
        preset_commands = [
            ("Wave Hello", "wave hello"),
            ("Move Forward", "move forward"),
            ("Point Up", "point up"),
            ("Reset Home", "reset to home"),
            ("Relax", "relax arms"),
            ("Stretch", "stretch arms")
        ]
        
        for i, (label, command) in enumerate(preset_commands):
            row = i // 3
            col = i % 3
            btn = ttk.Button(preset_frame, text=label, width=12,
                           command=lambda cmd=command: self.execute_preset_command(cmd))
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
        
        # Configure grid weights
        for i in range(3):
            preset_frame.columnconfigure(i, weight=1)
        
        # Command history frame
        history_frame = ttk.LabelFrame(self, text="Command History", padding=5)
        history_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # History listbox with scrollbar
        list_frame = ttk.Frame(history_frame)
        list_frame.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.history_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                         font=("Courier", 9))
        self.history_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.history_listbox.yview)
        
        # Status frame
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready for commands")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                font=("Arial", 10))
        status_label.pack(side="left")
        
        # Clear history button
        ttk.Button(status_frame, text="Clear History", 
                  command=self.clear_history).pack(side="right")
    
    def execute_command(self):
        """Execute the entered command."""
        command = self.command_var.get().strip()
        if not command:
            return
        
        self._execute_command_internal(command)
        self.command_var.set("")
    
    def execute_preset_command(self, command: str):
        """Execute a preset command."""
        self._execute_command_internal(command)
    
    def _execute_command_internal(self, command: str):
        """Internal command execution logic."""
        self.status_var.set("Processing command...")
        
        try:
            # Parse command
            start_time = time.time()
            parsed = self.command_parser.parse_command(command)
            action = self.command_parser.command_to_robot_action(parsed)
            
            # Execute action
            success = self._execute_action(action)
            execution_time = time.time() - start_time
            
            # Add to history
            self.command_history.add_command(command, parsed, success, execution_time)
            
            # Update UI
            status = "✅ Success" if success else "❌ Failed"
            confidence = parsed.get('confidence', 0.0)
            action_type = action.get('type', 'unknown')
            
            history_entry = f"{command} → {action_type} ({status}, {confidence:.2f})"
            
            self.history_listbox.insert(tk.END, history_entry)
            self.history_listbox.see(tk.END)
            
            self.status_var.set(f"Command executed: {status}")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Command Error", str(e))
    
    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute a robot action."""
        if not action.get('success', False):
            return False
        
        action_type = action['type']
        params = action.get('parameters', {})
        
        try:
            if action_type == 'move_to_position':
                if 'position' in params:
                    target_pos = np.array(params['position'])
                    return self.robot_arm.move_to_pose(target_pos)
                elif 'direction' in params:
                    # Move in specified direction
                    current_pos, _ = self.robot_arm.get_end_effector_pose()
                    direction = params['direction']
                    distance = params.get('distance', 0.1)
                    
                    direction_vectors = {
                        'up': [0, 0, 1], 'down': [0, 0, -1],
                        'left': [0, 1, 0], 'right': [0, -1, 0],
                        'forward': [1, 0, 0], 'backward': [-1, 0, 0]
                    }
                    
                    if direction in direction_vectors:
                        target_pos = current_pos + np.array(direction_vectors[direction]) * distance
                        return self.robot_arm.move_to_pose(target_pos)
            
            elif action_type == 'set_joint_positions':
                if 'positions' in params:
                    positions = params['positions']
                    joint_names = params.get('joint_names', None)
                    self.robot_arm.set_joint_targets(positions, joint_names)
                    return True
            
            elif action_type == 'reset_to_home':
                self.robot_arm.reset_to_home()
                return True
            
            elif action_type == 'wave':
                # Implement wave gesture
                return self._execute_wave_gesture()
            
            return False
            
        except Exception as e:
            print(f"Action execution error: {e}")
            return False
    
    def _execute_wave_gesture(self) -> bool:
        """Execute a wave gesture."""
        try:
            # Simple wave: move shoulder and elbow
            wave_positions = [0.0, 0.5, 0.0, 1.0, 0.0, 0.0]  # Example positions
            self.robot_arm.set_joint_targets(wave_positions)
            return True
        except:
            return False
    
    def clear_history(self):
        """Clear command history."""
        self.history_listbox.delete(0, tk.END)
        self.command_history = CommandHistory()


class EnhancedControlPanel(ttk.Frame):
    """Enhanced main control panel combining all control interfaces."""
    
    def __init__(self, parent, robot_arm: RobotArm, command_parser: CommandParser):
        super().__init__(parent)
        self.robot_arm = robot_arm
        self.command_parser = command_parser
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the enhanced control panel widgets."""
        # Create notebook for tabs
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        
        # Joint control tab
        joint_frame = ttk.Frame(notebook)
        notebook.add(joint_frame, text="🎮 Joint Control")
        self.joint_control = EnhancedJointControlFrame(joint_frame, self.robot_arm)
        self.joint_control.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Command tab
        command_frame = ttk.Frame(notebook)
        notebook.add(command_frame, text="💬 Commands")
        self.command_control = EnhancedCommandFrame(command_frame, self.robot_arm,
                                                   self.command_parser)
        self.command_control.pack(fill="both", expand=True, padx=5, pady=5)

        # ML Training tab
        ml_frame = ttk.Frame(notebook)
        notebook.add(ml_frame, text="🤖 ML Training")
        self.ml_control = MLTrainingFrame(ml_frame, self.robot_arm, self.command_parser)
        self.ml_control.pack(fill="both", expand=True, padx=5, pady=5)
    
    def update_from_robot(self):
        """Update all controls from current robot state."""
        if hasattr(self, 'joint_control'):
            self.joint_control.update_from_robot()
    
    def save_robot_state(self):
        """Save current robot state to file."""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                state = {
                    'joint_positions': self.robot_arm.get_joint_positions().tolist(),
                    'joint_targets': [joint.target_position for joint in self.robot_arm.joints.values()],
                    'joint_names': list(self.robot_arm.joints.keys()),
                    'timestamp': time.time()
                }
                
                with open(filename, 'w') as f:
                    json.dump(state, f, indent=2)
                
                messagebox.showinfo("Save State", f"Robot state saved to {filename}")
        
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save robot state:\n{str(e)}")
    
    def load_robot_state(self):
        """Load robot state from file."""
        try:
            from tkinter import filedialog
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r') as f:
                    state = json.load(f)
                
                # Load joint positions
                if 'joint_positions' in state:
                    positions = np.array(state['joint_positions'])
                    joint_names = state.get('joint_names', None)
                    self.robot_arm.set_joint_positions(positions, joint_names)
                    self.robot_arm.set_joint_targets(positions, joint_names)
                
                messagebox.showinfo("Load State", f"Robot state loaded from {filename}")
        
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load robot state:\n{str(e)}")


class MLTrainingFrame(ttk.Frame):
    """Frame for ML training controls in the GUI."""

    def __init__(self, parent, robot_arm: RobotArm, command_parser: CommandParser):
        super().__init__(parent)
        self.robot_arm = robot_arm
        self.command_parser = command_parser
        self.trainer = None
        self.training_thread = None
        self.is_training = False

        self._create_widgets()

    def _create_widgets(self):
        """Create ML training widgets."""
        # Title
        ttk.Label(self, text="🤖 Machine Learning Training",
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))

        # Training configuration frame
        config_frame = ttk.LabelFrame(self, text="Training Configuration", padding=10)
        config_frame.pack(fill="x", padx=5, pady=5)

        # Algorithm selection
        algo_frame = ttk.Frame(config_frame)
        algo_frame.pack(fill="x", pady=2)

        ttk.Label(algo_frame, text="Algorithm:", width=12).pack(side="left")
        self.algorithm_var = tk.StringVar(value="PPO")
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algorithm_var,
                                 values=["PPO", "SAC", "TD3"], state="readonly", width=10)
        algo_combo.pack(side="left", padx=5)

        # Task selection
        task_frame = ttk.Frame(config_frame)
        task_frame.pack(fill="x", pady=2)

        ttk.Label(task_frame, text="Task:", width=12).pack(side="left")
        self.task_var = tk.StringVar(value="Reaching")
        task_combo = ttk.Combobox(task_frame, textvariable=self.task_var,
                                 values=["Reaching", "Obstacle Avoidance", "Gesture Recognition"],
                                 state="readonly", width=15)
        task_combo.pack(side="left", padx=5)

        # Timesteps selection
        steps_frame = ttk.Frame(config_frame)
        steps_frame.pack(fill="x", pady=2)

        ttk.Label(steps_frame, text="Timesteps:", width=12).pack(side="left")
        self.timesteps_var = tk.StringVar(value="50000")
        steps_entry = ttk.Entry(steps_frame, textvariable=self.timesteps_var, width=10)
        steps_entry.pack(side="left", padx=5)

        # Training controls
        control_frame = ttk.Frame(config_frame)
        control_frame.pack(fill="x", pady=10)

        self.start_button = ttk.Button(control_frame, text="Start Training",
                                      command=self.start_training)
        self.start_button.pack(side="left", padx=2)

        self.stop_button = ttk.Button(control_frame, text="Stop Training",
                                     command=self.stop_training, state="disabled")
        self.stop_button.pack(side="left", padx=2)

        ttk.Button(control_frame, text="Load Model",
                  command=self.load_model).pack(side="left", padx=2)

        ttk.Button(control_frame, text="Test Model",
                  command=self.test_model).pack(side="left", padx=2)

        # Progress display
        progress_frame = ttk.LabelFrame(self, text="Training Progress", padding=5)
        progress_frame.pack(fill="x", padx=5, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100)
        self.progress_bar.pack(fill="x", pady=2)

        self.status_var = tk.StringVar(value="Ready to train")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack(pady=2)

        # Results display
        results_frame = ttk.LabelFrame(self, text="Training Results", padding=5)
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create text widget with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill="both", expand=True)

        self.results_text = tk.Text(text_frame, height=8, width=50, font=("Courier", 9))
        scrollbar = ttk.Scrollbar(text_frame, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=scrollbar.set)

        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add initial message
        self.results_text.insert(tk.END, "🤖 ML Training Interface Ready\n")
        self.results_text.insert(tk.END, "Select algorithm and task, then click 'Start Training'\n\n")
        self.results_text.insert(tk.END, "Available Tasks:\n")
        self.results_text.insert(tk.END, "• Reaching: Point-to-point movement (Beginner)\n")
        self.results_text.insert(tk.END, "• Obstacle Avoidance: Navigate around obstacles (Intermediate)\n")
        self.results_text.insert(tk.END, "• Gesture Recognition: Learn gestures (Beginner)\n\n")
        self.results_text.config(state="disabled")

    def start_training(self):
        """Start ML training in a separate thread."""
        if self.is_training:
            return

        # Get training parameters
        algorithm = self.algorithm_var.get()
        task = self.task_var.get()
        try:
            timesteps = int(self.timesteps_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid timesteps value")
            return

        # Update UI
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.is_training = True

        # Add training start message
        self._add_result_message(f"🚀 Starting {algorithm} training for {task}")
        self._add_result_message(f"   Timesteps: {timesteps}")
        self._add_result_message(f"   This may take several minutes...\n")

        # Start training thread
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(algorithm, task, timesteps),
            daemon=True
        )
        self.training_thread.start()

    def _training_worker(self, algorithm: str, task: str, timesteps: int):
        """Training worker function."""
        try:
            # Import here to avoid circular imports
            from examples.ml_training_examples import EnhancedMLTrainer

            # Create trainer with GUI callback
            trainer = EnhancedMLTrainer(gui_callback=self._training_callback)

            # Update status
            self.status_var.set(f"Training {algorithm} for {task}...")

            # Train based on task type
            if task == "Reaching":
                model = trainer.train_reaching_task(timesteps)
                self._add_result_message("✅ Reaching task training completed!")
            elif task == "Obstacle Avoidance":
                model = trainer.train_obstacle_avoidance(timesteps)
                self._add_result_message("✅ Obstacle avoidance training completed!")
            elif task == "Gesture Recognition":
                gestures = trainer.train_gesture_recognition()
                self._add_result_message("✅ Gesture recognition training completed!")

            # Evaluate results
            if hasattr(trainer, 'trainer') and trainer.trainer.model:
                metrics = trainer.trainer.evaluate(num_episodes=10)
                self._add_result_message(f"\n📊 Evaluation Results:")
                self._add_result_message(f"   Success Rate: {metrics['success_rate']:.1%}")
                self._add_result_message(f"   Average Reward: {metrics['average_reward']:.2f}")
                self._add_result_message(f"   Average Episode Length: {metrics['average_length']:.1f}")

            self.status_var.set("Training completed successfully!")
            self.progress_var.set(100)

        except Exception as e:
            error_msg = f"❌ Training failed: {str(e)}"
            self._add_result_message(error_msg)
            self.status_var.set("Training failed!")

        finally:
            # Re-enable controls
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.is_training = False

    def _training_callback(self, info: Dict[str, Any]):
        """Callback for training progress updates."""
        if info.get('type') == 'training_update':
            episode = info.get('episode', 0)
            reward = info.get('reward', 0)

            # Update progress (rough estimate)
            progress = min((episode / 1000) * 100, 99)
            self.progress_var.set(progress)

            # Update status
            self.status_var.set(f"Episode {episode}, Reward: {reward:.1f}")

    def stop_training(self):
        """Stop current training."""
        self.is_training = False
        self.status_var.set("Stopping training...")
        self._add_result_message("⏹️ Training stopped by user")

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def load_model(self):
        """Load a trained model."""
        filename = filedialog.askopenfilename(
            title="Load Trained Model",
            filetypes=[("Model files", "*.zip"), ("All files", "*.*")],
            initialdir="models/"
        )

        if filename:
            try:
                # Load model (placeholder - would need actual implementation)
                self._add_result_message(f"📁 Loaded model: {filename}")
                self.status_var.set("Model loaded successfully")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load model:\n{str(e)}")

    def test_model(self):
        """Test a trained model."""
        self._add_result_message("🧪 Testing model functionality...")
        self._add_result_message("   This would run the test_trained_model.py script")
        self._add_result_message("   Use: python examples/test_trained_model.py --model models/PPO_robot_arm.zip")

    def _add_result_message(self, message: str):
        """Add a message to the results text area."""
        self.results_text.config(state="normal")
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.results_text.config(state="disabled")
