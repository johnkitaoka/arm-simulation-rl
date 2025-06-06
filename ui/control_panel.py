"""GUI control panel for robot arm simulation."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import threading
import time

from robot_arm.robot_arm import RobotArm
from ml.nlp_processor import CommandParser, CommandHistory
from ml.rl_trainer import RLTrainer
from core.config import config


class JointControlFrame(ttk.Frame):
    """Frame for manual joint control."""
    
    def __init__(self, parent, robot_arm: RobotArm):
        super().__init__(parent)
        self.robot_arm = robot_arm
        self.joint_vars = {}
        self.joint_scales = {}
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create joint control widgets."""
        ttk.Label(self, text="Joint Control", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Create sliders for each joint
        for joint_name, joint in self.robot_arm.joints.items():
            frame = ttk.Frame(self)
            frame.pack(fill="x", padx=5, pady=2)
            
            # Joint label
            ttk.Label(frame, text=joint_name, width=15).pack(side="left")
            
            # Joint value variable
            var = tk.DoubleVar(value=joint.position)
            self.joint_vars[joint_name] = var
            
            # Joint slider
            scale = ttk.Scale(
                frame,
                from_=joint.limits[0],
                to=joint.limits[1],
                variable=var,
                orient="horizontal",
                command=lambda val, name=joint_name: self._on_joint_change(name, val)
            )
            scale.pack(side="left", fill="x", expand=True, padx=5)
            self.joint_scales[joint_name] = scale
            
            # Value label
            value_label = ttk.Label(frame, text=f"{joint.position:.3f}", width=8)
            value_label.pack(side="right")
            
            # Update value label when slider changes
            var.trace("w", lambda *args, label=value_label, v=var: 
                     label.config(text=f"{v.get():.3f}"))
    
    def _on_joint_change(self, joint_name: str, value: str):
        """Handle joint slider change."""
        try:
            val = float(value)
            self.robot_arm.joints[joint_name].target_position = val
        except ValueError:
            pass
    
    def update_from_robot(self):
        """Update sliders from current robot state."""
        for joint_name, var in self.joint_vars.items():
            current_pos = self.robot_arm.joints[joint_name].position
            var.set(current_pos)
    
    def reset_joints(self):
        """Reset all joints to home position."""
        for joint_name, var in self.joint_vars.items():
            var.set(0.0)
            self.robot_arm.joints[joint_name].target_position = 0.0


class CommandFrame(ttk.Frame):
    """Frame for natural language command input."""
    
    def __init__(self, parent, robot_arm: RobotArm):
        super().__init__(parent)
        self.robot_arm = robot_arm
        self.command_parser = CommandParser()
        self.command_history = CommandHistory()
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create command input widgets."""
        ttk.Label(self, text="Natural Language Commands", 
                 font=("Arial", 12, "bold")).pack(pady=5)
        
        # Command input
        input_frame = ttk.Frame(self)
        input_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(input_frame, text="Command:").pack(side="left")
        
        self.command_var = tk.StringVar()
        self.command_entry = ttk.Entry(input_frame, textvariable=self.command_var)
        self.command_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.command_entry.bind("<Return>", lambda e: self.execute_command())
        
        ttk.Button(input_frame, text="Execute", 
                  command=self.execute_command).pack(side="right")
        
        # Command history
        history_frame = ttk.Frame(self)
        history_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ttk.Label(history_frame, text="Command History:").pack(anchor="w")
        
        # History listbox with scrollbar
        list_frame = ttk.Frame(history_frame)
        list_frame.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.history_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.history_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.history_listbox.yview)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var).pack(pady=5)
    
    def execute_command(self):
        """Execute the entered command."""
        command = self.command_var.get().strip()
        if not command:
            return
        
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
            status = "Success" if success else "Failed"
            confidence = parsed.get('confidence', 0.0)
            history_entry = f"{command} -> {action['type']} ({status}, {confidence:.2f})"
            
            self.history_listbox.insert(tk.END, history_entry)
            self.history_listbox.see(tk.END)
            
            self.status_var.set(f"Command executed: {status}")
            self.command_var.set("")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Command Error", str(e))
    
    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute a robot action.
        
        Args:
            action: Action dictionary from command parser
            
        Returns:
            True if successful
        """
        if not action['success']:
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
                        'up': [0, 0, 1],
                        'down': [0, 0, -1],
                        'left': [0, 1, 0],
                        'right': [0, -1, 0],
                        'forward': [1, 0, 0],
                        'backward': [-1, 0, 0]
                    }
                    
                    if direction in direction_vectors:
                        dir_vec = np.array(direction_vectors[direction])
                        target_pos = current_pos + dir_vec * distance
                        return self.robot_arm.move_to_pose(target_pos)
            
            elif action_type == 'reset_to_home':
                self.robot_arm.reset_to_home()
                return True
            
            elif action_type == 'stop':
                self.robot_arm.emergency_stop()
                return True
            
            elif action_type == 'gesture':
                gesture_type = params.get('gesture_type', 'wave')
                return self._execute_gesture(gesture_type, params)
            
            # Add more action types as needed
            
        except Exception as e:
            print(f"Error executing action: {e}")
            return False
        
        return False
    
    def _execute_gesture(self, gesture_type: str, params: Dict) -> bool:
        """Execute a gesture.
        
        Args:
            gesture_type: Type of gesture
            params: Gesture parameters
            
        Returns:
            True if successful
        """
        if gesture_type == 'wave':
            # Simple wave gesture
            wave_positions = [
                [0.3, 0.2, 0.4],
                [0.3, -0.2, 0.4],
                [0.3, 0.2, 0.4],
                [0.3, 0.0, 0.3]
            ]
            
            for pos in wave_positions:
                self.robot_arm.move_to_pose(np.array(pos))
                time.sleep(0.5)
            
            return True
        
        elif gesture_type == 'point':
            target = params.get('target', [0.5, 0.0, 0.3])
            return self.robot_arm.move_to_pose(np.array(target))
        
        return False


class TrainingFrame(ttk.Frame):
    """Frame for ML training controls."""
    
    def __init__(self, parent, robot_arm: RobotArm):
        super().__init__(parent)
        self.robot_arm = robot_arm
        self.trainer = None
        self.training_thread = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create training control widgets."""
        ttk.Label(self, text="Machine Learning Training", 
                 font=("Arial", 12, "bold")).pack(pady=5)
        
        # Algorithm selection
        algo_frame = ttk.Frame(self)
        algo_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(algo_frame, text="Algorithm:").pack(side="left")
        self.algorithm_var = tk.StringVar(value="PPO")
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algorithm_var,
                                 values=["PPO", "SAC", "TD3"], state="readonly")
        algo_combo.pack(side="left", padx=5)
        
        # Training parameters
        params_frame = ttk.Frame(self)
        params_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(params_frame, text="Timesteps:").pack(side="left")
        self.timesteps_var = tk.StringVar(value="100000")
        ttk.Entry(params_frame, textvariable=self.timesteps_var, width=10).pack(side="left", padx=5)
        
        # Training controls
        control_frame = ttk.Frame(self)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        self.train_button = ttk.Button(control_frame, text="Start Training",
                                      command=self.start_training)
        self.train_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Training",
                                     command=self.stop_training, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        
        ttk.Button(control_frame, text="Load Model",
                  command=self.load_model).pack(side="left", padx=5)
        
        ttk.Button(control_frame, text="Save Model",
                  command=self.save_model).pack(side="left", padx=5)
        
        ttk.Button(control_frame, text="Evaluate",
                  command=self.evaluate_model).pack(side="left", padx=5)
        
        # Progress and status
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.progress_var).pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(self, mode='indeterminate')
        self.progress_bar.pack(fill="x", padx=5, pady=5)
    
    def start_training(self):
        """Start training in a separate thread."""
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Training", "Training is already in progress")
            return
        
        try:
            timesteps = int(self.timesteps_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid timesteps value")
            return
        
        self.train_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.progress_bar.start()
        self.progress_var.set("Initializing training...")
        
        # Start training in separate thread
        self.training_thread = threading.Thread(
            target=self._train_worker,
            args=(self.algorithm_var.get(), timesteps)
        )
        self.training_thread.start()
    
    def _train_worker(self, algorithm: str, timesteps: int):
        """Worker function for training."""
        try:
            if self.trainer is None:
                self.trainer = RLTrainer(self.robot_arm)
            
            self.progress_var.set(f"Training with {algorithm}...")
            self.trainer.train(algorithm, timesteps)
            self.progress_var.set("Training completed successfully")
            
        except Exception as e:
            self.progress_var.set(f"Training failed: {str(e)}")
            messagebox.showerror("Training Error", str(e))
        
        finally:
            self.progress_bar.stop()
            self.train_button.config(state="normal")
            self.stop_button.config(state="disabled")
    
    def stop_training(self):
        """Stop training (placeholder - would need proper implementation)."""
        self.progress_var.set("Stopping training...")
        # In a real implementation, you would signal the training to stop
        messagebox.showinfo("Training", "Training stop requested")
    
    def load_model(self):
        """Load a trained model."""
        filepath = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Model files", "*.zip"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                if self.trainer is None:
                    self.trainer = RLTrainer(self.robot_arm)
                
                self.trainer.load_model(filepath, self.algorithm_var.get())
                self.progress_var.set(f"Model loaded: {filepath}")
                messagebox.showinfo("Success", "Model loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def save_model(self):
        """Save the current model."""
        if self.trainer is None or self.trainer.model is None:
            messagebox.showwarning("Warning", "No model to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".zip",
            filetypes=[("Model files", "*.zip"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.trainer.model.save(filepath)
                self.progress_var.set(f"Model saved: {filepath}")
                messagebox.showinfo("Success", "Model saved successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def evaluate_model(self):
        """Evaluate the current model."""
        if self.trainer is None or self.trainer.model is None:
            messagebox.showwarning("Warning", "No model to evaluate")
            return
        
        self.progress_var.set("Evaluating model...")
        self.progress_bar.start()
        
        # Run evaluation in separate thread
        eval_thread = threading.Thread(target=self._evaluate_worker)
        eval_thread.start()
    
    def _evaluate_worker(self):
        """Worker function for evaluation."""
        try:
            metrics = self.trainer.evaluate(num_episodes=5)
            
            result_text = (
                f"Evaluation Results:\n"
                f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}\n"
                f"Mean Episode Length: {metrics['mean_length']:.1f}\n"
                f"Success Rate: {metrics['success_rate']:.2%}"
            )
            
            self.progress_var.set("Evaluation completed")
            messagebox.showinfo("Evaluation Results", result_text)
            
        except Exception as e:
            self.progress_var.set(f"Evaluation failed: {str(e)}")
            messagebox.showerror("Evaluation Error", str(e))
        
        finally:
            self.progress_bar.stop()


class ControlPanel:
    """Main control panel for robot arm simulation."""
    
    def __init__(self, robot_arm: RobotArm):
        """Initialize the control panel.
        
        Args:
            robot_arm: Robot arm instance to control
        """
        self.robot_arm = robot_arm
        self.root = tk.Tk()
        self.root.title("Robot Arm Control Panel")
        self.root.geometry("800x600")
        
        self._create_widgets()
        self._setup_update_loop()
    
    def _create_widgets(self):
        """Create the main UI widgets."""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Joint control tab
        joint_frame = ttk.Frame(notebook)
        notebook.add(joint_frame, text="Joint Control")
        self.joint_control = JointControlFrame(joint_frame, self.robot_arm)
        self.joint_control.pack(fill="both", expand=True)
        
        # Command tab
        command_frame = ttk.Frame(notebook)
        notebook.add(command_frame, text="Commands")
        self.command_control = CommandFrame(command_frame, self.robot_arm)
        self.command_control.pack(fill="both", expand=True)
        
        # Training tab
        training_frame = ttk.Frame(notebook)
        notebook.add(training_frame, text="Training")
        self.training_control = TrainingFrame(training_frame, self.robot_arm)
        self.training_control.pack(fill="both", expand=True)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", side="bottom")
        
        self.status_var = tk.StringVar(value="Robot arm control panel ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left", padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(status_frame)
        button_frame.pack(side="right", padx=5)
        
        ttk.Button(button_frame, text="Emergency Stop",
                  command=self.emergency_stop).pack(side="right", padx=2)
        
        ttk.Button(button_frame, text="Reset",
                  command=self.reset_robot).pack(side="right", padx=2)
        
        ttk.Button(button_frame, text="Enable",
                  command=self.enable_robot).pack(side="right", padx=2)
    
    def _setup_update_loop(self):
        """Setup the UI update loop."""
        self._update_ui()
    
    def _update_ui(self):
        """Update UI with current robot state."""
        try:
            # Update joint control sliders
            self.joint_control.update_from_robot()
            
            # Update status
            joint_info = self.robot_arm.get_joint_info()
            num_at_limits = sum(1 for info in joint_info.values() if info['at_limit'])
            
            status = f"Robot: {'Enabled' if self.robot_arm.is_enabled else 'Disabled'}"
            if num_at_limits > 0:
                status += f" | {num_at_limits} joints at limits"
            
            self.status_var.set(status)
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
        
        # Schedule next update
        self.root.after(100, self._update_ui)
    
    def emergency_stop(self):
        """Emergency stop the robot."""
        self.robot_arm.emergency_stop()
        messagebox.showwarning("Emergency Stop", "Robot arm has been stopped")
    
    def reset_robot(self):
        """Reset robot to home position."""
        self.robot_arm.reset_to_home()
        self.joint_control.reset_joints()
    
    def enable_robot(self):
        """Enable/disable robot."""
        if self.robot_arm.is_enabled:
            self.robot_arm.disable()
        else:
            self.robot_arm.enable()
    
    def run(self):
        """Run the control panel."""
        self.root.mainloop()
    
    def destroy(self):
        """Cleanup and destroy the control panel."""
        self.root.destroy()
