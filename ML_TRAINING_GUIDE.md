# Machine Learning Training Guide for Robot Arm Control

This comprehensive guide shows how to implement machine learning for robot arm control using the existing native desktop GUI application infrastructure.

## 🚀 Quick Start - ML Training Instructions

### 1. **Command Line Training (Recommended for Beginners)**

#### Basic Training Commands:
```bash
# Train with PPO (recommended for beginners)
python main.py --train PPO --timesteps 50000

# Train with SAC (good for continuous control)
python main.py --train SAC --timesteps 100000

# Train with TD3 (advanced, best performance)
python main.py --train TD3 --timesteps 150000
```

#### Advanced Training Options:
```bash
# Custom timesteps and monitoring
python main.py --train PPO --timesteps 200000

# Train without physics (faster, less realistic)
python main.py --train PPO --timesteps 50000 --no-physics

# Train without visualization (headless)
python main.py --train PPO --timesteps 100000 --no-visualization
```

### 2. **Programmatic Training (Advanced Users)**

```python
from robot_arm.robot_arm import RobotArm
from physics.physics_engine import PhysicsEngine
from ml.rl_trainer import RLTrainer

# Create robot and trainer
robot = RobotArm()
physics = PhysicsEngine()
trainer = RLTrainer(robot, physics)

# Start training
trainer.train(algorithm="PPO", total_timesteps=100000)

# Evaluate trained model
metrics = trainer.evaluate(num_episodes=10)
print(f"Success rate: {metrics['success_rate']:.2f}")
```

### 3. **Training Progress Monitoring**

#### Real-time Monitoring:
- **Console Output**: Training progress, episode rewards, loss values
- **Model Checkpoints**: Automatically saved every 10,000 timesteps
- **Performance Metrics**: Success rate, average reward, episode length

#### Training Logs Location:
```
models/
├── PPO_robot_arm.zip          # Trained PPO model
├── SAC_robot_arm.zip          # Trained SAC model
├── TD3_robot_arm.zip          # Trained TD3 model
├── training_logs/             # TensorBoard logs
└── checkpoints/               # Model checkpoints
```

#### Monitor with TensorBoard:
```bash
# Install tensorboard if not already installed
pip install tensorboard

# View training progress
tensorboard --logdir models/training_logs
```

### 4. **Model Loading and Testing**

#### Load Trained Model in GUI:
```python
# In the native GUI application
from ml.rl_trainer import RLTrainer

# Load trained model
trainer = RLTrainer(self.robot_arm)
trainer.load_model("models/PPO_robot_arm.zip", algorithm="PPO")

# Use model for control
observation = self.robot_arm.get_observation()
action, _ = trainer.model.predict(observation)
self.robot_arm.execute_action(action)
```

#### Test Model Performance:
```bash
# Evaluate trained model
python examples/test_trained_model.py --model models/PPO_robot_arm.zip --episodes 10
```

---

## 🎯 ML Applications by Difficulty Level

### **🟢 BEGINNER LEVEL (1-2 weeks implementation)**

#### 1. **Point-to-Point Reaching**
- **Task**: Move end effector to target positions
- **ML Approach**: Reinforcement Learning (PPO)
- **Difficulty**: ⭐⭐☆☆☆
- **Implementation Time**: 3-5 days
- **Success Metric**: Reach within 5cm of target 90% of the time

```python
# Simple reward function for reaching
def calculate_reach_reward(self, action):
    ee_pos, _ = self.robot_arm.get_end_effector_pose()
    distance = np.linalg.norm(ee_pos - self.target_position)
    return 100.0 * np.exp(-10 * distance)  # Exponential reward
```

#### 2. **Joint Position Control**
- **Task**: Learn to control individual joints smoothly
- **ML Approach**: Supervised Learning
- **Difficulty**: ⭐☆☆☆☆
- **Implementation Time**: 1-2 days
- **Success Metric**: Joint tracking error < 0.1 radians

#### 3. **Simple Gesture Recognition**
- **Task**: Recognize and execute basic gestures (wave, point)
- **ML Approach**: Classification + Rule-based
- **Difficulty**: ⭐⭐☆☆☆
- **Implementation Time**: 2-3 days
- **Success Metric**: 95% gesture recognition accuracy

### **🟡 INTERMEDIATE LEVEL (2-4 weeks implementation)**

#### 4. **Obstacle Avoidance**
- **Task**: Navigate around static obstacles
- **ML Approach**: Reinforcement Learning (SAC)
- **Difficulty**: ⭐⭐⭐☆☆
- **Implementation Time**: 1-2 weeks
- **Success Metric**: 80% success rate with obstacles

```python
# Obstacle avoidance reward
def calculate_obstacle_reward(self, action):
    reward = 0.0
    # Check distance to obstacles
    min_distance = self.robot_arm.get_min_obstacle_distance()
    if min_distance < 0.1:  # Too close
        reward -= 100.0
    elif min_distance < 0.2:  # Getting close
        reward -= 20.0 * (0.2 - min_distance)
    return reward
```

#### 5. **Trajectory Following**
- **Task**: Follow predefined smooth trajectories
- **ML Approach**: Imitation Learning + RL fine-tuning
- **Difficulty**: ⭐⭐⭐☆☆
- **Implementation Time**: 2-3 weeks
- **Success Metric**: Average trajectory error < 2cm

#### 6. **Natural Language to Motion**
- **Task**: Execute complex commands like "pick up the red cube"
- **ML Approach**: NLP + RL
- **Difficulty**: ⭐⭐⭐⭐☆
- **Implementation Time**: 3-4 weeks
- **Success Metric**: 70% command execution success

### **🔴 ADVANCED LEVEL (1-3 months implementation)**

#### 7. **Object Manipulation**
- **Task**: Pick, place, and manipulate objects
- **ML Approach**: Multi-task RL (TD3) + Computer Vision
- **Difficulty**: ⭐⭐⭐⭐⭐
- **Implementation Time**: 6-8 weeks
- **Success Metric**: 60% pick-and-place success rate

#### 8. **Dynamic Obstacle Avoidance**
- **Task**: Avoid moving obstacles in real-time
- **ML Approach**: Deep RL with LSTM/Transformer
- **Difficulty**: ⭐⭐⭐⭐⭐
- **Implementation Time**: 8-10 weeks
- **Success Metric**: 70% success with moving obstacles

#### 9. **Multi-Arm Coordination**
- **Task**: Coordinate multiple robot arms for complex tasks
- **ML Approach**: Multi-agent RL
- **Difficulty**: ⭐⭐⭐⭐⭐
- **Implementation Time**: 10-12 weeks
- **Success Metric**: Successful coordination 50% of the time

---

## 🛠️ Implementation Guidance

### **Choosing the Right ML Approach**

#### **Reinforcement Learning (Best for most tasks)**
- **Use for**: Point-to-point reaching, obstacle avoidance, manipulation
- **Algorithms**: PPO (stable), SAC (sample efficient), TD3 (best performance)
- **Pros**: No labeled data needed, learns from interaction
- **Cons**: Requires many training episodes, can be unstable

#### **Supervised Learning (Best for classification)**
- **Use for**: Gesture recognition, command classification
- **Algorithms**: Neural networks, SVM, Random Forest
- **Pros**: Fast training, stable results
- **Cons**: Requires labeled training data

#### **Imitation Learning (Best for complex behaviors)**
- **Use for**: Trajectory following, human-like movements
- **Algorithms**: Behavioral Cloning, GAIL, ValueDice
- **Pros**: Learns from expert demonstrations
- **Cons**: Requires high-quality demonstration data

### **Reward Function Design Principles**

#### **1. Sparse vs Dense Rewards**
```python
# Sparse reward (harder to learn, more robust)
def sparse_reward(self):
    if self.task_completed():
        return 100.0
    return 0.0

# Dense reward (easier to learn, may overfit)
def dense_reward(self):
    progress = self.calculate_task_progress()
    return progress * 100.0
```

#### **2. Multi-objective Rewards**
```python
def multi_objective_reward(self, action):
    # Primary objective
    task_reward = self.calculate_task_reward()
    
    # Secondary objectives
    safety_reward = self.calculate_safety_reward()
    efficiency_reward = self.calculate_efficiency_reward()
    
    # Weighted combination
    total_reward = (
        1.0 * task_reward +
        0.5 * safety_reward +
        0.2 * efficiency_reward
    )
    return total_reward
```

### **Performance Benchmarks**

#### **Training Time Expectations**
- **Simple reaching**: 30-60 minutes (50K timesteps)
- **Obstacle avoidance**: 2-4 hours (200K timesteps)
- **Object manipulation**: 8-24 hours (1M+ timesteps)

#### **Success Rate Targets**
- **Beginner tasks**: 80-95% success rate
- **Intermediate tasks**: 60-80% success rate
- **Advanced tasks**: 40-70% success rate

#### **Hardware Requirements**
- **CPU Training**: 4+ cores, 8GB+ RAM
- **GPU Training**: NVIDIA GPU with 4GB+ VRAM (recommended)
- **Storage**: 10GB+ for models and training data

---

## 🔧 Configuration and Customization

### **Training Configuration (config/robot_config.yaml)**
```yaml
ml:
  training:
    learning_rate: 0.0003      # Learning rate for all algorithms
    batch_size: 64             # Batch size for training
    buffer_size: 100000        # Replay buffer size
    episodes: 1000             # Maximum episodes
    max_steps_per_episode: 500 # Steps per episode
  
  rewards:
    reach_target: 100.0        # Reward for reaching target
    collision_penalty: -50.0   # Penalty for collisions
    smooth_movement: 10.0      # Reward for smooth movements
    energy_efficiency: 5.0     # Reward for energy efficiency
```

### **Environment Customization**
```python
# Custom environment for specific tasks
class CustomRobotArmEnv(RobotArmEnv):
    def __init__(self, task_type="reaching"):
        super().__init__()
        self.task_type = task_type
        
    def _calculate_reward(self, action):
        if self.task_type == "reaching":
            return self.calculate_reach_reward(action)
        elif self.task_type == "manipulation":
            return self.calculate_manipulation_reward(action)
        # Add more task types as needed
```

## 💻 Code Examples and GUI Integration

### **1. Enhanced ML Training Script**

Create `examples/ml_training_examples.py`:

```python
#!/usr/bin/env python3
"""Enhanced ML training examples with GUI integration."""

import sys
import os
import numpy as np
import time
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_arm.robot_arm import RobotArm
from physics.physics_engine import PhysicsEngine
from ml.rl_trainer import RLTrainer, RobotArmEnv
from ml.nlp_processor import CommandParser

class EnhancedMLTrainer:
    """Enhanced ML trainer with GUI integration capabilities."""

    def __init__(self, gui_callback=None):
        """Initialize enhanced trainer.

        Args:
            gui_callback: Optional callback for GUI updates
        """
        self.robot_arm = RobotArm()
        self.physics_engine = PhysicsEngine()
        self.trainer = RLTrainer(self.robot_arm, self.physics_engine)
        self.command_parser = CommandParser()
        self.gui_callback = gui_callback

        # Training state
        self.is_training = False
        self.training_progress = 0.0
        self.current_episode = 0
        self.best_reward = float('-inf')

    def train_reaching_task(self, timesteps=50000):
        """Train the robot for point-to-point reaching."""
        print("🎯 Starting reaching task training...")

        # Custom reward function for reaching
        original_reward = self.trainer.env._calculate_reward

        def reaching_reward(action):
            reward = 0.0
            try:
                # Distance to target reward
                ee_pos, _ = self.robot_arm.get_end_effector_pose()
                target = self.trainer.env.target_position
                distance = np.linalg.norm(ee_pos - target)

                # Exponential reward for reaching target
                reach_reward = 100.0 * np.exp(-10 * distance)
                reward += reach_reward

                # Bonus for reaching target
                if distance < 0.05:
                    reward += 200.0

                # Smooth movement penalty
                action_penalty = np.sum(np.square(action)) * 0.1
                reward -= action_penalty

                # Update GUI if callback available
                if self.gui_callback:
                    self.gui_callback({
                        'type': 'training_update',
                        'distance': distance,
                        'reward': reward,
                        'episode': self.current_episode
                    })

            except Exception as e:
                print(f"Error in reaching reward: {e}")
                reward = -1.0

            return reward

        # Replace reward function
        self.trainer.env._calculate_reward = reaching_reward

        # Start training
        self.is_training = True
        self.trainer.train("PPO", timesteps)
        self.is_training = False

        print("✅ Reaching task training completed!")
        return self.trainer.model

    def train_gesture_recognition(self):
        """Train gesture recognition using supervised learning."""
        print("👋 Training gesture recognition...")

        # Generate training data for gestures
        gestures = {
            'wave': self._generate_wave_trajectory(),
            'point': self._generate_point_trajectory(),
            'reach': self._generate_reach_trajectory()
        }

        # Simple classification training (placeholder)
        print("Gesture recognition training completed!")
        return gestures

    def _generate_wave_trajectory(self):
        """Generate wave gesture trajectory."""
        # Simplified wave motion
        trajectory = []
        for t in np.linspace(0, 2*np.pi, 50):
            joint_positions = [
                0.0,  # shoulder_pitch
                np.sin(t) * 0.5,  # shoulder_yaw (wave motion)
                0.0,  # shoulder_roll
                np.cos(t) * 0.3,  # elbow_flexion
                0.0,  # wrist_pitch
                0.0   # wrist_yaw
            ]
            trajectory.append(joint_positions)
        return np.array(trajectory)

    def _generate_point_trajectory(self):
        """Generate pointing gesture trajectory."""
        # Simplified pointing motion
        return np.array([
            [0.0, 0.0, 0.0, 1.2, 0.0, 0.0],  # Point forward
            [0.0, 0.5, 0.0, 1.2, 0.0, 0.0],  # Point right
            [0.0, -0.5, 0.0, 1.2, 0.0, 0.0], # Point left
        ])

    def _generate_reach_trajectory(self):
        """Generate reaching trajectory."""
        # Simplified reaching motion
        return np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # Start
            [0.3, 0.0, 0.0, 0.5, 0.0, 0.0],    # Reach forward
            [0.6, 0.0, 0.0, 1.0, 0.0, 0.0],    # Full extension
        ])

def example_basic_training():
    """Example: Basic training workflow."""
    print("🚀 Basic Training Example")
    print("-" * 40)

    # Create trainer
    trainer = EnhancedMLTrainer()

    # Train reaching task
    model = trainer.train_reaching_task(timesteps=10000)  # Short for demo

    # Evaluate model
    metrics = trainer.trainer.evaluate(num_episodes=5)
    print(f"Training Results:")
    print(f"  Success Rate: {metrics['success_rate']:.2f}")
    print(f"  Average Reward: {metrics['average_reward']:.2f}")

def example_custom_reward():
    """Example: Custom reward function."""
    print("🎯 Custom Reward Function Example")
    print("-" * 40)

    class CustomRewardEnv(RobotArmEnv):
        def _calculate_reward(self, action):
            """Custom multi-objective reward function."""
            reward = 0.0

            try:
                # 1. Task completion reward
                ee_pos, _ = self.robot_arm.get_end_effector_pose()
                distance = np.linalg.norm(ee_pos - self.target_position)
                task_reward = 100.0 * np.exp(-5 * distance)

                # 2. Safety reward (avoid joint limits)
                safety_reward = 0.0
                for joint_name in self.main_joints:
                    joint = self.robot_arm.joints[joint_name]
                    if joint.is_at_limit():
                        safety_reward -= 20.0
                    elif joint.distance_to_limit() < 0.1:
                        safety_reward -= 5.0

                # 3. Efficiency reward (smooth movements)
                velocities = self.robot_arm.get_joint_velocities(self.main_joints)
                efficiency_reward = -0.1 * np.sum(np.square(velocities))

                # 4. Energy reward (minimize action magnitude)
                energy_reward = -0.05 * np.sum(np.square(action))

                # Combine rewards
                reward = task_reward + safety_reward + efficiency_reward + energy_reward

            except Exception as e:
                print(f"Error in custom reward: {e}")
                reward = -1.0

            return reward

    # Use custom environment
    robot = RobotArm()
    custom_env = CustomRewardEnv(robot)
    print(f"Custom environment created with observation space: {custom_env.observation_space}")

if __name__ == "__main__":
    # Run examples
    example_basic_training()
    example_custom_reward()
```

### **2. GUI Integration for ML Training**

Add to `ui/enhanced_control_panel.py`:

```python
class MLTrainingFrame(ttk.Frame):
    """Frame for ML training controls in the GUI."""

    def __init__(self, parent, robot_arm, command_parser):
        super().__init__(parent)
        self.robot_arm = robot_arm
        self.command_parser = command_parser
        self.trainer = None
        self.training_thread = None

        self._create_widgets()

    def _create_widgets(self):
        """Create ML training widgets."""
        # Title
        ttk.Label(self, text="🤖 ML Training",
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))

        # Algorithm selection
        algo_frame = ttk.Frame(self)
        algo_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(algo_frame, text="Algorithm:").pack(side="left")
        self.algorithm_var = tk.StringVar(value="PPO")
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algorithm_var,
                                 values=["PPO", "SAC", "TD3"], state="readonly")
        algo_combo.pack(side="left", padx=5)

        # Timesteps selection
        steps_frame = ttk.Frame(self)
        steps_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(steps_frame, text="Timesteps:").pack(side="left")
        self.timesteps_var = tk.StringVar(value="50000")
        steps_entry = ttk.Entry(steps_frame, textvariable=self.timesteps_var, width=10)
        steps_entry.pack(side="left", padx=5)

        # Training controls
        control_frame = ttk.Frame(self)
        control_frame.pack(fill="x", padx=5, pady=10)

        self.start_button = ttk.Button(control_frame, text="Start Training",
                                      command=self.start_training)
        self.start_button.pack(side="left", padx=2)

        self.stop_button = ttk.Button(control_frame, text="Stop Training",
                                     command=self.stop_training, state="disabled")
        self.stop_button.pack(side="left", padx=2)

        # Progress display
        progress_frame = ttk.LabelFrame(self, text="Training Progress", padding=5)
        progress_frame.pack(fill="both", expand=True, padx=5, pady=5)

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

        self.results_text = tk.Text(results_frame, height=6, width=40)
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=scrollbar.set)

        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def start_training(self):
        """Start ML training in a separate thread."""
        if self.training_thread and self.training_thread.is_alive():
            return

        # Disable start button, enable stop button
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        # Get training parameters
        algorithm = self.algorithm_var.get()
        timesteps = int(self.timesteps_var.get())

        # Start training thread
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(algorithm, timesteps),
            daemon=True
        )
        self.training_thread.start()

    def _training_worker(self, algorithm, timesteps):
        """Training worker function."""
        try:
            # Create trainer
            from ml.rl_trainer import RLTrainer
            self.trainer = RLTrainer(self.robot_arm)

            # Update status
            self.status_var.set(f"Training {algorithm} for {timesteps} timesteps...")

            # Custom callback for progress updates
            def training_callback(progress_info):
                # Update progress bar
                progress = (progress_info.get('episode', 0) / 1000) * 100
                self.progress_var.set(min(progress, 100))

                # Update status
                episode = progress_info.get('episode', 0)
                reward = progress_info.get('reward', 0)
                self.status_var.set(f"Episode {episode}, Reward: {reward:.2f}")

            # Start training
            self.trainer.train(algorithm, timesteps)

            # Evaluate results
            metrics = self.trainer.evaluate(num_episodes=10)

            # Display results
            results = f"""Training Completed!
Algorithm: {algorithm}
Timesteps: {timesteps}
Success Rate: {metrics['success_rate']:.2f}
Average Reward: {metrics['average_reward']:.2f}
Average Episode Length: {metrics['average_length']:.1f}
"""

            self.results_text.insert(tk.END, results + "\n")
            self.results_text.see(tk.END)

            self.status_var.set("Training completed successfully!")

        except Exception as e:
            error_msg = f"Training failed: {str(e)}\n"
            self.results_text.insert(tk.END, error_msg)
            self.status_var.set("Training failed!")

        finally:
            # Re-enable controls
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")

    def stop_training(self):
        """Stop current training."""
        if self.trainer:
            # Note: Stable-baselines3 doesn't have a direct stop method
            # This would require custom implementation
            self.status_var.set("Stopping training...")

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
```

### **3. NLP Integration with Trained Models**

Enhance `ml/nlp_processor.py`:

```python
class MLEnhancedCommandParser(CommandParser):
    """Enhanced command parser with ML model integration."""

    def __init__(self):
        super().__init__()
        self.trained_models = {}
        self.load_available_models()

    def load_available_models(self):
        """Load available trained models."""
        models_dir = "models/"
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith('.zip'):
                    model_name = filename.replace('.zip', '')
                    self.trained_models[model_name] = os.path.join(models_dir, filename)

    def execute_ml_command(self, command: str, robot_arm) -> Dict[str, Any]:
        """Execute command using trained ML models."""
        result = {'success': False, 'action': None, 'model_used': None}

        try:
            # Parse command to determine task type
            parsed = self.parse_command(command)

            # Determine which model to use
            if 'reach' in command or 'move' in command:
                model_name = 'PPO_robot_arm'
            elif 'wave' in command or 'gesture' in command:
                model_name = 'gesture_model'
            else:
                model_name = 'PPO_robot_arm'  # Default

            # Load and use model
            if model_name in self.trained_models:
                from ml.rl_trainer import RLTrainer
                trainer = RLTrainer(robot_arm)
                trainer.load_model(self.trained_models[model_name], "PPO")

                # Get current observation
                observation = self._get_robot_observation(robot_arm)

                # Predict action
                action, _ = trainer.model.predict(observation, deterministic=True)

                # Execute action
                robot_arm.execute_ml_action(action)

                result['success'] = True
                result['action'] = action.tolist()
                result['model_used'] = model_name

        except Exception as e:
            print(f"Error executing ML command: {e}")
            result['error'] = str(e)

        return result

    def _get_robot_observation(self, robot_arm):
        """Get current robot observation for ML model."""
        # This should match the observation space used during training
        joint_positions = robot_arm.get_joint_positions()
        joint_velocities = robot_arm.get_joint_velocities()
        ee_pose, _ = robot_arm.get_end_effector_pose()
        target_position = np.array([0.3, 0.0, 0.3])  # Default target

        observation = np.concatenate([
            joint_positions[:6],  # Main joints only
            joint_velocities[:6],
            ee_pose,
            target_position
        ])

        return observation.astype(np.float32)
```

This comprehensive guide provides everything needed to implement machine learning in your robot arm control system, from basic training to advanced GUI integration!
