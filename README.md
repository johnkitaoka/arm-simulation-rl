# 3D Robot Arm Simulation with Machine Learning

A comprehensive Python-based 3D robot arm simulation application featuring an anthropomorphic robotic arm with machine learning integration for natural language command processing and reinforcement learning training.

## Features

### Robot Arm Specifications
- **Fully anthropomorphic design** with 5 main articulation points:
  - Shoulder joint (3 DOF: pitch, yaw, roll)
  - Elbow joint (1 DOF: flexion/extension)
  - Wrist joint (2 DOF: pitch, yaw)
  - Hand with 4 digits: 1 thumb + 3 fingers
  - Each finger has 3 joints (metacarpal, proximal, distal)
  - Thumb has 2 joints (metacarpal, interphalangeal)

### 3D Visualization
- Real-time 3D visualization using OpenGL
- Interactive camera controls (rotate, zoom, pan)
- Joint angle and position display
- Coordinate frames for each joint
- Smooth animation of arm movements

### Machine Learning Integration
- **Natural Language Processing**: Accepts commands like "reach for the red ball", "wave hello", "point at the target"
- **Reinforcement Learning**: PPO, SAC, and TD3 algorithms for training
- **Visual Input Support**: Framework for camera feed integration
- **Reward System**: Configurable reward functions for training
- **Model Persistence**: Save and load trained models

### Technical Implementation
- **Physics Simulation**: PyBullet integration for realistic physics
- **Forward/Inverse Kinematics**: Complete kinematic chain calculations
- **Collision Detection**: Self-collision and environment collision checking
- **Modular Architecture**: Independent simulation, ML, and visualization components

### User Interface
- **3D Web Interface**: Modern browser-based 3D visualization (Apple Silicon optimized)
- **GUI Control Panel**: Manual joint control sliders
- **Natural Language Interface**: Text input for commands
- **Training Controls**: Start/stop/monitor ML training
- **Real-time Monitoring**: Joint states and system status

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenGL support
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone or download the project
cd robot-simulation

# Install required packages
pip install -r requirements.txt

# For 3D web interface (recommended for Apple Silicon):
python install_3d_deps.py

# Or install manually:
pip install numpy scipy PyOpenGL PyOpenGL-accelerate glfw moderngl
pip install pybullet torch transformers stable-baselines3 gymnasium
pip install opencv-python Pillow pyyaml tqdm matplotlib
pip install flask flask-socketio python-socketio
```

### Alternative Installation
```bash
# Install as a package
pip install -e .
```

### Apple Silicon (M1/M2) Setup
For optimal performance on Apple Silicon Macs:
```bash
# Use the provided setup script
./setup_apple_silicon.sh

# Or manually with conda + uv:
conda create -n robot-sim -c conda-forge python=3.11 pybullet
conda activate robot-sim
uv pip install -r requirements.txt
python install_3d_deps.py
```

## Quick Start

### 1. 3D Web Interface (Recommended)
```bash
# Start the modern 3D web interface
python run_3d_gui.py

# Or run directly:
python web_3d_interface.py
```
Then open your browser to: http://localhost:5000

### 2. Traditional GUI (Desktop)
```bash
# Start the full simulation with GUI
python main.py

# Start without physics simulation
python main.py --no-physics

# Start headless (no GUI or visualization)
python main.py --no-gui --no-visualization
```

### 3. Natural Language Commands
```bash
# Execute a single command
python main.py --command "move to position 0.3, 0.0, 0.4"
python main.py --command "wave hello"
python main.py --command "point at the target"
python main.py --command "reset to home"
```

### 4. Movement Demonstration
```bash
# Run a pre-programmed movement demo
python main.py --demo
```

### 5. Machine Learning Training
```bash
# Train with PPO algorithm
python main.py --train PPO --timesteps 100000

# Train with SAC algorithm
python main.py --train SAC --timesteps 50000

# Train with TD3 algorithm
python main.py --train TD3 --timesteps 200000
```

## Usage Guide

### 3D Web Interface

The modern web-based interface provides the best experience, especially on Apple Silicon Macs:

#### Features
- **Real-time 3D Visualization**: Interactive Three.js-based robot rendering
- **Mouse Controls**: Drag to orbit, scroll to zoom, intuitive camera movement
- **Live Updates**: Real-time robot state updates via WebSocket
- **Cross-platform**: Works on any modern browser
- **Responsive Design**: Adapts to different screen sizes

#### Interface Layout
- **Left Panel**: 3D viewport with interactive robot visualization
- **Right Panel**: Control interface with tabs for:
  - Status monitoring
  - Joint controls with sliders
  - Natural language command input
  - Command history and logging

#### 3D Viewport Controls
- **Left Mouse + Drag**: Orbit camera around robot
- **Mouse Wheel**: Zoom in/out
- **Auto-rotation**: Optional automatic camera rotation

### Traditional GUI Control Panel

The application provides a tabbed interface with three main sections:

#### 1. Joint Control Tab
- Manual control sliders for each joint
- Real-time joint position display
- Reset to home position button
- Joint limit indicators

#### 2. Commands Tab
- Text input for natural language commands
- Command history display
- Execution status feedback
- Supported commands:
  - Movement: "move forward", "go up", "reach position X,Y,Z"
  - Gestures: "wave", "point at target"
  - Control: "stop", "reset", "home position"

#### 3. Training Tab
- Algorithm selection (PPO, SAC, TD3)
- Training parameter configuration
- Start/stop training controls
- Model save/load functionality
- Evaluation metrics display

### 3D Visualization Controls

- **Mouse Left Click + Drag**: Orbit camera around target
- **Mouse Right Click + Drag**: Pan camera
- **Mouse Scroll**: Zoom in/out
- **Keyboard Shortcuts**:
  - `F`: Toggle coordinate frames
  - `J`: Toggle joint spheres
  - `W`: Toggle wireframe mode
  - `ESC`: Exit application

### Configuration

Edit `config/robot_config.yaml` to customize:

- Joint limits and ranges
- Link lengths and masses
- Simulation parameters
- Visualization settings
- ML training parameters
- Reward function weights

### Natural Language Commands

The system understands various command patterns:

#### Movement Commands
- "move to position 0.3, 0.0, 0.4"
- "go forward"
- "reach up"
- "move left"
- "extend arm"

#### Object Interaction
- "grab the red ball"
- "pick up the cube"
- "release object"
- "drop item"

#### Gestures
- "wave hello"
- "point at target"
- "gesture greeting"

#### Control Commands
- "stop"
- "emergency stop"
- "reset to home"
- "return to start"

## Architecture

### Core Components

```
robot-simulation/
├── core/                   # Core utilities and configuration
│   ├── config.py          # Configuration management
│   └── math_utils.py      # Mathematical utilities
├── robot_arm/             # Robot arm model and kinematics
│   ├── robot_arm.py       # Main robot arm class
│   ├── joint.py           # Joint definitions
│   ├── link.py            # Link definitions
│   └── kinematics.py      # Forward/inverse kinematics
├── physics/               # Physics simulation
│   └── physics_engine.py  # PyBullet integration
├── visualization/         # 3D rendering
│   └── renderer.py        # OpenGL renderer
├── ml/                    # Machine learning components
│   ├── nlp_processor.py   # Natural language processing
│   └── rl_trainer.py      # Reinforcement learning
├── ui/                    # User interface
│   └── control_panel.py   # GUI control panel
├── config/                # Configuration files
│   └── robot_config.yaml  # Robot configuration
└── main.py               # Main application
```

### Key Classes

- **RobotArm**: Main robot arm with full anthropomorphic design
- **Joint**: Individual joint with limits, control, and state
- **Link**: Physical links with collision geometry
- **ForwardKinematics/InverseKinematics**: Kinematic solvers
- **PhysicsEngine**: PyBullet physics integration
- **Renderer**: OpenGL 3D visualization
- **CommandParser**: Natural language processing
- **RLTrainer**: Reinforcement learning training
- **ControlPanel**: GUI interface

## Development

### Adding New Commands

1. Add keywords to `ml/nlp_processor.py`:
```python
self.action_keywords['new_action'] = ['keyword1', 'keyword2']
```

2. Implement action in `command_to_robot_action()`:
```python
elif cmd_action == 'new_action':
    action['type'] = 'custom_action'
    action['parameters'] = {...}
    action['success'] = True
```

3. Handle execution in UI or main application.

### Custom Reward Functions

Modify `ml/rl_trainer.py` in the `_calculate_reward()` method:

```python
def _calculate_reward(self, action):
    reward = 0.0

    # Add custom reward components
    custom_reward = self._calculate_custom_reward()
    reward += custom_reward

    return reward
```

### Extending the Robot Model

1. Add new joints in `robot_arm.py`:
```python
self.joints['new_joint'] = Joint(
    'new_joint', JointType.REVOLUTE,
    axis=np.array([0, 1, 0]),
    limits=[-1.57, 1.57]
)
```

2. Update kinematics chain and visualization.

## Troubleshooting

### Common Issues

1. **OpenGL Errors**: Ensure graphics drivers are up to date
2. **PyBullet Import Error**: Install with `pip install pybullet`
3. **GLFW Window Creation Failed**: Install GLFW system libraries
4. **Transformer Model Download**: Requires internet connection on first run

### Performance Optimization

- Use `--no-physics` for faster simulation
- Reduce visualization quality in config
- Use GPU acceleration for ML training
- Adjust simulation timestep in config

### Debug Mode

Run with Python's debug flag for detailed error information:
```bash
python -u main.py --command "test command"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- PyBullet for physics simulation
- Stable-Baselines3 for reinforcement learning
- Transformers library for NLP
- OpenGL and GLFW for 3D visualization
