# Robot Arm Simulation - Native Desktop Application

A comprehensive native Python desktop application for robot arm simulation featuring an anthropomorphic robotic arm with machine learning integration, real-time 3D visualization, and natural language command processing.

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

### Native Desktop Interface
- **Native Python GUI**: Cross-platform desktop application using tkinter
- **Real-time 3D Visualization**: Optional OpenGL-based 3D rendering with camera controls
- **Enhanced Joint Controls**: Interactive sliders with real-time feedback and status indicators
- **Natural Language Interface**: Advanced command processing with history and presets
- **Comprehensive Status Monitoring**: Real-time joint states, end effector pose, and system metrics
- **Offline Operation**: No web server required, fully standalone application

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenGL support
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone or download the project
cd robot-arm-simulation

# Install required packages
pip install -r requirements.txt

# Or install manually:
pip install numpy scipy matplotlib
pip install torch transformers stable-baselines3 gymnasium
pip install opencv-python Pillow pyyaml tqdm

# Optional: For 3D visualization (may not work on all systems)
pip install PyOpenGL PyOpenGL-accelerate glfw
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

# Or manually with conda:
conda create -n robot-sim -c conda-forge python=3.11 pybullet
conda activate robot-sim
conda install numpy scipy matplotlib pytorch transformers -c pytorch
pip install stable-baselines3 gymnasium opencv-python Pillow pyyaml tqdm
```

## Quick Start

### 1. Native Desktop GUI (Recommended)
```bash
# Launch the native desktop application with compatibility checks
python run_native_gui.py

# Or run directly:
python native_desktop_gui.py
```

### 2. Command Line Interface
```bash
# Start the full simulation with command line interface
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

### Native Desktop GUI

The native desktop application provides the best experience with full functionality and performance:

#### Features
- **Real-time 3D Visualization**: Optional OpenGL-based robot rendering with smooth animations
- **Enhanced Controls**: Native GUI controls with real-time feedback and status indicators
- **Offline Operation**: No web server required, fully standalone application
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Native Integration**: System menus, keyboard shortcuts, and file dialogs

#### Interface Layout
- **Main Window**: Split-pane layout with control panel and status monitoring
- **Control Panel**: Tabbed interface with:
  - Joint control sliders with real-time position vs target feedback
  - Natural language command input with history and presets
- **Status Panel**: Comprehensive monitoring with:
  - End effector pose (position and orientation)
  - System status and performance metrics
  - Joint status summary with color-coded indicators
- **3D Visualization Window**: Optional separate window for 3D robot visualization

#### 3D Visualization Controls (if available)
- **Left Mouse + Drag**: Orbit camera around robot
- **Right Mouse + Drag**: Pan camera view
- **Mouse Wheel**: Zoom in/out
- **Keyboard Shortcuts**: G (grid), F (frames), J (joints), W (workspace)

### Desktop GUI Control Panel

The application provides a comprehensive tabbed interface:

#### 1. Joint Control Tab
- **Enhanced Joint Sliders**: Control each joint with real-time position vs target indicators
- **Status Indicators**: Color-coded joint limit warnings and movement status
- **Velocity and Force Display**: Real-time joint velocity and force feedback
- **Quick Controls**: Reset all joints, stop all movements, individual joint controls

#### 2. Commands Tab
- **Natural Language Input**: Advanced command processing with confidence scoring
- **Predefined Commands**: Quick action buttons for common gestures and movements
- **Command History**: Complete history with execution status and timing
- **Real-time Feedback**: Success/failure indicators with detailed error messages
- **Supported Commands**:
  - Movement: "move forward", "go up", "reach position X,Y,Z"
  - Gestures: "wave hello", "point up", "stretch arms"
  - Control: "stop", "reset to home", "relax"

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
├── ui/                    # Native desktop user interface
│   ├── enhanced_control_panel.py  # Enhanced joint and command controls
│   ├── robot_status_panel.py      # Comprehensive status monitoring
│   └── visualization_window.py    # 3D visualization window
├── config/                # Configuration files
│   └── robot_config.yaml  # Robot configuration
├── native_desktop_gui.py    # Main native desktop application
├── run_native_gui.py         # Safe launcher with compatibility checks
├── test_native_gui.py        # Test suite for native GUI
└── main.py                   # Command line interface
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

1. **Native GUI Won't Start**:
   - Check Python version (requires 3.8+)
   - Install tkinter: `sudo apt-get install python3-tk` (Linux)
   - Use launcher: `python run_native_gui.py` for automatic checks

2. **3D Visualization Not Working**:
   - Install OpenGL: `pip install PyOpenGL PyOpenGL-accelerate glfw`
   - Update graphics drivers
   - Application continues with 2D controls if 3D fails

3. **macOS Compatibility Issues**:
   - ✅ **Apple Silicon (M1/M2/M3)**: Fully supported with automatic tkinter compatibility fixes
   - ✅ **GUI Compatibility**: Automatic detection and graceful fallback for unsupported color options
   - ✅ **No more tkinter errors**: Fixed "-bg" option errors on newer macOS versions
   - Use compatibility mode (launcher detects automatically)
   - Some OpenGL features may not work on Apple Silicon
   - Application provides fallback options

4. **PyBullet Import Error**: Install with `pip install pybullet`
5. **Transformer Model Download**: Requires internet connection on first run

### Alternative Interfaces

If the native GUI cannot start, use these alternatives:
```bash
# Command line interface with demo
python main.py --demo

# Single command execution
python main.py --command "wave hello"

# Basic functionality test
python test_native_gui.py
```

### Performance Optimization

- Use `--no-physics` for faster simulation
- Disable 3D visualization if not needed
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
