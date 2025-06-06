#!/usr/bin/env python3
"""Enhanced 3D web interface for robot arm simulation with Three.js visualization."""

import sys
import os
import json
import threading
import time
from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import dependencies with graceful fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️  NumPy not available - using fallback implementations")
    NUMPY_AVAILABLE = False
    # Create a minimal numpy-like interface
    class np:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def zeros(shape):
            if isinstance(shape, int):
                return [0.0] * shape
            return [[0.0] * shape[1] for _ in range(shape[0])]

try:
    from robot_arm.robot_arm import RobotArm
    ROBOT_ARM_AVAILABLE = True
except ImportError:
    print("⚠️  Robot arm simulation not available - using mock implementation")
    ROBOT_ARM_AVAILABLE = False

    # Mock robot arm for demonstration
    class RobotArm:
        def __init__(self):
            self.joints = {
                'shoulder_pitch': MockJoint('shoulder_pitch', -1.57, 1.57),
                'shoulder_yaw': MockJoint('shoulder_yaw', -1.57, 1.57),
                'shoulder_roll': MockJoint('shoulder_roll', -1.57, 1.57),
                'elbow': MockJoint('elbow', 0, 2.0),
                'wrist_pitch': MockJoint('wrist_pitch', -1.57, 1.57),
                'wrist_yaw': MockJoint('wrist_yaw', -1.57, 1.57),
            }
            self.is_enabled = True
            self.fk_solver = MockFKSolver(self)

        def update(self, dt):
            # Update joint positions towards targets
            for joint in self.joints.values():
                joint.update(dt)

        def get_end_effector_pose(self):
            # Calculate end effector position from forward kinematics
            if self.fk_solver:
                positions = self.fk_solver.get_link_positions()
                if positions:
                    # Return the last link position as end effector
                    return positions[-1], [0, 0, 0, 1]
            # Fallback position
            return [0.3, 0.0, 0.4], [0, 0, 0, 1]

        def move_to_pose(self, position):
            return True

        def reset_to_home(self):
            for joint in self.joints.values():
                joint.target_position = 0.0

        def emergency_stop(self):
            pass

    class MockJoint:
        def __init__(self, name, min_limit, max_limit):
            self.name = name
            self.position = 0.0
            self.target_position = 0.0
            self.limits = [min_limit, max_limit]

        def update(self, dt):
            # Simple position interpolation
            diff = self.target_position - self.position
            self.position += diff * 0.1

    class MockFKSolver:
        def __init__(self, robot_arm):
            self.robot_arm = robot_arm
            self.link_lengths = [0.1, 0.15, 0.12, 0.08]  # Link lengths for each segment

        def update(self):
            pass

        def get_link_positions(self):
            # Simple, stable forward kinematics - build arm segment by segment
            joints = self.robot_arm.joints

            # Get joint angles (use small values to keep visualization reasonable)
            shoulder_pitch = joints['shoulder_pitch'].position * 0.5  # Scale down for stability
            shoulder_yaw = joints['shoulder_yaw'].position * 0.5
            elbow = joints['elbow'].position * 0.5
            wrist_pitch = joints['wrist_pitch'].position * 0.3

            import math

            positions = []

            # Start at base
            x, y, z = 0.0, 0.0, 0.0

            # Base to shoulder (vertical)
            z += self.link_lengths[0]
            positions.append([x, y, z])

            # Shoulder to elbow (affected by shoulder pitch and yaw)
            upper_arm_length = self.link_lengths[1]
            # Simple 2D rotation in XZ plane for pitch, then rotate around Z for yaw
            x += upper_arm_length * math.sin(shoulder_pitch) * math.cos(shoulder_yaw)
            y += upper_arm_length * math.sin(shoulder_pitch) * math.sin(shoulder_yaw)
            z += upper_arm_length * math.cos(shoulder_pitch)
            positions.append([x, y, z])

            # Elbow to wrist (affected by elbow angle)
            forearm_length = self.link_lengths[2]
            # Continue in the direction established by shoulder, then bend at elbow
            elbow_direction_x = math.sin(shoulder_pitch + elbow) * math.cos(shoulder_yaw)
            elbow_direction_y = math.sin(shoulder_pitch + elbow) * math.sin(shoulder_yaw)
            elbow_direction_z = math.cos(shoulder_pitch + elbow)

            x += forearm_length * elbow_direction_x
            y += forearm_length * elbow_direction_y
            z += forearm_length * elbow_direction_z
            positions.append([x, y, z])

            # Wrist to end effector (affected by wrist pitch)
            hand_length = self.link_lengths[3]
            # Continue in forearm direction, then bend at wrist
            wrist_direction_x = math.sin(shoulder_pitch + elbow + wrist_pitch) * math.cos(shoulder_yaw)
            wrist_direction_y = math.sin(shoulder_pitch + elbow + wrist_pitch) * math.sin(shoulder_yaw)
            wrist_direction_z = math.cos(shoulder_pitch + elbow + wrist_pitch)

            x += hand_length * wrist_direction_x
            y += hand_length * wrist_direction_y
            z += hand_length * wrist_direction_z
            positions.append([x, y, z])

            return positions

try:
    from ml.nlp_processor import CommandParser
    NLP_AVAILABLE = True
except ImportError:
    print("⚠️  NLP processor not available - using simple command parser")
    NLP_AVAILABLE = False

    # Mock command parser
    class CommandParser:
        def parse_command(self, command):
            # Simple keyword-based parsing
            command = command.lower()
            if 'wave' in command:
                return {'action': 'gesture', 'gesture_type': 'wave', 'confidence': 0.9}
            elif 'reset' in command or 'home' in command:
                return {'action': 'reset_to_home', 'confidence': 0.95}
            elif 'move' in command or 'go' in command:
                return {'action': 'move_to_position', 'direction': 'forward', 'confidence': 0.8}
            else:
                return {'action': 'unknown', 'confidence': 0.1}

        def command_to_robot_action(self, parsed):
            action_type = parsed.get('action', 'unknown')
            return {
                'type': action_type,
                'success': action_type != 'unknown',
                'parameters': parsed
            }

app = Flask(__name__)
app.config['SECRET_KEY'] = 'robot_simulation_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global instances
robot_arm = None
command_parser = None
update_thread = None

def create_robot_instances():
    """Create robot arm and command parser instances."""
    global robot_arm, command_parser
    robot_arm = RobotArm()
    command_parser = CommandParser()
    print(f"✅ Robot arm created with {len(robot_arm.joints)} joints")
    print(f"✅ Command parser created")

def get_robot_state():
    """Get current robot state for visualization."""
    if robot_arm is None:
        return None

    try:
        # Get joint positions
        joint_positions = {}
        for name, joint in robot_arm.joints.items():
            joint_positions[name] = {
                'position': float(joint.position),
                'target': float(joint.target_position),
                'limits': [float(joint.limits[0]), float(joint.limits[1])]
            }

        # Get end effector pose
        end_pos, end_rot = robot_arm.get_end_effector_pose()

        # Get link positions for visualization
        link_positions = []
        if robot_arm.fk_solver:
            robot_arm.fk_solver.update()
            positions = robot_arm.fk_solver.get_link_positions()
            link_positions = [pos.tolist() if hasattr(pos, 'tolist') else pos for pos in positions]

        return {
            'joints': joint_positions,
            'end_effector': {
                'position': end_pos.tolist() if hasattr(end_pos, 'tolist') else end_pos,
                'rotation': end_rot.tolist() if hasattr(end_rot, 'tolist') else [0, 0, 0, 1]
            },
            'links': link_positions,
            'enabled': robot_arm.is_enabled,
            'timestamp': time.time()
        }
    except Exception as e:
        print(f"Error getting robot state: {e}")
        return None

def robot_update_loop():
    """Background thread to update robot and emit state."""
    print("🔄 Robot update loop started")
    loop_count = 0
    while True:
        try:
            if robot_arm:
                robot_arm.update(dt=0.1)
                state = get_robot_state()
                if state:
                    socketio.emit('robot_state', state)
                    loop_count += 1
                    if loop_count % 50 == 0:  # Log every 5 seconds
                        print(f"📡 Sent robot state update #{loop_count}")
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error in robot update loop: {e}")
            time.sleep(1.0)

@app.route('/')
def index():
    """Serve the main 3D interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/command', methods=['POST'])
def handle_command():
    """Handle natural language commands."""
    try:
        data = request.get_json()
        command = data.get('command', '')

        if not command:
            return jsonify({'success': False, 'error': 'No command provided'})

        # Parse command
        parsed = command_parser.parse_command(command)
        action = command_parser.command_to_robot_action(parsed)

        # Execute action
        success = execute_robot_action(action)

        response = {
            'success': success,
            'action_type': action['type'],
            'confidence': parsed.get('confidence', 0.0),
            'parsed': parsed
        }

        # Emit command result to all clients
        socketio.emit('command_result', response)

        return jsonify(response)

    except Exception as e:
        error_response = {
            'success': False,
            'error': str(e),
            'action_type': 'error',
            'confidence': 0.0
        }
        return jsonify(error_response)

def execute_robot_action(action):
    """Execute a robot action."""
    if not action.get('success', False):
        return False

    action_type = action['type']
    params = action.get('parameters', {})

    try:
        if action_type == 'move_to_position':
            if 'position' in params:
                target_pos = params['position']
                return robot_arm.move_to_pose(target_pos)
            elif 'direction' in params:
                current_pos, _ = robot_arm.get_end_effector_pose()
                direction = params['direction']
                distance = params.get('distance', 0.1)

                direction_vectors = {
                    'up': [0, 0, 1], 'down': [0, 0, -1],
                    'left': [0, 1, 0], 'right': [0, -1, 0],
                    'forward': [1, 0, 0], 'backward': [-1, 0, 0]
                }

                if direction in direction_vectors:
                    dir_vec = direction_vectors[direction]
                    # Simple list addition instead of numpy
                    target_pos = [
                        current_pos[0] + dir_vec[0] * distance,
                        current_pos[1] + dir_vec[1] * distance,
                        current_pos[2] + dir_vec[2] * distance
                    ]
                    return robot_arm.move_to_pose(target_pos)

        elif action_type == 'reset_to_home':
            robot_arm.reset_to_home()
            return True

        elif action_type == 'stop':
            robot_arm.emergency_stop()
            return True

        elif action_type == 'gesture':
            gesture_type = params.get('gesture_type', 'wave')
            return execute_gesture(gesture_type, params)

    except Exception as e:
        print(f"Error executing action: {e}")
        return False

    return False

def execute_gesture(gesture_type, params):
    """Execute a gesture."""
    if gesture_type == 'wave':
        # Simple wave gesture - move shoulder joints
        robot_arm.joints['shoulder_pitch'].target_position = 0.5
        robot_arm.joints['shoulder_yaw'].target_position = 0.3
        return True
    elif gesture_type == 'point':
        # Point gesture - extend arm forward
        robot_arm.joints['shoulder_pitch'].target_position = 0.3
        robot_arm.joints['elbow'].target_position = 0.8
        return True
    return False

@app.route('/api/joint/<joint_name>', methods=['POST'])
def move_joint(joint_name):
    """Move a specific joint."""
    try:
        data = request.get_json()
        position = data.get('position')

        if joint_name in robot_arm.joints:
            robot_arm.joints[joint_name].target_position = float(position)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Joint not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/reset', methods=['POST'])
def reset_robot():
    """Reset robot to home position."""
    try:
        robot_arm.reset_to_home()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/state', methods=['GET'])
def get_current_state():
    """Get current robot state for debugging."""
    try:
        state = get_robot_state()
        return jsonify(state if state else {'error': 'No state available'})
    except Exception as e:
        return jsonify({'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    # Send initial robot state
    state = get_robot_state()
    if state:
        emit('robot_state', state)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

@socketio.on('request_state')
def handle_state_request():
    """Handle request for current robot state."""
    state = get_robot_state()
    if state:
        emit('robot_state', state)

# HTML template will be added in the next part due to length constraints
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Arm 3D Visualization</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { margin: 0; padding: 0; font-family: Arial, sans-serif; background: #1a1a1a; color: white; overflow: hidden; }
        #container { display: flex; height: 100vh; }
        #viewport { flex: 1; position: relative; }
        #controls { width: 350px; background: #2a2a2a; padding: 20px; overflow-y: auto; border-left: 1px solid #444; }
        .control-section { margin-bottom: 25px; padding: 15px; background: #333; border-radius: 8px; }
        .control-section h3 { margin: 0 0 15px 0; color: #4CAF50; border-bottom: 1px solid #4CAF50; padding-bottom: 5px; }
        .joint-control { margin: 10px 0; }
        .joint-control label { display: block; margin-bottom: 5px; font-size: 12px; color: #ccc; }
        .joint-control input[type="range"] { width: 100%; margin: 5px 0; }
        .joint-value { font-family: monospace; font-size: 11px; color: #4CAF50; }
        button { padding: 8px 16px; margin: 5px 5px 5px 0; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
        button:hover { background: #45a049; }
        button.danger { background: #f44336; }
        button.danger:hover { background: #da190b; }
        #commandInput { width: 100%; padding: 10px; margin: 10px 0; background: #444; border: 1px solid #666; color: white; border-radius: 4px; }
        #status { background: #444; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 11px; margin: 10px 0; }
        #log { height: 150px; overflow-y: auto; background: #1a1a1a; padding: 10px; border: 1px solid #444; border-radius: 4px; font-family: monospace; font-size: 11px; }
        .log-entry { margin: 2px 0; }
        .log-success { color: #4CAF50; }
        .log-error { color: #f44336; }
        .log-info { color: #2196F3; }
        #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #4CAF50; font-size: 18px; }
    </style>
</head>
<body>
    <div id="container">
        <div id="viewport">
            <div id="loading">Loading 3D Visualization...</div>
        </div>
        <div id="controls">
            <h2>🤖 Robot Control</h2>

            <div class="control-section">
                <h3>📊 Status</h3>
                <div id="status">Connecting...</div>
                <button onclick="requestState()">Refresh State</button>
                <button onclick="forceRefreshVisualization()">Force Refresh 3D</button>
                <button class="danger" onclick="emergencyStop()">Emergency Stop</button>
            </div>

            <div class="control-section">
                <h3>🎮 Joint Control</h3>
                <div id="joints">Loading joints...</div>
                <button onclick="resetRobot()">Reset to Home</button>
                <button onclick="testSliders()">Test Sliders</button>
                <button onclick="debugSliders()">Debug Sliders</button>
            </div>

            <div class="control-section">
                <h3>💬 Commands</h3>
                <input type="text" id="commandInput" placeholder="Enter command (e.g., 'wave hello', 'move up')">
                <button onclick="sendCommand()">Execute</button>
                <div>
                    <button onclick="sendExampleCommand('wave hello')">Wave</button>
                    <button onclick="sendExampleCommand('move forward')">Forward</button>
                    <button onclick="sendExampleCommand('point at target')">Point</button>
                </div>
            </div>

            <div class="control-section">
                <h3>📝 Log</h3>
                <div id="log"></div>
            </div>
        </div>
    </div>

    <!-- Three.js and Socket.IO -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script>
        // Global variables
        let scene, camera, renderer, robot, socket;
        let robotParts = {};
        let currentRobotState = null;

        // Initialize 3D scene
        function init3D() {
            const viewport = document.getElementById('viewport');
            const loading = document.getElementById('loading');

            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a1a);

            // Camera
            camera = new THREE.PerspectiveCamera(75, viewport.clientWidth / viewport.clientHeight, 0.1, 1000);
            camera.position.set(2, 2, 1.5);
            camera.lookAt(0, 0, 0.5);

            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(viewport.clientWidth, viewport.clientHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            viewport.appendChild(renderer.domElement);

            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(2, 2, 3);
            directionalLight.castShadow = true;
            directionalLight.shadow.mapSize.width = 2048;
            directionalLight.shadow.mapSize.height = 2048;
            scene.add(directionalLight);

            // Grid
            const gridHelper = new THREE.GridHelper(4, 20, 0x444444, 0x222222);
            scene.add(gridHelper);

            // Coordinate axes
            const axesHelper = new THREE.AxesHelper(0.5);
            scene.add(axesHelper);

            // Robot base
            createRobotBase();

            // Controls
            setupControls();

            // Remove loading message
            loading.style.display = 'none';

            // Start render loop
            animate();

            log('3D visualization initialized', 'info');
        }

        function createRobotBase() {
            const baseGeometry = new THREE.CylinderGeometry(0.08, 0.1, 0.05, 16);
            const baseMaterial = new THREE.MeshLambertMaterial({ color: 0x666666 });
            const base = new THREE.Mesh(baseGeometry, baseMaterial);
            base.position.set(0, 0.025, 0);
            base.castShadow = true;
            scene.add(base);
            robotParts.base = base;
        }

        function updateRobotVisualization(state) {
            if (!state || !state.links) {
                console.log('No state or links data received');
                return;
            }

            console.log('Updating robot visualization with state:', state);

            // Clear existing robot links and joints (but keep base)
            Object.keys(robotParts).forEach(key => {
                if (key.startsWith('link_') || key.startsWith('joint_') || key === 'end_effector') {
                    scene.remove(robotParts[key]);
                    delete robotParts[key];
                }
            });

            // Draw links
            const linkColors = [0xff4444, 0x44ff44, 0x4444ff, 0xffff44];
            let prevPos = new THREE.Vector3(0, 0, 0);

            state.links.forEach((linkPos, index) => {
                const currentPos = new THREE.Vector3(linkPos[0], linkPos[2], linkPos[1]); // Convert Y-Z
                console.log(`Link ${index}: [${linkPos[0].toFixed(3)}, ${linkPos[1].toFixed(3)}, ${linkPos[2].toFixed(3)}] -> [${currentPos.x.toFixed(3)}, ${currentPos.y.toFixed(3)}, ${currentPos.z.toFixed(3)}]`);

                // Create link cylinder
                const direction = new THREE.Vector3().subVectors(currentPos, prevPos);
                const length = direction.length();

                if (length > 0.01) {
                    const linkGeometry = new THREE.CylinderGeometry(0.02, 0.02, length, 8);
                    const linkMaterial = new THREE.MeshLambertMaterial({
                        color: linkColors[index % linkColors.length]
                    });
                    const link = new THREE.Mesh(linkGeometry, linkMaterial);

                    // Position and orient the link
                    const midpoint = new THREE.Vector3().addVectors(prevPos, currentPos).multiplyScalar(0.5);
                    link.position.copy(midpoint);

                    // Orient the cylinder to point from prevPos to currentPos
                    const up = new THREE.Vector3(0, 1, 0);
                    const axis = direction.clone().normalize();
                    link.quaternion.setFromUnitVectors(up, axis);

                    link.castShadow = true;
                    scene.add(link);
                    robotParts[`link_${index}`] = link;
                }

                // Create joint sphere
                const jointGeometry = new THREE.SphereGeometry(0.025, 8, 8);
                const jointMaterial = new THREE.MeshLambertMaterial({ color: 0xff0000 });
                const joint = new THREE.Mesh(jointGeometry, jointMaterial);
                joint.position.copy(currentPos);
                joint.castShadow = true;
                scene.add(joint);
                robotParts[`joint_${index}`] = joint;

                prevPos = currentPos;
            });

            // End effector
            if (state.end_effector && state.end_effector.position) {
                // Remove existing end effector
                if (robotParts.end_effector) {
                    scene.remove(robotParts.end_effector);
                }

                const endPos = state.end_effector.position;
                console.log(`End effector: [${endPos[0].toFixed(3)}, ${endPos[1].toFixed(3)}, ${endPos[2].toFixed(3)}]`);

                const endEffectorGeometry = new THREE.SphereGeometry(0.03, 8, 8);
                const endEffectorMaterial = new THREE.MeshLambertMaterial({ color: 0xffff00 });
                const endEffector = new THREE.Mesh(endEffectorGeometry, endEffectorMaterial);
                endEffector.position.set(endPos[0], endPos[2], endPos[1]); // Convert Y-Z
                endEffector.castShadow = true;
                scene.add(endEffector);
                robotParts.end_effector = endEffector;
            }

            console.log('Robot visualization updated');
        }

        function setupControls() {
            let isMouseDown = false;
            let mouseX = 0, mouseY = 0;

            renderer.domElement.addEventListener('mousedown', (event) => {
                isMouseDown = true;
                mouseX = event.clientX;
                mouseY = event.clientY;
            });

            renderer.domElement.addEventListener('mouseup', () => {
                isMouseDown = false;
            });

            renderer.domElement.addEventListener('mousemove', (event) => {
                if (!isMouseDown) return;

                const deltaX = event.clientX - mouseX;
                const deltaY = event.clientY - mouseY;

                // Orbit camera
                const spherical = new THREE.Spherical();
                spherical.setFromVector3(camera.position);
                spherical.theta -= deltaX * 0.01;
                spherical.phi += deltaY * 0.01;
                spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

                camera.position.setFromSpherical(spherical);
                camera.lookAt(0, 0, 0.5);

                mouseX = event.clientX;
                mouseY = event.clientY;
            });

            renderer.domElement.addEventListener('wheel', (event) => {
                const scale = event.deltaY > 0 ? 1.1 : 0.9;
                camera.position.multiplyScalar(scale);
                camera.position.clampLength(0.5, 10);
            });
        }

        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }

        // Handle window resize
        window.addEventListener('resize', () => {
            const viewport = document.getElementById('viewport');
            camera.aspect = viewport.clientWidth / viewport.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(viewport.clientWidth, viewport.clientHeight);
        });

        // Socket.IO connection
        function initSocket() {
            socket = io();

            socket.on('connect', () => {
                log('Connected to robot server', 'info');
                socket.emit('request_state');
            });

            socket.on('disconnect', () => {
                log('Disconnected from robot server', 'error');
            });

            socket.on('robot_state', (state) => {
                console.log('Received robot state:', state);
                currentRobotState = state;
                updateRobotVisualization(state);
                updateUI(state);
            });

            socket.on('command_result', (result) => {
                const status = result.success ? 'success' : 'error';
                log(`Command "${result.action_type}" ${result.success ? 'succeeded' : 'failed'}`, status);
            });
        }

        // Track which sliders are being actively manipulated
        let activeSliders = new Set();
        let slidersInitialized = false;

        // Throttle slider input to prevent too many API calls
        let sliderThrottleTimers = {};

        // UI Functions
        function updateUI(state) {
            if (!state) return;

            // Update status
            const statusDiv = document.getElementById('status');
            const endPos = state.end_effector.position;
            statusDiv.innerHTML = `
                Enabled: ${state.enabled}<br>
                Joints: ${Object.keys(state.joints).length}<br>
                End Effector: [${endPos.map(x => x.toFixed(3)).join(', ')}]<br>
                Last Update: ${new Date(state.timestamp * 1000).toLocaleTimeString()}
            `;

            // Update joint controls
            updateJointControls(state.joints);
        }

        function updateJointControls(joints) {
            const jointsDiv = document.getElementById('joints');

            if (!jointsDiv) {
                console.error('❌ joints div not found!');
                return;
            }

            // If sliders haven't been initialized yet, create them
            if (!slidersInitialized) {
                createJointSliders(joints);
                slidersInitialized = true;
                return;
            }

            // Update existing sliders without recreating them
            Object.entries(joints).forEach(([name, joint]) => {
                if (name.includes('finger') || name.includes('thumb')) return; // Skip finger joints

                const slider = document.getElementById(`slider_${name}`);
                const valueDiv = document.getElementById(`value_${name}`);

                if (slider && valueDiv) {
                    // Only update slider value if user is not actively manipulating it
                    if (!activeSliders.has(name)) {
                        slider.value = joint.position;
                    }

                    // Always update the value display
                    valueDiv.textContent = `${joint.position.toFixed(3)} / ${joint.target.toFixed(3)}`;
                }
            });
        }

        function createJointSliders(joints) {
            console.log('🎮 Creating joint sliders for:', joints);
            const jointsDiv = document.getElementById('joints');

            let html = '';

            Object.entries(joints).forEach(([name, joint]) => {
                if (name.includes('finger') || name.includes('thumb')) return; // Skip finger joints

                console.log(`Creating slider for ${name}:`, joint);

                html += `
                    <div class="joint-control">
                        <label>${name}</label>
                        <input type="range"
                               min="${joint.limits[0]}"
                               max="${joint.limits[1]}"
                               step="0.01"
                               value="${joint.position}"
                               id="slider_${name}"
                               oninput="handleSliderInput('${name}', this.value)"
                               onchange="handleSliderChange('${name}', this.value)"
                               onmousedown="handleSliderStart('${name}')"
                               onmouseup="handleSliderEnd('${name}')"
                               ontouchstart="handleSliderStart('${name}')"
                               ontouchend="handleSliderEnd('${name}')">
                        <div class="joint-value" id="value_${name}">${joint.position.toFixed(3)} / ${joint.target.toFixed(3)}</div>
                    </div>
                `;
            });

            console.log('Generated HTML:', html);
            jointsDiv.innerHTML = html;

            // Verify sliders were created
            const sliders = document.querySelectorAll('input[type="range"]');
            console.log(`✅ Created ${sliders.length} sliders`);
            log(`Created ${sliders.length} joint sliders`, 'success');
        }

        function handleSliderStart(name) {
            console.log(`🎮 Slider interaction started: ${name}`);
            activeSliders.add(name);
        }

        function handleSliderEnd(name) {
            console.log(`🎮 Slider interaction ended: ${name}`);
            // Small delay to ensure the final value is sent before allowing updates
            setTimeout(() => {
                activeSliders.delete(name);
            }, 100);
        }

        function handleSliderInput(name, value) {
            // Update value display immediately for responsive feedback
            const valueDiv = document.getElementById(`value_${name}`);
            if (valueDiv) {
                valueDiv.textContent = `${parseFloat(value).toFixed(3)} / targeting...`;
            }

            // Throttle API calls - clear existing timer and set new one
            if (sliderThrottleTimers[name]) {
                clearTimeout(sliderThrottleTimers[name]);
            }

            sliderThrottleTimers[name] = setTimeout(() => {
                moveJoint(name, value);
                delete sliderThrottleTimers[name];
            }, 50); // 50ms throttle - allows up to 20 updates per second
        }

        function handleSliderChange(name, value) {
            console.log(`🎮 Slider change completed: ${name} = ${value}`);

            // Clear any pending throttled call and send final value immediately
            if (sliderThrottleTimers[name]) {
                clearTimeout(sliderThrottleTimers[name]);
                delete sliderThrottleTimers[name];
            }

            // Send final value update
            moveJoint(name, value);
        }

        function log(message, type = 'info') {
            const logDiv = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = `log-entry log-${type}`;
            entry.innerHTML = `[${timestamp}] ${message}`;
            logDiv.appendChild(entry);
            logDiv.scrollTop = logDiv.scrollHeight;
        }

        // Control functions
        function moveJoint(name, value) {
            console.log(`🎮 moveJoint called: ${name} = ${value}`);

            fetch(`/api/joint/${name}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ position: parseFloat(value) })
            })
            .then(response => {
                console.log(`API response for ${name}:`, response);
                return response.json();
            })
            .then(data => {
                console.log(`API data for ${name}:`, data);
                if (data.success) {
                    // Don't log every single movement to reduce spam
                    // log(`✅ Moved ${name} to ${value}`, 'success');

                    // Don't force refresh - let the normal WebSocket updates handle it
                    // The robot update loop will send updates automatically
                } else {
                    log(`❌ Failed to move ${name}: ${data.error}`, 'error');
                }
            })
            .catch(error => {
                console.error(`Error moving joint ${name}:`, error);
                log(`❌ Error moving joint: ${error}`, 'error');
            });
        }

        function sendCommand() {
            const input = document.getElementById('commandInput');
            const command = input.value.trim();
            if (!command) return;

            fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: command })
            })
            .then(response => response.json())
            .then(data => {
                log(`Command: "${command}" -> ${data.action_type} (${data.confidence.toFixed(2)})`, 'info');
                input.value = '';
            })
            .catch(error => log(`Command error: ${error}`, 'error'));
        }

        function sendExampleCommand(command) {
            document.getElementById('commandInput').value = command;
            sendCommand();
        }

        function resetRobot() {
            fetch('/api/reset', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        log('Robot reset to home position', 'success');
                    } else {
                        log(`Reset failed: ${data.error}`, 'error');
                    }
                })
                .catch(error => log(`Reset error: ${error}`, 'error'));
        }

        function emergencyStop() {
            // Implementation would depend on your robot's emergency stop mechanism
            log('Emergency stop activated', 'error');
        }

        function requestState() {
            if (socket) {
                socket.emit('request_state');
                log('Requested robot state update', 'info');
            }
        }

        function forceRefreshVisualization() {
            // Force refresh by fetching current state via API
            fetch('/api/state')
                .then(response => response.json())
                .then(state => {
                    console.log('Force refresh - received state:', state);
                    updateRobotVisualization(state);
                    updateUI(state);
                    log('Visualization force refreshed', 'info');
                })
                .catch(error => {
                    console.error('Force refresh failed:', error);
                    log(`Force refresh failed: ${error}`, 'error');
                });
        }

        function testSliders() {
            console.log('🧪 Testing slider functionality...');
            log('Testing sliders...', 'info');

            // Test if moveJoint function exists
            if (typeof moveJoint === 'function') {
                log('✅ moveJoint function exists', 'success');

                // Test moving shoulder_pitch
                console.log('Testing shoulder_pitch movement...');
                moveJoint('shoulder_pitch', 0.5);

                setTimeout(() => {
                    moveJoint('elbow', 1.0);
                }, 1000);

                setTimeout(() => {
                    moveJoint('shoulder_pitch', 0.0);
                    moveJoint('elbow', 0.0);
                }, 2000);

            } else {
                log('❌ moveJoint function not found!', 'error');
                console.error('moveJoint function not found!');
            }

            // Check if sliders exist
            const sliders = document.querySelectorAll('input[type="range"]');
            log(`Found ${sliders.length} sliders`, 'info');
            console.log('Sliders found:', sliders);

            // Test slider events
            sliders.forEach((slider, index) => {
                console.log(`Slider ${index}:`, slider.id, 'value:', slider.value);
            });
        }

        function debugSliders() {
            console.log('🔍 Debugging slider state...');
            log('Debugging sliders...', 'info');

            // Check initialization state
            log(`Sliders initialized: ${slidersInitialized}`, 'info');
            log(`Active sliders: ${Array.from(activeSliders).join(', ') || 'none'}`, 'info');
            log(`Throttle timers: ${Object.keys(sliderThrottleTimers).join(', ') || 'none'}`, 'info');

            // Check all sliders
            const sliders = document.querySelectorAll('input[type="range"]');
            log(`Found ${sliders.length} sliders in DOM`, 'info');

            sliders.forEach((slider, index) => {
                const name = slider.id.replace('slider_', '');
                const valueDiv = document.getElementById(`value_${name}`);
                log(`Slider ${index} (${name}): value=${slider.value}, active=${activeSliders.has(name)}, valueDiv=${valueDiv ? 'exists' : 'missing'}`, 'info');
            });

            // Check current robot state
            if (currentRobotState && currentRobotState.joints) {
                log('Current robot joint positions:', 'info');
                Object.entries(currentRobotState.joints).forEach(([name, joint]) => {
                    if (!name.includes('finger') && !name.includes('thumb')) {
                        log(`  ${name}: pos=${joint.position.toFixed(3)}, target=${joint.target.toFixed(3)}`, 'info');
                    }
                });
            } else {
                log('No robot state available', 'error');
            }
        }

        // Handle Enter key in command input
        document.addEventListener('DOMContentLoaded', () => {
            document.getElementById('commandInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sendCommand();
                }
            });
        });

        // Initialize everything
        window.addEventListener('load', () => {
            init3D();
            initSocket();
            log('3D Robot Visualization ready', 'success');
        });
    </script>
</body>
</html>
"""

def main():
    """Main entry point."""
    print("🤖 Starting Enhanced 3D Robot Arm Interface...")

    # Create robot instances
    create_robot_instances()

    # Start robot update thread
    global update_thread
    update_thread = threading.Thread(target=robot_update_loop, daemon=True)
    update_thread.start()

    print("🌐 3D Web interface starting...")
    print("🚀 Open your browser to: http://localhost:8080")
    print("📱 Features available:")
    print("   • Real-time 3D robot visualization")
    print("   • Interactive joint controls")
    print("   • Natural language commands")
    print("   • Mouse controls: drag to orbit, scroll to zoom")

    # Run the Flask-SocketIO app
    socketio.run(app, host='0.0.0.0', port=8080, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()
