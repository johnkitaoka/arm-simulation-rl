#!/usr/bin/env python3
"""Web-based interface for robot arm simulation (macOS compatible)."""

import sys
import os
import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import webbrowser
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot_arm.robot_arm import RobotArm
from ml.nlp_processor import CommandParser

class RobotWebHandler(SimpleHTTPRequestHandler):
    """HTTP handler for robot control web interface."""

    def __init__(self, *args, robot_arm=None, command_parser=None, **kwargs):
        self.robot_arm = robot_arm
        self.command_parser = command_parser
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.send_web_interface()
        elif self.path == '/api/status':
            self.send_robot_status()
        elif self.path == '/api/joints':
            self.send_joint_info()
        else:
            super().do_GET()

    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/api/command':
            self.handle_command()
        elif self.path == '/api/move_joint':
            self.handle_joint_move()
        elif self.path == '/api/reset':
            self.handle_reset()
        else:
            self.send_error(404)

    def send_web_interface(self):
        """Send the main web interface."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Arm Control</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .joint-control { display: flex; align-items: center; margin: 10px 0; }
        .joint-control label { width: 150px; }
        .joint-control input { flex: 1; margin: 0 10px; }
        .joint-control span { width: 80px; text-align: right; }
        button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .status { background: #e9ecef; padding: 10px; border-radius: 5px; font-family: monospace; }
        .command-input { width: 100%; padding: 10px; margin: 10px 0; }
        .log { height: 200px; overflow-y: scroll; background: #f8f9fa; padding: 10px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Robot Arm Control Panel</h1>

        <div class="section">
            <h2>📊 Robot Status</h2>
            <div id="status" class="status">Loading...</div>
            <button onclick="updateStatus()">Refresh Status</button>
        </div>

        <div class="section">
            <h2>🎮 Joint Control</h2>
            <div id="joints">Loading joints...</div>
            <button onclick="resetRobot()">Reset to Home</button>
        </div>

        <div class="section">
            <h2>💬 Natural Language Commands</h2>
            <input type="text" id="commandInput" class="command-input" placeholder="Enter command (e.g., 'wave hello', 'move forward')">
            <button onclick="sendCommand()">Execute Command</button>
            <div>
                <h3>Example Commands:</h3>
                <button onclick="sendExampleCommand('wave hello')">Wave Hello</button>
                <button onclick="sendExampleCommand('move forward')">Move Forward</button>
                <button onclick="sendExampleCommand('point at target')">Point at Target</button>
                <button onclick="sendExampleCommand('reset to home')">Reset Home</button>
            </div>
        </div>

        <div class="section">
            <h2>📝 Command Log</h2>
            <div id="log" class="log"></div>
        </div>
    </div>

    <script>
        function log(message) {
            const logDiv = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            logDiv.innerHTML += `[${timestamp}] ${message}<br>`;
            logDiv.scrollTop = logDiv.scrollHeight;
        }

        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').innerHTML =
                        `Enabled: ${data.enabled}<br>` +
                        `Joints: ${data.joint_count}<br>` +
                        `End Effector: [${data.end_effector.map(x => x.toFixed(3)).join(', ')}]`;
                })
                .catch(error => log('Error updating status: ' + error));
        }

        function updateJoints() {
            fetch('/api/joints')
                .then(response => response.json())
                .then(data => {
                    let html = '';
                    for (const [name, info] of Object.entries(data)) {
                        if (name.includes('finger') || name.includes('thumb')) continue; // Skip finger joints for simplicity
                        html += `
                            <div class="joint-control">
                                <label>${name}:</label>
                                <input type="range" min="${info.limits[0]}" max="${info.limits[1]}"
                                       step="0.01" value="${info.position}"
                                       onchange="moveJoint('${name}', this.value)">
                                <span>${info.position.toFixed(3)}</span>
                            </div>`;
                    }
                    document.getElementById('joints').innerHTML = html;
                })
                .catch(error => log('Error updating joints: ' + error));
        }

        function moveJoint(name, value) {
            fetch('/api/move_joint', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({joint: name, position: parseFloat(value)})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    log(`Moved ${name} to ${value}`);
                } else {
                    log(`Failed to move ${name}: ${data.error}`);
                }
            })
            .catch(error => log('Error moving joint: ' + error));
        }

        function sendCommand() {
            const command = document.getElementById('commandInput').value;
            if (!command) return;

            fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: command})
            })
            .then(response => response.json())
            .then(data => {
                log(`Command: "${command}" -> ${data.action_type} (confidence: ${data.confidence.toFixed(2)})`);
                if (data.success) {
                    log('✅ Command executed successfully');
                } else {
                    log('❌ Command execution failed');
                }
                document.getElementById('commandInput').value = '';
            })
            .catch(error => log('Error sending command: ' + error));
        }

        function sendExampleCommand(command) {
            document.getElementById('commandInput').value = command;
            sendCommand();
        }

        function resetRobot() {
            fetch('/api/reset', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        log('✅ Robot reset to home position');
                        updateJoints();
                    } else {
                        log('❌ Reset failed');
                    }
                })
                .catch(error => log('Error resetting robot: ' + error));
        }

        // Auto-update every 2 seconds
        setInterval(() => {
            updateStatus();
            updateJoints();
        }, 2000);

        // Initial load
        updateStatus();
        updateJoints();
        log('🤖 Robot Arm Control Panel loaded');

        // Handle Enter key in command input
        document.getElementById('commandInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendCommand();
            }
        });
    </script>
</body>
</html>
        """

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def send_robot_status(self):
        """Send robot status as JSON."""
        try:
            end_effector_pos, _ = self.robot_arm.get_end_effector_pose()
            status = {
                'enabled': self.robot_arm.is_enabled,
                'joint_count': len(self.robot_arm.joints),
                'end_effector': end_effector_pos.tolist()
            }
        except:
            status = {
                'enabled': False,
                'joint_count': 0,
                'end_effector': [0, 0, 0]
            }

        self.send_json_response(status)

    def send_joint_info(self):
        """Send joint information as JSON."""
        joint_info = self.robot_arm.get_joint_info()
        self.send_json_response(joint_info)

    def handle_command(self):
        """Handle natural language command."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())

        command = data.get('command', '')

        try:
            parsed = self.command_parser.parse_command(command)
            action = self.command_parser.command_to_robot_action(parsed)

            # Simple execution
            success = False
            if action['type'] == 'reset_to_home':
                self.robot_arm.reset_to_home()
                success = True
            elif action['type'] == 'move_to_position':
                success = True  # Just acknowledge for now

            response = {
                'success': success,
                'action_type': action['type'],
                'confidence': parsed['confidence']
            }
        except Exception as e:
            response = {
                'success': False,
                'action_type': 'error',
                'confidence': 0.0,
                'error': str(e)
            }

        self.send_json_response(response)

    def handle_joint_move(self):
        """Handle joint movement."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())

        joint_name = data.get('joint')
        position = data.get('position')

        try:
            if joint_name in self.robot_arm.joints:
                self.robot_arm.joints[joint_name].target_position = position
                response = {'success': True}
            else:
                response = {'success': False, 'error': 'Joint not found'}
        except Exception as e:
            response = {'success': False, 'error': str(e)}

        self.send_json_response(response)

    def handle_reset(self):
        """Handle robot reset."""
        try:
            self.robot_arm.reset_to_home()
            response = {'success': True}
        except Exception as e:
            response = {'success': False, 'error': str(e)}

        self.send_json_response(response)

    def send_json_response(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Convert numpy types and booleans to JSON-serializable types
        def convert_types(obj):
            if hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        safe_data = convert_types(data)
        self.wfile.write(json.dumps(safe_data).encode())

def run_web_interface(port=8080):
    """Run the web interface."""
    print("🤖 Starting Robot Arm Web Interface...")

    # Create robot and command parser
    robot = RobotArm()
    command_parser = CommandParser()

    # Create handler with robot instance
    def handler(*args, **kwargs):
        return RobotWebHandler(*args, robot_arm=robot, command_parser=command_parser, **kwargs)

    # Start robot update thread
    def update_robot():
        while True:
            robot.update(dt=0.1)
            time.sleep(0.1)

    robot_thread = threading.Thread(target=update_robot, daemon=True)
    robot_thread.start()

    # Start web server
    server = HTTPServer(('localhost', port), handler)

    print(f"🌐 Web interface running at: http://localhost:{port}")
    print("🚀 Opening browser...")

    # Open browser
    webbrowser.open(f'http://localhost:{port}')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down web interface...")
        server.shutdown()

if __name__ == "__main__":
    run_web_interface()
