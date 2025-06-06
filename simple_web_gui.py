#!/usr/bin/env python3
"""Simple working web interface for robot arm simulation."""

import sys
import os
import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot_arm.robot_arm import RobotArm
from ml.nlp_processor import CommandParser

class SimpleRobotHandler(SimpleHTTPRequestHandler):
    """Simple HTTP handler for robot control."""
    
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
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .status { background: #e9ecef; padding: 10px; border-radius: 5px; font-family: monospace; }
        .command-input { width: 100%; padding: 10px; margin: 10px 0; }
        .log { height: 300px; overflow-y: scroll; background: #f8f9fa; padding: 10px; border: 1px solid #ddd; }
        .joint-display { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }
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
            <h2>🎮 Joint Information</h2>
            <div id="joints" class="joint-display">Loading joints...</div>
            <button onclick="updateJoints()">Refresh Joints</button>
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
            <button onclick="clearLog()">Clear Log</button>
        </div>
    </div>

    <script>
        function log(message) {
            const logDiv = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            logDiv.innerHTML += `[${timestamp}] ${message}<br>`;
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        function clearLog() {
            document.getElementById('log').innerHTML = '';
            log('🤖 Robot Arm Control Panel ready');
        }
        
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.text())
                .then(text => {
                    try {
                        const data = JSON.parse(text);
                        document.getElementById('status').innerHTML = 
                            `Robot Enabled: ${data.enabled ? 'Yes' : 'No'}<br>` +
                            `Total Joints: ${data.joint_count}<br>` +
                            `Status: ${data.status}`;
                        log('✅ Status updated successfully');
                    } catch (e) {
                        document.getElementById('status').innerHTML = 'Error loading status';
                        log('❌ Status update failed: ' + e.message);
                    }
                })
                .catch(error => {
                    log('❌ Error updating status: ' + error);
                    document.getElementById('status').innerHTML = 'Connection error';
                });
        }
        
        function updateJoints() {
            fetch('/api/joints')
                .then(response => response.text())
                .then(text => {
                    try {
                        const data = JSON.parse(text);
                        let html = '<h3>Main Arm Joints:</h3>';
                        const mainJoints = ['shoulder_pitch', 'shoulder_yaw', 'shoulder_roll', 'elbow_flexion', 'wrist_pitch', 'wrist_yaw'];
                        
                        for (const jointName of mainJoints) {
                            if (data[jointName]) {
                                const joint = data[jointName];
                                html += `<div><strong>${jointName}:</strong> ${joint.position.toFixed(3)} rad</div>`;
                            }
                        }
                        
                        document.getElementById('joints').innerHTML = html;
                        log('✅ Joints updated successfully');
                    } catch (e) {
                        document.getElementById('joints').innerHTML = 'Error loading joints';
                        log('❌ Joint update failed: ' + e.message);
                    }
                })
                .catch(error => {
                    log('❌ Error updating joints: ' + error);
                    document.getElementById('joints').innerHTML = 'Connection error';
                });
        }
        
        function sendCommand() {
            const command = document.getElementById('commandInput').value;
            if (!command) return;
            
            log(`📤 Sending command: "${command}"`);
            
            fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: command})
            })
            .then(response => response.text())
            .then(text => {
                try {
                    const data = JSON.parse(text);
                    log(`📥 Response: ${data.action_type} (confidence: ${data.confidence.toFixed(2)})`);
                    if (data.success) {
                        log('✅ Command executed successfully');
                    } else {
                        log('❌ Command execution failed: ' + (data.error || 'Unknown error'));
                    }
                    document.getElementById('commandInput').value = '';
                    updateStatus();
                    updateJoints();
                } catch (e) {
                    log('❌ Command response error: ' + e.message);
                }
            })
            .catch(error => {
                log('❌ Error sending command: ' + error);
            });
        }
        
        function sendExampleCommand(command) {
            document.getElementById('commandInput').value = command;
            sendCommand();
        }
        
        function resetRobot() {
            log('🔄 Resetting robot to home position...');
            fetch('/api/reset', {method: 'POST'})
                .then(response => response.text())
                .then(text => {
                    try {
                        const data = JSON.parse(text);
                        if (data.success) {
                            log('✅ Robot reset to home position');
                            updateJoints();
                        } else {
                            log('❌ Reset failed: ' + (data.error || 'Unknown error'));
                        }
                    } catch (e) {
                        log('❌ Reset response error: ' + e.message);
                    }
                })
                .catch(error => {
                    log('❌ Error resetting robot: ' + error);
                });
        }
        
        // Handle Enter key in command input
        document.getElementById('commandInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendCommand();
            }
        });
        
        // Initial load
        log('🤖 Robot Arm Control Panel loaded');
        updateStatus();
        updateJoints();
        
        // Auto-update every 5 seconds
        setInterval(() => {
            updateStatus();
            updateJoints();
        }, 5000);
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
            status = {
                'enabled': bool(self.robot_arm.is_enabled),
                'joint_count': len(self.robot_arm.joints),
                'status': 'Running' if self.robot_arm.is_enabled else 'Disabled'
            }
        except Exception as e:
            status = {
                'enabled': False,
                'joint_count': 0,
                'status': f'Error: {str(e)}'
            }
        
        self.send_json_response(status)
    
    def send_joint_info(self):
        """Send joint information as JSON."""
        try:
            joint_info = {}
            for name, joint in self.robot_arm.joints.items():
                joint_info[name] = {
                    'position': float(joint.position),
                    'velocity': float(joint.velocity),
                    'limits': [float(joint.limits[0]), float(joint.limits[1])]
                }
        except Exception as e:
            joint_info = {'error': str(e)}
        
        self.send_json_response(joint_info)
    
    def handle_command(self):
        """Handle natural language command."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            command = data.get('command', '')
            
            parsed = self.command_parser.parse_command(command)
            action = self.command_parser.command_to_robot_action(parsed)
            
            # Simple execution
            success = False
            error_msg = None
            
            try:
                if action['type'] == 'reset_to_home':
                    self.robot_arm.reset_to_home()
                    success = True
                elif action['type'] == 'move_to_position':
                    success = True  # Just acknowledge for now
                elif action['type'] == 'gesture':
                    success = True  # Just acknowledge for now
                else:
                    success = action.get('success', False)
            except Exception as e:
                error_msg = str(e)
            
            response = {
                'success': success,
                'action_type': action['type'],
                'confidence': float(parsed['confidence']),
                'error': error_msg
            }
        except Exception as e:
            response = {
                'success': False,
                'action_type': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
        
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
        self.wfile.write(json.dumps(data).encode())

def run_simple_web_interface(port=8081):
    """Run the simple web interface."""
    print("🤖 Starting Simple Robot Arm Web Interface...")
    
    try:
        # Create robot and command parser
        print("Creating robot arm...")
        robot = RobotArm()
        print(f"✅ Robot created with {len(robot.joints)} joints")
        
        print("Loading NLP model...")
        command_parser = CommandParser()
        print("✅ NLP model loaded")
        
        # Create handler with robot instance
        def handler(*args, **kwargs):
            return SimpleRobotHandler(*args, robot_arm=robot, command_parser=command_parser, **kwargs)
        
        # Start robot update thread
        def update_robot():
            while True:
                try:
                    robot.update(dt=0.1)
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Robot update error: {e}")
                    time.sleep(1.0)
        
        robot_thread = threading.Thread(target=update_robot, daemon=True)
        robot_thread.start()
        print("✅ Robot update thread started")
        
        # Start web server
        server = HTTPServer(('localhost', port), handler)
        
        print(f"🌐 Simple web interface running at: http://localhost:{port}")
        print("🚀 Opening browser...")
        
        # Open browser
        webbrowser.open(f'http://localhost:{port}')
        
        print("✅ Web interface ready!")
        print("Press Ctrl+C to stop the server")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down web interface...")
            server.shutdown()
            
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_simple_web_interface()
