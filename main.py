#!/usr/bin/env python3
"""Main application for robot arm simulation."""

import sys
import os
import argparse
import threading
import time
from typing import Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot_arm.robot_arm import RobotArm
from physics.physics_engine import PhysicsEngine
from visualization.renderer import Renderer
from ui.control_panel import ControlPanel
from ml.rl_trainer import RLTrainer
from ml.nlp_processor import CommandParser
from core.config import config


class RobotSimulation:
    """Main robot arm simulation application."""

    def __init__(self, use_physics: bool = True, use_gui: bool = True,
                 use_visualization: bool = True):
        """Initialize the simulation.

        Args:
            use_physics: Whether to use physics simulation
            use_gui: Whether to show GUI control panel
            use_visualization: Whether to show 3D visualization
        """
        self.use_physics = use_physics
        self.use_gui = use_gui
        self.use_visualization = use_visualization

        # Core components
        self.robot_arm = None
        self.physics_engine = None
        self.renderer = None
        self.control_panel = None
        self.trainer = None
        self.command_parser = None

        # Simulation state
        self.running = False
        self.simulation_thread = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all simulation components."""
        print("Initializing robot arm simulation...")

        # Create robot arm
        print("Creating robot arm...")
        self.robot_arm = RobotArm()

        # Create physics engine if requested
        if self.use_physics:
            print("Initializing physics engine...")
            try:
                self.physics_engine = PhysicsEngine(gui=False)
                self.physics_engine.load_robot(self.robot_arm)
                print("Physics engine initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize physics engine: {e}")
                self.physics_engine = None

        # Create 3D renderer if requested
        if self.use_visualization:
            print("Initializing 3D visualization...")
            try:
                import platform
                if platform.system() == "Darwin":  # macOS
                    print("Skipping 3D visualization on macOS (OpenGL compatibility issues)")
                    self.renderer = None
                else:
                    window_size = config.window_size
                    self.renderer = Renderer(window_size[0], window_size[1])
                    print("3D visualization initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize 3D visualization: {e}")
                self.renderer = None

        # Create GUI control panel if requested
        if self.use_gui:
            print("Initializing GUI control panel...")
            try:
                import platform
                import os

                # Check if GUI is force-enabled via environment variable
                force_gui = os.environ.get('FORCE_GUI', 'false').lower() == 'true'

                if platform.system() == "Darwin" and not force_gui:  # macOS
                    print("Skipping GUI on macOS (Tkinter compatibility issues)")
                    print("To force-enable GUI, run: FORCE_GUI=true python main.py")
                    print("Use command line interface instead:")
                    print("  python main.py --command 'wave hello'")
                    print("  python main.py --demo")
                    self.control_panel = None
                else:
                    if force_gui:
                        print("Force-enabling GUI (may crash on macOS)...")
                    self.control_panel = ControlPanel(self.robot_arm)
                    print("GUI control panel initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize GUI: {e}")
                print("Use command line interface instead:")
                print("  python main.py --command 'wave hello'")
                print("  python main.py --demo")
                self.control_panel = None

        # Create ML components
        print("Initializing ML components...")
        try:
            self.trainer = RLTrainer(self.robot_arm, self.physics_engine)
            self.command_parser = CommandParser()
            print("ML components initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize ML components: {e}")

        print("Initialization complete!")

    def start_simulation(self):
        """Start the simulation loop."""
        if self.running:
            print("Simulation is already running")
            return

        print("Starting simulation...")
        self.running = True

        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

        # Start GUI if available
        if self.control_panel:
            print("Starting GUI control panel...")
            self.control_panel.run()
        elif self.renderer:
            print("Starting 3D visualization...")
            self._visualization_loop()
        else:
            print("Running headless simulation...")
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Simulation interrupted by user")

        self.stop_simulation()

    def stop_simulation(self):
        """Stop the simulation."""
        if not self.running:
            return

        print("Stopping simulation...")
        self.running = False

        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)

        self._cleanup()
        print("Simulation stopped")

    def _simulation_loop(self):
        """Main simulation loop."""
        dt = config.simulation_timestep

        while self.running:
            start_time = time.time()

            try:
                # Update robot arm
                self.robot_arm.update(dt)

                # Update physics
                if self.physics_engine:
                    self.physics_engine.step_simulation()

            except Exception as e:
                print(f"Error in simulation loop: {e}")

            # Maintain target frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)

    def _visualization_loop(self):
        """3D visualization loop."""
        if not self.renderer:
            return

        print("Starting 3D visualization loop...")

        while self.running and not self.renderer.should_close():
            try:
                # Poll events
                self.renderer.poll_events()

                # Render frame
                self.renderer.render_frame(self.robot_arm)

            except Exception as e:
                print(f"Error in visualization loop: {e}")
                break

        self.running = False

    def _cleanup(self):
        """Cleanup resources."""
        if self.control_panel:
            try:
                self.control_panel.destroy()
            except:
                pass

        if self.renderer:
            try:
                self.renderer.cleanup()
            except:
                pass

        if self.physics_engine:
            try:
                self.physics_engine.disconnect()
            except:
                pass

    def execute_command(self, command: str) -> bool:
        """Execute a natural language command.

        Args:
            command: Natural language command

        Returns:
            True if successful
        """
        if not self.command_parser:
            print("Command parser not available")
            return False

        try:
            print(f"Executing command: {command}")

            # Parse command
            parsed = self.command_parser.parse_command(command)
            action = self.command_parser.command_to_robot_action(parsed)

            print(f"Parsed action: {action}")

            if not action['success']:
                print("Command could not be understood")
                return False

            # Execute action (simplified implementation)
            action_type = action['type']
            params = action.get('parameters', {})

            if action_type == 'move_to_position':
                if 'position' in params:
                    import numpy as np
                    target_pos = np.array(params['position'])
                    return self.robot_arm.move_to_pose(target_pos)

            elif action_type == 'reset_to_home':
                self.robot_arm.reset_to_home()
                return True

            elif action_type == 'stop':
                self.robot_arm.emergency_stop()
                return True

            print(f"Action type '{action_type}' not implemented")
            return False

        except Exception as e:
            print(f"Error executing command: {e}")
            return False

    def train_model(self, algorithm: str = "PPO", timesteps: int = 100000):
        """Train a reinforcement learning model.

        Args:
            algorithm: RL algorithm to use
            timesteps: Number of training timesteps
        """
        if not self.trainer:
            print("Trainer not available")
            return

        print(f"Starting training with {algorithm} for {timesteps} timesteps...")

        try:
            self.trainer.train(algorithm, timesteps)
            print("Training completed successfully")
        except Exception as e:
            print(f"Training failed: {e}")

    def demo_movements(self):
        """Demonstrate various robot movements."""
        print("Starting movement demonstration...")

        import numpy as np

        # Demo positions
        positions = [
            [0.3, 0.0, 0.3],   # Forward
            [0.2, 0.2, 0.4],   # Right and up
            [0.2, -0.2, 0.4],  # Left and up
            [0.4, 0.0, 0.2],   # Forward and down
            [0.0, 0.0, 0.5],   # Straight up
        ]

        for i, pos in enumerate(positions):
            print(f"Moving to position {i+1}: {pos}")
            success = self.robot_arm.move_to_pose(np.array(pos))
            print(f"Movement {'successful' if success else 'failed'}")
            time.sleep(2.0)

        # Return to home
        print("Returning to home position")
        self.robot_arm.reset_to_home()
        print("Demo complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Robot Arm Simulation")
    parser.add_argument("--no-physics", action="store_true",
                       help="Disable physics simulation")
    parser.add_argument("--no-gui", action="store_true",
                       help="Disable GUI control panel")
    parser.add_argument("--no-visualization", action="store_true",
                       help="Disable 3D visualization")
    parser.add_argument("--command", type=str,
                       help="Execute a single command and exit")
    parser.add_argument("--demo", action="store_true",
                       help="Run movement demonstration")
    parser.add_argument("--train", type=str, choices=["PPO", "SAC", "TD3"],
                       help="Train a model with specified algorithm")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Number of training timesteps")

    args = parser.parse_args()

    # Create simulation
    simulation = RobotSimulation(
        use_physics=not args.no_physics,
        use_gui=not args.no_gui,
        use_visualization=not args.no_visualization
    )

    try:
        if args.command:
            # Execute single command
            simulation.execute_command(args.command)
        elif args.demo:
            # Run demo
            simulation.start_simulation()
            time.sleep(1.0)  # Let simulation start
            simulation.demo_movements()
            simulation.stop_simulation()
        elif args.train:
            # Train model
            simulation.train_model(args.train, args.timesteps)
        else:
            # Start interactive simulation
            simulation.start_simulation()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation.stop_simulation()


if __name__ == "__main__":
    main()
