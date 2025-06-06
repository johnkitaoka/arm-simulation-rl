"""Natural Language Processing for robot commands."""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModel
import re
import json
import os

from core.config import config


class CommandParser:
    """Parse natural language commands into robot actions."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """Initialize the command parser.

        Args:
            model_name: Pre-trained model name for NLP
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Command vocabulary and patterns
        self.action_keywords = {
            'move': ['move', 'go', 'reach', 'extend', 'position', 'forward', 'backward', 'up', 'down', 'left', 'right'],
            'grab': ['grab', 'grasp', 'pick', 'take', 'hold'],
            'release': ['release', 'drop', 'let go', 'open'],
            'wave': ['wave', 'greet', 'hello', 'greeting', 'hi'],
            'point': ['point', 'indicate', 'show', 'target'],
            'stop': ['stop', 'halt', 'freeze', 'pause'],
            'reset': ['reset', 'home', 'initial', 'start', 'return']
        }

        self.object_keywords = {
            'ball': ['ball', 'sphere', 'orb'],
            'cube': ['cube', 'box', 'block'],
            'target': ['target', 'goal', 'destination'],
            'object': ['object', 'item', 'thing']
        }

        self.color_keywords = {
            'red': ['red', 'crimson', 'scarlet'],
            'blue': ['blue', 'azure', 'navy'],
            'green': ['green', 'emerald', 'lime'],
            'yellow': ['yellow', 'gold', 'amber'],
            'white': ['white', 'ivory'],
            'black': ['black', 'dark']
        }

        self.direction_keywords = {
            'up': ['up', 'above', 'higher', 'upward'],
            'down': ['down', 'below', 'lower', 'downward'],
            'left': ['left', 'leftward'],
            'right': ['right', 'rightward'],
            'forward': ['forward', 'ahead', 'front'],
            'backward': ['backward', 'back', 'behind']
        }

        # Load or initialize the model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the NLP model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded NLP model: {self.model_name}")
        except Exception as e:
            print(f"Warning: Could not load NLP model: {e}")
            print("Using rule-based parsing only")

    def parse_command(self, command: str) -> Dict[str, Any]:
        """Parse a natural language command into structured action.

        Args:
            command: Natural language command string

        Returns:
            Dictionary containing parsed action information
        """
        command = command.lower().strip()

        # Initialize result structure
        result = {
            'action': None,
            'target_object': None,
            'target_position': None,
            'target_color': None,
            'direction': None,
            'confidence': 0.0,
            'raw_command': command
        }

        # Rule-based parsing
        result.update(self._rule_based_parse(command))

        # If we have a model, use it for additional context
        if self.model is not None:
            model_result = self._model_based_parse(command)
            # Combine results, giving preference to rule-based for now
            if result['confidence'] < 0.5:
                result.update(model_result)

        return result

    def _rule_based_parse(self, command: str) -> Dict[str, Any]:
        """Parse command using rule-based approach.

        Args:
            command: Preprocessed command string

        Returns:
            Parsed action dictionary
        """
        result = {
            'action': None,
            'target_object': None,
            'target_position': None,
            'target_color': None,
            'direction': None,
            'confidence': 0.0
        }

        # Extract action
        for action, keywords in self.action_keywords.items():
            if any(keyword in command for keyword in keywords):
                result['action'] = action
                result['confidence'] += 0.3
                break

        # Extract object
        for obj, keywords in self.object_keywords.items():
            if any(keyword in command for keyword in keywords):
                result['target_object'] = obj
                result['confidence'] += 0.2
                break

        # Extract color
        for color, keywords in self.color_keywords.items():
            if any(keyword in command for keyword in keywords):
                result['target_color'] = color
                result['confidence'] += 0.2
                break

        # Extract direction
        for direction, keywords in self.direction_keywords.items():
            if any(keyword in command for keyword in keywords):
                result['direction'] = direction
                result['confidence'] += 0.2
                break

        # Extract position coordinates if present
        position_match = re.search(r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', command)
        if position_match:
            try:
                x, y, z = map(float, position_match.groups())
                result['target_position'] = [x, y, z]
                result['confidence'] += 0.3
            except ValueError:
                pass

        return result

    def _model_based_parse(self, command: str) -> Dict[str, Any]:
        """Parse command using transformer model.

        Args:
            command: Command string

        Returns:
            Parsed action dictionary
        """
        result = {
            'action': None,
            'target_object': None,
            'target_position': None,
            'target_color': None,
            'direction': None,
            'confidence': 0.0
        }

        try:
            # Tokenize and encode
            inputs = self.tokenizer(command, return_tensors="pt",
                                  padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling

            # Simple similarity-based classification
            # In a full implementation, you would train a classifier on top
            result['confidence'] = 0.1  # Low confidence for now

        except Exception as e:
            print(f"Error in model-based parsing: {e}")

        return result

    def command_to_robot_action(self, parsed_command: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parsed command to robot action parameters.

        Args:
            parsed_command: Output from parse_command

        Returns:
            Robot action parameters
        """
        action = {
            'type': 'none',
            'parameters': {},
            'success': False
        }

        if parsed_command['confidence'] < 0.3:
            action['error'] = "Command not understood"
            return action

        cmd_action = parsed_command['action']

        if cmd_action == 'move':
            action['type'] = 'move_to_position'

            # Determine target position
            if parsed_command['target_position']:
                action['parameters']['position'] = parsed_command['target_position']
            elif parsed_command['direction']:
                action['parameters']['direction'] = parsed_command['direction']
                action['parameters']['distance'] = 0.1  # Default distance
            else:
                action['parameters']['position'] = [0.3, 0.0, 0.3]  # Default position

            action['success'] = True

        elif cmd_action == 'grab':
            action['type'] = 'grasp_object'
            action['parameters']['object_type'] = parsed_command.get('target_object', 'unknown')
            action['parameters']['object_color'] = parsed_command.get('target_color', 'any')
            action['success'] = True

        elif cmd_action == 'release':
            action['type'] = 'release_object'
            action['success'] = True

        elif cmd_action == 'wave':
            action['type'] = 'gesture'
            action['parameters']['gesture_type'] = 'wave'
            action['success'] = True

        elif cmd_action == 'point':
            action['type'] = 'gesture'
            action['parameters']['gesture_type'] = 'point'
            if parsed_command['target_position']:
                action['parameters']['target'] = parsed_command['target_position']
            action['success'] = True

        elif cmd_action == 'stop':
            action['type'] = 'stop'
            action['success'] = True

        elif cmd_action == 'reset':
            action['type'] = 'reset_to_home'
            action['success'] = True

        return action


class CommandHistory:
    """Manage command history and learning."""

    def __init__(self, max_history: int = 100):
        """Initialize command history.

        Args:
            max_history: Maximum number of commands to store
        """
        self.max_history = max_history
        self.history = []
        self.success_rate = {}

    def add_command(self, command: str, parsed: Dict[str, Any],
                   success: bool, execution_time: float) -> None:
        """Add a command to history.

        Args:
            command: Original command string
            parsed: Parsed command result
            success: Whether execution was successful
            execution_time: Time taken to execute
        """
        entry = {
            'timestamp': torch.tensor(len(self.history)),
            'command': command,
            'parsed': parsed,
            'success': success,
            'execution_time': execution_time
        }

        self.history.append(entry)

        # Maintain max history size
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Update success rate
        action_type = parsed.get('action', 'unknown')
        if action_type not in self.success_rate:
            self.success_rate[action_type] = {'total': 0, 'success': 0}

        self.success_rate[action_type]['total'] += 1
        if success:
            self.success_rate[action_type]['success'] += 1

    def get_success_rate(self, action_type: Optional[str] = None) -> float:
        """Get success rate for action type.

        Args:
            action_type: Specific action type, or None for overall

        Returns:
            Success rate between 0 and 1
        """
        if action_type and action_type in self.success_rate:
            stats = self.success_rate[action_type]
            return stats['success'] / stats['total'] if stats['total'] > 0 else 0.0
        else:
            # Overall success rate
            total_commands = sum(stats['total'] for stats in self.success_rate.values())
            total_success = sum(stats['success'] for stats in self.success_rate.values())
            return total_success / total_commands if total_commands > 0 else 0.0

    def save_history(self, filepath: str) -> None:
        """Save command history to file.

        Args:
            filepath: Path to save file
        """
        data = {
            'history': self.history,
            'success_rate': self.success_rate
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_history(self, filepath: str) -> None:
        """Load command history from file.

        Args:
            filepath: Path to load file
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.history = data.get('history', [])
            self.success_rate = data.get('success_rate', {})

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not load command history: {e}")
