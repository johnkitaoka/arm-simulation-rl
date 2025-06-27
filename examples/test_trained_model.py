#!/usr/bin/env python3
"""
Test Trained Model Script

This script allows you to test and evaluate trained ML models
for robot arm control with various scenarios and metrics.
"""

import sys
import os
import argparse
import numpy as np
import time
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_arm.robot_arm import RobotArm
from physics.physics_engine import PhysicsEngine
from ml.rl_trainer import RLTrainer, RobotArmEnv
from ml.nlp_processor import CommandParser


class ModelTester:
    """Test and evaluate trained ML models."""
    
    def __init__(self, model_path: str, algorithm: str = "PPO"):
        """Initialize model tester.
        
        Args:
            model_path: Path to trained model
            algorithm: Algorithm used for training
        """
        self.model_path = model_path
        self.algorithm = algorithm
        
        # Initialize components
        self.robot_arm = RobotArm()
        self.physics_engine = PhysicsEngine()
        self.trainer = RLTrainer(self.robot_arm, self.physics_engine)
        
        # Load model
        self.load_model()
        
        # Test scenarios
        self.test_scenarios = self._create_test_scenarios()
        
    def load_model(self):
        """Load the trained model."""
        try:
            self.trainer.load_model(self.model_path, self.algorithm)
            print(f"✅ Model loaded successfully: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _create_test_scenarios(self) -> List[Dict[str, Any]]:
        """Create diverse test scenarios."""
        scenarios = [
            # Easy scenarios
            {
                'name': 'Close Target',
                'target': [0.2, 0.0, 0.3],
                'difficulty': 'Easy',
                'expected_success': 0.95
            },
            {
                'name': 'Medium Distance',
                'target': [0.4, 0.1, 0.4],
                'difficulty': 'Medium',
                'expected_success': 0.85
            },
            {
                'name': 'Far Target',
                'target': [0.6, 0.0, 0.5],
                'difficulty': 'Hard',
                'expected_success': 0.70
            },
            # Corner cases
            {
                'name': 'High Target',
                'target': [0.3, 0.0, 0.7],
                'difficulty': 'Hard',
                'expected_success': 0.60
            },
            {
                'name': 'Side Target',
                'target': [0.2, 0.4, 0.3],
                'difficulty': 'Medium',
                'expected_success': 0.75
            },
            {
                'name': 'Low Target',
                'target': [0.3, 0.0, 0.1],
                'difficulty': 'Medium',
                'expected_success': 0.80
            },
            # Challenging scenarios
            {
                'name': 'Workspace Edge',
                'target': [0.8, 0.0, 0.3],
                'difficulty': 'Very Hard',
                'expected_success': 0.40
            },
            {
                'name': 'Behind Robot',
                'target': [-0.2, 0.0, 0.3],
                'difficulty': 'Hard',
                'expected_success': 0.50
            }
        ]
        return scenarios
    
    def test_single_scenario(self, scenario: Dict[str, Any], num_episodes: int = 10) -> Dict[str, Any]:
        """Test model on a single scenario.
        
        Args:
            scenario: Test scenario configuration
            num_episodes: Number of episodes to run
            
        Returns:
            Test results
        """
        print(f"\n🎯 Testing scenario: {scenario['name']}")
        print(f"   Target: {scenario['target']}")
        print(f"   Difficulty: {scenario['difficulty']}")
        print(f"   Expected success: {scenario['expected_success']:.1%}")
        
        results = {
            'scenario_name': scenario['name'],
            'target_position': scenario['target'],
            'episodes': [],
            'success_count': 0,
            'total_episodes': num_episodes,
            'average_reward': 0.0,
            'average_steps': 0.0,
            'average_final_distance': 0.0,
            'success_rate': 0.0
        }
        
        total_reward = 0.0
        total_steps = 0
        total_final_distance = 0.0
        
        for episode in range(num_episodes):
            # Set target for this scenario
            self.trainer.env.target_position = np.array(scenario['target'])
            
            # Run episode
            obs, _ = self.trainer.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            
            for step in range(500):  # Max steps per episode
                action, _ = self.trainer.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.trainer.env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if terminated or truncated:
                    break
            
            # Calculate final distance
            final_distance = info.get('distance_to_target', float('inf'))
            success = final_distance < 0.05  # 5cm tolerance
            
            if success:
                results['success_count'] += 1
            
            # Store episode results
            episode_result = {
                'episode': episode + 1,
                'reward': episode_reward,
                'steps': episode_steps,
                'final_distance': final_distance,
                'success': success
            }
            results['episodes'].append(episode_result)
            
            # Accumulate totals
            total_reward += episode_reward
            total_steps += episode_steps
            total_final_distance += final_distance
            
            # Print progress
            status = "✅" if success else "❌"
            print(f"   Episode {episode + 1:2d}: {status} "
                  f"Reward: {episode_reward:6.1f}, "
                  f"Steps: {episode_steps:3d}, "
                  f"Distance: {final_distance:.3f}m")
        
        # Calculate averages
        results['average_reward'] = total_reward / num_episodes
        results['average_steps'] = total_steps / num_episodes
        results['average_final_distance'] = total_final_distance / num_episodes
        results['success_rate'] = results['success_count'] / num_episodes
        
        # Print summary
        print(f"\n📊 Scenario Results:")
        print(f"   Success Rate: {results['success_rate']:.1%} "
              f"({results['success_count']}/{num_episodes})")
        print(f"   Average Reward: {results['average_reward']:.1f}")
        print(f"   Average Steps: {results['average_steps']:.1f}")
        print(f"   Average Final Distance: {results['average_final_distance']:.3f}m")
        
        # Compare to expected
        expected = scenario['expected_success']
        if results['success_rate'] >= expected:
            print(f"   ✅ Meets expectations ({expected:.1%})")
        else:
            print(f"   ⚠️  Below expectations ({expected:.1%})")
        
        return results
    
    def test_all_scenarios(self, episodes_per_scenario: int = 10) -> Dict[str, Any]:
        """Test model on all scenarios.
        
        Args:
            episodes_per_scenario: Number of episodes per scenario
            
        Returns:
            Complete test results
        """
        print(f"🧪 Testing model on {len(self.test_scenarios)} scenarios")
        print(f"   Model: {self.model_path}")
        print(f"   Algorithm: {self.algorithm}")
        print(f"   Episodes per scenario: {episodes_per_scenario}")
        
        all_results = {
            'model_path': self.model_path,
            'algorithm': self.algorithm,
            'total_scenarios': len(self.test_scenarios),
            'episodes_per_scenario': episodes_per_scenario,
            'scenario_results': [],
            'overall_success_rate': 0.0,
            'overall_average_reward': 0.0
        }
        
        total_successes = 0
        total_episodes = 0
        total_reward = 0.0
        
        start_time = time.time()
        
        for i, scenario in enumerate(self.test_scenarios):
            print(f"\n{'='*60}")
            print(f"Scenario {i+1}/{len(self.test_scenarios)}")
            
            scenario_results = self.test_single_scenario(scenario, episodes_per_scenario)
            all_results['scenario_results'].append(scenario_results)
            
            # Accumulate overall stats
            total_successes += scenario_results['success_count']
            total_episodes += scenario_results['total_episodes']
            total_reward += scenario_results['average_reward'] * scenario_results['total_episodes']
        
        test_time = time.time() - start_time
        
        # Calculate overall metrics
        all_results['overall_success_rate'] = total_successes / total_episodes
        all_results['overall_average_reward'] = total_reward / total_episodes
        all_results['test_duration'] = test_time
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"🏆 FINAL TEST RESULTS")
        print(f"{'='*60}")
        print(f"Model: {self.model_path}")
        print(f"Overall Success Rate: {all_results['overall_success_rate']:.1%}")
        print(f"Overall Average Reward: {all_results['overall_average_reward']:.1f}")
        print(f"Total Episodes: {total_episodes}")
        print(f"Test Duration: {test_time:.1f} seconds")
        
        # Performance rating
        success_rate = all_results['overall_success_rate']
        if success_rate >= 0.85:
            rating = "🌟 Excellent"
        elif success_rate >= 0.70:
            rating = "✅ Good"
        elif success_rate >= 0.50:
            rating = "⚠️  Fair"
        else:
            rating = "❌ Poor"
        
        print(f"Performance Rating: {rating}")
        
        # Scenario breakdown
        print(f"\n📋 Scenario Breakdown:")
        for result in all_results['scenario_results']:
            name = result['scenario_name']
            success_rate = result['success_rate']
            print(f"   {name:15s}: {success_rate:.1%}")
        
        return all_results
    
    def benchmark_performance(self, num_episodes: int = 100) -> Dict[str, Any]:
        """Benchmark model performance and speed.
        
        Args:
            num_episodes: Number of episodes for benchmarking
            
        Returns:
            Benchmark results
        """
        print(f"\n⚡ Benchmarking model performance...")
        print(f"   Episodes: {num_episodes}")
        
        # Standard reaching task
        self.trainer.env.target_position = np.array([0.3, 0.0, 0.3])
        
        start_time = time.time()
        inference_times = []
        
        for episode in range(num_episodes):
            obs, _ = self.trainer.env.reset()
            
            for step in range(200):  # Max steps
                # Time inference
                inference_start = time.time()
                action, _ = self.trainer.model.predict(obs, deterministic=True)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                obs, reward, terminated, truncated, info = self.trainer.env.step(action)
                
                if terminated or truncated:
                    break
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        std_inference_time = np.std(inference_times) * 1000
        max_inference_time = np.max(inference_times) * 1000
        
        benchmark_results = {
            'total_episodes': num_episodes,
            'total_time': total_time,
            'episodes_per_second': num_episodes / total_time,
            'average_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': std_inference_time,
            'max_inference_time_ms': max_inference_time,
            'total_inferences': len(inference_times)
        }
        
        print(f"📊 Benchmark Results:")
        print(f"   Episodes per second: {benchmark_results['episodes_per_second']:.1f}")
        print(f"   Average inference time: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
        print(f"   Max inference time: {max_inference_time:.2f} ms")
        print(f"   Total inferences: {len(inference_times)}")
        
        # Performance assessment
        if avg_inference_time < 1.0:
            perf_rating = "🚀 Excellent (Real-time capable)"
        elif avg_inference_time < 5.0:
            perf_rating = "✅ Good (Near real-time)"
        elif avg_inference_time < 20.0:
            perf_rating = "⚠️  Fair (Acceptable for most uses)"
        else:
            perf_rating = "❌ Poor (Too slow for real-time)"
        
        print(f"   Performance Rating: {perf_rating}")
        
        return benchmark_results


def main():
    """Main function for testing trained models."""
    parser = argparse.ArgumentParser(description="Test trained ML models for robot arm control")
    parser.add_argument("--model", required=True, help="Path to trained model (.zip file)")
    parser.add_argument("--algorithm", default="PPO", choices=["PPO", "SAC", "TD3"],
                       help="Algorithm used for training")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes per scenario")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")
    parser.add_argument("--scenario", type=str,
                       help="Test specific scenario only")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return 1
    
    try:
        # Create tester
        tester = ModelTester(args.model, args.algorithm)
        
        if args.benchmark:
            # Run performance benchmark
            benchmark_results = tester.benchmark_performance(100)
            
        if args.scenario:
            # Test specific scenario
            scenario = next((s for s in tester.test_scenarios if s['name'].lower() == args.scenario.lower()), None)
            if scenario:
                tester.test_single_scenario(scenario, args.episodes)
            else:
                print(f"❌ Scenario '{args.scenario}' not found")
                print("Available scenarios:")
                for s in tester.test_scenarios:
                    print(f"   - {s['name']}")
                return 1
        else:
            # Test all scenarios
            results = tester.test_all_scenarios(args.episodes)
        
        print(f"\n✅ Testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
