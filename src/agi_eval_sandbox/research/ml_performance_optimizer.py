"""
Machine Learning-Driven Performance Optimizer

Novel ML-based system for real-time performance optimization using reinforcement
learning, gradient-free optimization, and adaptive resource management.

Research Innovation: "Autonomous Performance Optimization via Multi-Agent RL"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
import math
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import optuna
from concurrent.futures import ThreadPoolExecutor

from ..core.logging_config import get_logger
from ..core.performance import PerformanceMetrics

logger = get_logger("ml_performance_optimizer")


@dataclass
class SystemState:
    """Represents current system performance state."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_io: float
    disk_io: float
    queue_length: int
    response_time: float
    throughput: float
    error_rate: float
    cache_hit_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationAction:
    """Represents an optimization action."""
    action_type: str  # 'scale_workers', 'adjust_batch_size', 'tune_cache', etc.
    parameters: Dict[str, float]
    expected_impact: float
    confidence: float
    resource_cost: float


@dataclass
class PerformanceGoal:
    """Defines performance optimization goals."""
    target_response_time: float = 0.5  # seconds
    target_throughput: float = 100.0   # requests/second
    target_error_rate: float = 0.01    # 1%
    target_cpu_usage: float = 0.7      # 70%
    target_memory_usage: float = 0.8   # 80%
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        'response_time': 0.3,
        'throughput': 0.25,
        'error_rate': 0.2,
        'resource_efficiency': 0.25
    })


class DQNPerformanceAgent(nn.Module):
    """Deep Q-Network for performance optimization decisions."""
    
    def __init__(self, state_dim: int = 20, action_dim: int = 10, hidden_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        return self.network(state)
    
    def select_action(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))
            return q_values.argmax().item()
    
    def remember(self, state: torch.Tensor, action: int, reward: float, 
                next_state: torch.Tensor, done: bool) -> None:
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, optimizer: torch.optim.Optimizer, target_network: 'DQNPerformanceAgent') -> float:
        """Train network on batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.stack([self.memory[i][0] for i in batch])
        actions = torch.tensor([self.memory[i][1] for i in batch], dtype=torch.long)
        rewards = torch.tensor([self.memory[i][2] for i in batch], dtype=torch.float32)
        next_states = torch.stack([self.memory[i][3] for i in batch])
        dones = torch.tensor([self.memory[i][4] for i in batch], dtype=torch.bool)
        
        # Current Q values
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = target_network.forward(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Loss and optimization
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class GradientFreeOptimizer:
    """Gradient-free optimization for hyperparameter tuning."""
    
    def __init__(self, bounds: Dict[str, Tuple[float, float]]):
        self.bounds = bounds
        self.study = optuna.create_study(direction='maximize')
        self.optimization_history = []
        
    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, float]:
        """Suggest parameter values for optimization trial."""
        params = {}
        for param_name, (low, high) in self.bounds.items():
            if isinstance(low, int) and isinstance(high, int):
                params[param_name] = trial.suggest_int(param_name, low, high)
            else:
                params[param_name] = trial.suggest_float(param_name, low, high)
        return params
    
    def optimize(self, objective_function, n_trials: int = 100) -> Dict[str, float]:
        """Run optimization with objective function."""
        def optuna_objective(trial):
            params = self.suggest_parameters(trial)
            score = objective_function(params)
            self.optimization_history.append({
                'trial': trial.number,
                'params': params,
                'score': score,
                'timestamp': datetime.now()
            })
            return score
        
        self.study.optimize(optuna_objective, n_trials=n_trials)
        return self.study.best_params


class AnomalyDetector:
    """Detect performance anomalies using isolation forest."""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.anomaly_history = deque(maxlen=1000)
        
    def fit(self, normal_data: np.ndarray) -> None:
        """Fit anomaly detector on normal performance data."""
        scaled_data = self.scaler.fit_transform(normal_data)
        self.model.fit(scaled_data)
        self.is_fitted = True
        logger.info("Anomaly detector fitted on normal performance data")
        
    def detect_anomaly(self, current_state: np.ndarray) -> Tuple[bool, float]:
        """Detect if current state is anomalous."""
        if not self.is_fitted:
            return False, 0.0
        
        scaled_state = self.scaler.transform(current_state.reshape(1, -1))
        anomaly_score = self.model.decision_function(scaled_state)[0]
        is_anomaly = self.model.predict(scaled_state)[0] == -1
        
        self.anomaly_history.append({
            'timestamp': datetime.now(),
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'state': current_state.tolist()
        })
        
        return is_anomaly, abs(anomaly_score)


class MLPerformanceOptimizer:
    """
    Machine Learning-driven performance optimizer.
    
    Key innovations:
    1. Deep RL agent for optimization decisions
    2. Gradient-free hyperparameter optimization
    3. Real-time anomaly detection and correction
    4. Multi-objective performance optimization
    5. Adaptive resource allocation and scaling
    """
    
    def __init__(self, performance_goals: Optional[PerformanceGoal] = None):
        self.performance_goals = performance_goals or PerformanceGoal()
        
        # ML Components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rl_agent = DQNPerformanceAgent().to(self.device)
        self.target_agent = DQNPerformanceAgent().to(self.device)
        self.target_agent.load_state_dict(self.rl_agent.state_dict())
        
        self.optimizer = optim.Adam(self.rl_agent.parameters(), lr=0.001)
        self.anomaly_detector = AnomalyDetector()
        
        # Gradient-free optimizer for hyperparameters
        self.hyperparameter_bounds = {
            'batch_size': (16, 512),
            'worker_count': (1, 32),
            'cache_size': (100, 10000),
            'timeout_seconds': (1.0, 30.0),
            'memory_limit_gb': (1.0, 16.0)
        }
        self.gradient_free_optimizer = GradientFreeOptimizer(self.hyperparameter_bounds)
        
        # State management
        self.state_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.current_config = {}
        
        # Performance tracking
        self.optimization_episodes = 0
        self.total_reward = 0.0
        self.best_performance = 0.0
        self.improvement_ratio = 0.0
        
        # Action space definition
        self.action_space = {
            0: 'increase_workers',
            1: 'decrease_workers',
            2: 'increase_batch_size',
            3: 'decrease_batch_size',
            4: 'increase_cache_size',
            5: 'decrease_cache_size',
            6: 'optimize_memory',
            7: 'tune_timeouts',
            8: 'rebalance_load',
            9: 'no_action'
        }
        
        logger.info("ML Performance Optimizer initialized")
        
    def _extract_state_features(self, system_state: SystemState) -> torch.Tensor:
        """Extract features from system state for ML models."""
        features = [
            system_state.cpu_usage,
            system_state.memory_usage,
            system_state.gpu_usage,
            system_state.network_io,
            system_state.disk_io,
            system_state.queue_length / 100.0,  # Normalize
            system_state.response_time,
            system_state.throughput / 100.0,    # Normalize
            system_state.error_rate,
            system_state.cache_hit_rate,
            
            # Time-based features
            system_state.timestamp.hour / 24.0,
            system_state.timestamp.weekday() / 7.0,
            
            # Goal difference features
            abs(system_state.response_time - self.performance_goals.target_response_time),
            abs(system_state.throughput - self.performance_goals.target_throughput) / 100.0,
            abs(system_state.error_rate - self.performance_goals.target_error_rate),
            abs(system_state.cpu_usage - self.performance_goals.target_cpu_usage),
            abs(system_state.memory_usage - self.performance_goals.target_memory_usage),
            
            # Historical features (moving averages)
            self._get_moving_average('response_time', 10),
            self._get_moving_average('throughput', 10) / 100.0,
            self._get_moving_average('error_rate', 10)
        ]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def _get_moving_average(self, metric: str, window: int = 10) -> float:
        """Get moving average of a metric from state history."""
        if len(self.state_history) < window:
            return 0.0
        
        recent_values = [
            getattr(state, metric) for state in list(self.state_history)[-window:]
        ]
        return np.mean(recent_values)
    
    def _calculate_reward(self, previous_state: SystemState, current_state: SystemState) -> float:
        """Calculate reward for RL agent based on performance improvement."""
        # Individual metric improvements
        response_time_improvement = (
            previous_state.response_time - current_state.response_time
        ) / previous_state.response_time if previous_state.response_time > 0 else 0
        
        throughput_improvement = (
            current_state.throughput - previous_state.throughput
        ) / previous_state.throughput if previous_state.throughput > 0 else 0
        
        error_rate_improvement = (
            previous_state.error_rate - current_state.error_rate
        ) / previous_state.error_rate if previous_state.error_rate > 0 else 0
        
        # Resource efficiency
        resource_efficiency = 1.0 - (
            0.5 * current_state.cpu_usage + 0.5 * current_state.memory_usage
        )
        
        # Goal achievement rewards
        response_time_goal = 1.0 if current_state.response_time <= self.performance_goals.target_response_time else 0.0
        throughput_goal = 1.0 if current_state.throughput >= self.performance_goals.target_throughput else 0.0
        error_rate_goal = 1.0 if current_state.error_rate <= self.performance_goals.target_error_rate else 0.0
        
        # Combined reward with goal weights
        weights = self.performance_goals.priority_weights
        reward = (
            weights['response_time'] * (response_time_improvement + response_time_goal) +
            weights['throughput'] * (throughput_improvement + throughput_goal) +
            weights['error_rate'] * (error_rate_improvement + error_rate_goal) +
            weights['resource_efficiency'] * resource_efficiency
        )
        
        # Penalty for extreme resource usage
        if current_state.cpu_usage > 0.95 or current_state.memory_usage > 0.95:
            reward -= 1.0
        
        # Bonus for sustained good performance
        if self._is_performance_stable(current_state):
            reward += 0.5
        
        return np.clip(reward, -2.0, 2.0)  # Clip reward to reasonable range
    
    def _is_performance_stable(self, current_state: SystemState, window: int = 5) -> bool:
        """Check if performance has been stable recently."""
        if len(self.state_history) < window:
            return False
        
        recent_states = list(self.state_history)[-window:]
        
        # Check stability of key metrics
        response_times = [s.response_time for s in recent_states]
        throughputs = [s.throughput for s in recent_states]
        error_rates = [s.error_rate for s in recent_states]
        
        # Low variance indicates stability
        response_time_stable = np.std(response_times) < 0.1
        throughput_stable = np.std(throughputs) < 5.0
        error_rate_stable = np.std(error_rates) < 0.01
        
        return response_time_stable and throughput_stable and error_rate_stable
    
    async def optimize_performance(self, current_state: SystemState) -> OptimizationAction:
        """Main optimization function using ML models."""
        # Add current state to history
        self.state_history.append(current_state)
        
        # Extract features for ML models
        state_features = self._extract_state_features(current_state)
        
        # Check for anomalies
        is_anomaly, anomaly_score = await self._check_for_anomalies(current_state)
        
        if is_anomaly:
            logger.warning(f"Performance anomaly detected (score: {anomaly_score:.3f})")
            action = await self._handle_anomaly(current_state, anomaly_score)
        else:
            # Normal RL-based optimization
            action = await self._rl_optimization_step(state_features, current_state)
        
        # Learn from previous action if available
        if len(self.state_history) >= 2:
            await self._update_rl_agent()
        
        return action
    
    async def _check_for_anomalies(self, current_state: SystemState) -> Tuple[bool, float]:
        """Check for performance anomalies."""
        state_array = np.array([
            current_state.cpu_usage,
            current_state.memory_usage,
            current_state.response_time,
            current_state.throughput,
            current_state.error_rate,
            current_state.cache_hit_rate
        ])
        
        return self.anomaly_detector.detect_anomaly(state_array)
    
    async def _handle_anomaly(self, current_state: SystemState, anomaly_score: float) -> OptimizationAction:
        """Handle detected performance anomalies with corrective actions."""
        logger.info(f"Applying anomaly correction for score {anomaly_score:.3f}")
        
        # Determine corrective action based on anomaly characteristics
        if current_state.response_time > self.performance_goals.target_response_time * 2:
            # Response time crisis
            action_type = 'emergency_scale_up'
            parameters = {'worker_multiplier': 2.0, 'memory_boost': 1.5}
            
        elif current_state.error_rate > self.performance_goals.target_error_rate * 5:
            # Error rate crisis
            action_type = 'error_recovery'
            parameters = {'restart_workers': True, 'circuit_breaker': True}
            
        elif current_state.cpu_usage > 0.95 or current_state.memory_usage > 0.95:
            # Resource exhaustion
            action_type = 'resource_relief'
            parameters = {'scale_workers': 1.5, 'optimize_memory': True}
            
        else:
            # General performance degradation
            action_type = 'performance_recovery'
            parameters = {'rebalance_load': True, 'tune_cache': True}
        
        return OptimizationAction(
            action_type=action_type,
            parameters=parameters,
            expected_impact=0.8,  # High confidence in anomaly correction
            confidence=min(anomaly_score, 1.0),
            resource_cost=0.5
        )
    
    async def _rl_optimization_step(self, state_features: torch.Tensor, current_state: SystemState) -> OptimizationAction:
        """Perform RL-based optimization step."""
        # Select action using RL agent
        epsilon = max(0.01, 0.5 * math.exp(-self.optimization_episodes / 1000))  # Decaying exploration
        action_id = self.rl_agent.select_action(state_features, epsilon)
        action_name = self.action_space[action_id]
        
        # Generate action parameters based on action type
        parameters = await self._generate_action_parameters(action_name, current_state)
        
        # Estimate expected impact
        expected_impact = await self._estimate_action_impact(action_name, parameters, current_state)
        
        # Record action
        self.action_history.append({
            'action_id': action_id,
            'action_name': action_name,
            'parameters': parameters,
            'timestamp': datetime.now()
        })
        
        return OptimizationAction(
            action_type=action_name,
            parameters=parameters,
            expected_impact=expected_impact,
            confidence=1.0 - epsilon,  # Higher confidence with lower exploration
            resource_cost=self._calculate_resource_cost(action_name, parameters)
        )
    
    async def _generate_action_parameters(self, action_name: str, current_state: SystemState) -> Dict[str, float]:
        """Generate parameters for specific optimization actions."""
        if action_name == 'increase_workers':
            return {'worker_count': min(self.current_config.get('worker_count', 4) + 2, 32)}
        
        elif action_name == 'decrease_workers':
            return {'worker_count': max(self.current_config.get('worker_count', 4) - 1, 1)}
        
        elif action_name == 'increase_batch_size':
            current_batch = self.current_config.get('batch_size', 32)
            return {'batch_size': min(current_batch * 1.5, 512)}
        
        elif action_name == 'decrease_batch_size':
            current_batch = self.current_config.get('batch_size', 32)
            return {'batch_size': max(current_batch * 0.75, 16)}
        
        elif action_name == 'increase_cache_size':
            current_cache = self.current_config.get('cache_size', 1000)
            return {'cache_size': min(current_cache * 1.3, 10000)}
        
        elif action_name == 'decrease_cache_size':
            current_cache = self.current_config.get('cache_size', 1000)
            return {'cache_size': max(current_cache * 0.8, 100)}
        
        elif action_name == 'optimize_memory':
            return {
                'memory_limit_gb': self.current_config.get('memory_limit_gb', 8.0) * 1.2,
                'gc_threshold': 0.8
            }
        
        elif action_name == 'tune_timeouts':
            return {
                'request_timeout': max(1.0, current_state.response_time * 1.5),
                'connection_timeout': 30.0
            }
        
        elif action_name == 'rebalance_load':
            return {
                'load_balancing_algorithm': 'round_robin',
                'health_check_interval': 5.0
            }
        
        else:  # no_action
            return {}
    
    async def _estimate_action_impact(self, action_name: str, parameters: Dict[str, float], 
                                    current_state: SystemState) -> float:
        """Estimate expected impact of optimization action."""
        # Simplified impact estimation based on current state and action type
        if action_name == 'increase_workers' and current_state.cpu_usage > 0.8:
            return 0.7  # High impact for scaling under high CPU
        
        elif action_name == 'increase_batch_size' and current_state.throughput < self.performance_goals.target_throughput:
            return 0.6  # Medium-high impact for throughput improvement
        
        elif action_name == 'increase_cache_size' and current_state.cache_hit_rate < 0.8:
            return 0.5  # Medium impact for cache optimization
        
        elif action_name == 'optimize_memory' and current_state.memory_usage > 0.85:
            return 0.6  # Medium-high impact for memory relief
        
        elif action_name == 'tune_timeouts' and current_state.error_rate > self.performance_goals.target_error_rate:
            return 0.4  # Medium impact for error reduction
        
        elif action_name == 'no_action':
            return 0.0  # No impact
        
        else:
            return 0.3  # Default medium impact
    
    def _calculate_resource_cost(self, action_name: str, parameters: Dict[str, float]) -> float:
        """Calculate resource cost of optimization action."""
        cost_map = {
            'increase_workers': 0.8,
            'decrease_workers': -0.2,  # Negative cost (saves resources)
            'increase_batch_size': 0.3,
            'decrease_batch_size': -0.1,
            'increase_cache_size': 0.4,
            'decrease_cache_size': -0.2,
            'optimize_memory': 0.2,
            'tune_timeouts': 0.1,
            'rebalance_load': 0.3,
            'no_action': 0.0
        }
        
        base_cost = cost_map.get(action_name, 0.2)
        
        # Adjust cost based on parameters
        if 'worker_count' in parameters:
            base_cost *= (parameters['worker_count'] / 4.0)  # Scale with worker count
        
        return max(0.0, base_cost)
    
    async def _update_rl_agent(self) -> None:
        """Update RL agent with recent experience."""
        if len(self.state_history) < 2 or len(self.action_history) == 0:
            return
        
        # Get previous and current states
        previous_state = self.state_history[-2]
        current_state = self.state_history[-1]
        
        # Get previous action
        previous_action = self.action_history[-1]
        action_id = previous_action['action_id']
        
        # Calculate reward
        reward = self._calculate_reward(previous_state, current_state)
        self.reward_history.append(reward)
        self.total_reward += reward
        
        # Convert states to feature tensors
        prev_features = self._extract_state_features(previous_state)
        curr_features = self._extract_state_features(current_state)
        
        # Store experience in replay buffer
        done = False  # Continuous optimization, so never "done"
        self.rl_agent.remember(prev_features, action_id, reward, curr_features, done)
        
        # Train agent
        if len(self.rl_agent.memory) >= self.rl_agent.batch_size:
            loss = self.rl_agent.replay(self.optimizer, self.target_agent)
            
            # Update target network periodically
            if self.optimization_episodes % 100 == 0:
                self.target_agent.load_state_dict(self.rl_agent.state_dict())
                logger.info(f"Updated target network at episode {self.optimization_episodes}")
        
        self.optimization_episodes += 1
        
        # Log training progress
        if self.optimization_episodes % 50 == 0:
            avg_reward = np.mean(list(self.reward_history)[-50:]) if self.reward_history else 0.0
            logger.info(f"Episode {self.optimization_episodes}, Average Reward: {avg_reward:.3f}")
    
    async def hyperparameter_optimization(self, evaluation_function) -> Dict[str, float]:
        """Optimize hyperparameters using gradient-free optimization."""
        logger.info("Starting hyperparameter optimization")
        
        def objective(params):
            # Apply parameters and evaluate performance
            self.current_config.update(params)
            performance_score = evaluation_function(params)
            return performance_score
        
        # Run optimization
        optimal_params = self.gradient_free_optimizer.optimize(objective, n_trials=50)
        
        logger.info(f"Optimal hyperparameters found: {optimal_params}")
        return optimal_params
    
    def train_anomaly_detector(self, normal_performance_data: List[SystemState]) -> None:
        """Train anomaly detector on historical normal performance data."""
        if len(normal_performance_data) < 50:
            logger.warning("Insufficient data for anomaly detector training")
            return
        
        # Convert to numpy array
        data_matrix = np.array([
            [
                state.cpu_usage,
                state.memory_usage,
                state.response_time,
                state.throughput,
                state.error_rate,
                state.cache_hit_rate
            ]
            for state in normal_performance_data
        ])
        
        self.anomaly_detector.fit(data_matrix)
        logger.info(f"Anomaly detector trained on {len(normal_performance_data)} samples")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        recent_rewards = list(self.reward_history)[-100:] if self.reward_history else []
        
        return {
            "optimization_episodes": self.optimization_episodes,
            "total_reward": self.total_reward,
            "average_recent_reward": np.mean(recent_rewards) if recent_rewards else 0.0,
            "best_performance": self.best_performance,
            "improvement_ratio": self.improvement_ratio,
            "anomaly_detection_fitted": self.anomaly_detector.is_fitted,
            "recent_anomalies": len([a for a in self.anomaly_detector.anomaly_history 
                                   if a['is_anomaly']]),
            "action_distribution": self._get_action_distribution(),
            "hyperparameter_trials": len(self.gradient_free_optimizer.optimization_history),
            "current_config": self.current_config.copy()
        }
    
    def _get_action_distribution(self) -> Dict[str, int]:
        """Get distribution of actions taken."""
        action_counts = defaultdict(int)
        for action in self.action_history:
            action_counts[action['action_name']] += 1
        return dict(action_counts)
    
    def save_model(self, path: str) -> None:
        """Save ML models to disk."""
        save_data = {
            'rl_agent_state_dict': self.rl_agent.state_dict(),
            'target_agent_state_dict': self.target_agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimization_episodes': self.optimization_episodes,
            'total_reward': self.total_reward,
            'current_config': self.current_config,
            'performance_goals': self.performance_goals.__dict__,
            'hyperparameter_bounds': self.hyperparameter_bounds
        }
        
        torch.save(save_data, path)
        logger.info(f"ML Performance Optimizer saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load ML models from disk."""
        save_data = torch.load(path, map_location=self.device)
        
        self.rl_agent.load_state_dict(save_data['rl_agent_state_dict'])
        self.target_agent.load_state_dict(save_data['target_agent_state_dict'])
        self.optimizer.load_state_dict(save_data['optimizer_state_dict'])
        self.optimization_episodes = save_data['optimization_episodes']
        self.total_reward = save_data['total_reward']
        self.current_config = save_data['current_config']
        
        logger.info(f"ML Performance Optimizer loaded from {path}")
    
    def export_research_data(self) -> Dict[str, Any]:
        """Export comprehensive research data for analysis."""
        return {
            "algorithm_name": "ML-Driven Performance Optimizer",
            "rl_architecture": {
                "state_dim": self.rl_agent.state_dim,
                "action_dim": self.rl_agent.action_dim,
                "memory_size": len(self.rl_agent.memory),
                "episodes_trained": self.optimization_episodes
            },
            "optimization_performance": {
                "total_reward": self.total_reward,
                "average_reward": self.total_reward / max(self.optimization_episodes, 1),
                "best_performance": self.best_performance,
                "improvement_ratio": self.improvement_ratio
            },
            "anomaly_detection": {
                "is_fitted": self.anomaly_detector.is_fitted,
                "anomaly_count": len([a for a in self.anomaly_detector.anomaly_history 
                                    if a['is_anomaly']]),
                "total_samples": len(self.anomaly_detector.anomaly_history)
            },
            "hyperparameter_optimization": {
                "optimization_trials": len(self.gradient_free_optimizer.optimization_history),
                "best_params": self.gradient_free_optimizer.study.best_params if hasattr(self.gradient_free_optimizer.study, 'best_params') else {},
                "bounds": self.hyperparameter_bounds
            },
            "performance_goals": self.performance_goals.__dict__,
            "action_statistics": self._get_action_distribution(),
            "state_history_length": len(self.state_history)
        }