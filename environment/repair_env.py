"""
Pothole Repair Environment for DRL
Gymnasium-compatible environment
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple
import sys
sys.path.append('..')

# Import from your data folder (adjust paths as needed)
from pothole_generator import PotholeGenerator, Pothole
from road_network import RoadNetwork

class PotholeRepairEnv(gym.Env):
    """Custom Environment for Pothole Repair Scheduling"""
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, num_segments=50, episode_length=365, initial_budget=500000):
        super().__init__()
        
        self.num_segments = num_segments
        self.episode_length = episode_length
        self.initial_budget = initial_budget
        
        # Initialize components
        self.road_network = RoadNetwork(num_segments)
        self.pothole_gen = PotholeGenerator(num_segments)
        
        # Action space: [segment_id, repair_type]
        # repair_type: 0=do nothing, 1=patch, 2=overlay, 3=mill-and-fill
        self.action_space = spaces.MultiDiscrete([num_segments, 4])
        
        # Observation space
        # For each segment: [condition, num_potholes, traffic, avg_severity, budget_remaining, day, weather]
        obs_dim = num_segments * 4 + 3  # segment features + global features
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # State variables
        self.current_day = 0
        self.budget_remaining = initial_budget
        self.potholes = []
        self.weather = 'sunny'
        self.total_repairs = 0
        self.cumulative_reward = 0
        
        # Statistics tracking
        self.episode_stats = {
            'repairs': [],
            'costs': [],
            'safety_scores': [],
            'budget_used': []
        }
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset state
        self.current_day = 0
        self.budget_remaining = self.initial_budget
        self.total_repairs = 0
        self.cumulative_reward = 0
        
        # Generate initial potholes
        self.potholes = self.pothole_gen.generate_initial_potholes(density=0.3)
        
        # Update road network with potholes
        for segment in self.road_network.get_all_segments():
            segment.potholes = []
        for pothole in self.potholes:
            self.road_network.get_segment(pothole.segment_id).add_pothole(pothole.id)
            
        # Random initial weather
        self.weather = np.random.choice(['sunny', 'rainy', 'snowy', 'extreme'])
        
        # Reset statistics
        self.episode_stats = {
            'repairs': [],
            'costs': [],
            'safety_scores': [],
            'budget_used': []
        }
        
        return self._get_obs(), {}
        
    def step(self, action):
        """Execute one step in environment"""
        segment_id, repair_type = action
        
        # Validate action
        segment_id = min(segment_id, self.num_segments - 1)
        repair_type = min(repair_type, 3)
        
        reward = 0
        info = {}
        
        # Process repair action
        if repair_type > 0:  # repair action
            repair_cost, repair_quality, potholes_fixed = self._execute_repair(
                segment_id, repair_type
            )
            
            if repair_cost <= self.budget_remaining:
                # Apply repair
                self.budget_remaining -= repair_cost
                self.total_repairs += potholes_fixed
                
                # Update segment condition
                segment = self.road_network.get_segment(segment_id)
                segment.update_condition(repair_quality)
                
                # Calculate reward components
                safety_improvement = segment.get_safety_score() / 100
                cost_efficiency = repair_quality / (repair_cost + 1e-6)
                traffic_penalty = segment.get_traffic_disruption_cost() / 10000
                urgency_bonus = sum(self._get_pothole_by_id(pid).urgency 
                                for pid in segment.potholes[:potholes_fixed]) / max(1, potholes_fixed)

                # Improved weighted reward with urgency bonus
                reward = (
                    0.5 * safety_improvement * 100 +  # Amplified safety reward
                    0.2 * cost_efficiency * 50 +      # Reduced cost weight
                    0.1 * (1 - traffic_penalty) * 10 +
                    0.2 * (potholes_fixed * urgency_bonus)  # Bonus for fixing urgent potholes
                )
                
                info['repair_executed'] = True
                info['cost'] = repair_cost
                info['potholes_fixed'] = potholes_fixed
            else:
                # Insufficient budget penalty
                reward = -5
                info['repair_executed'] = False
                info['reason'] = 'insufficient_budget'
        else:
            # Do nothing action - small penalty if segment has critical potholes
            segment = self.road_network.get_segment(segment_id)
            critical_potholes = sum(1 for pid in segment.potholes 
                                   if self._get_pothole_by_id(pid).severity == 'critical')
            if critical_potholes > 0:
                reward = -2 * critical_potholes
            else:
                reward = 0.1  # small reward for conserving budget
                
        # Advance time
        self.current_day += 1
        
        # Deteriorate existing potholes
        weather_factor = {'sunny': 1.0, 'rainy': 1.5, 'snowy': 1.3, 'extreme': 2.0}[self.weather]
        for pothole in self.potholes:
            pothole.deteriorate(weather_factor)
            
        # Spawn new potholes
        new_potholes = self.pothole_gen.spawn_new_potholes(self.weather, self.current_day)
        for pothole in new_potholes:
            self.potholes.append(pothole)
            self.road_network.get_segment(pothole.segment_id).add_pothole(pothole.id)
            
        # Update weather randomly
        self.weather = np.random.choice(['sunny', 'rainy', 'snowy', 'extreme'], 
                                       p=[0.5, 0.3, 0.1, 0.1])
        
        # Update segment conditions
        for segment in self.road_network.get_all_segments():
            segment.update_condition()
            
        # Calculate network-wide penalty for poor conditions
        network_stats = self.road_network.get_network_statistics()
        safety_penalty = -0.1 * (100 - network_stats['avg_safety_score']) / 100
        reward += safety_penalty
        
        # Track statistics
        self.episode_stats['safety_scores'].append(network_stats['avg_safety_score'])
        self.episode_stats['budget_used'].append(self.initial_budget - self.budget_remaining)
        
        self.cumulative_reward += reward
        
        # Check termination
        terminated = self.current_day >= self.episode_length or self.budget_remaining <= 0
        truncated = False
        
        info['day'] = self.current_day
        info['budget_remaining'] = self.budget_remaining
        info['total_potholes'] = len(self.potholes)
        info['network_safety'] = network_stats['avg_safety_score']
        
        return self._get_obs(), reward, terminated, truncated, info
        
    def _execute_repair(self, segment_id: int, repair_type: int) -> Tuple[float, float, int]:
        """Execute repair and return cost, quality improvement, potholes fixed"""
        segment = self.road_network.get_segment(segment_id)
        
        # Repair type parameters
        repair_params = {
            1: {'cost_mult': 0.5, 'quality': 20, 'coverage': 0.5},   # patch
            2: {'cost_mult': 1.0, 'quality': 50, 'coverage': 0.8},   # overlay
            3: {'cost_mult': 1.5, 'quality': 80, 'coverage': 1.0}    # mill-and-fill
        }
        
        params = repair_params[repair_type]
        
        # Calculate cost based on segment potholes
        segment_potholes = [self._get_pothole_by_id(pid) for pid in segment.potholes]
        total_volume = sum(p.get_volume() for p in segment_potholes)
        base_cost = sum(p.get_repair_cost() for p in segment_potholes)
        
        cost = base_cost * params['cost_mult'] + segment.get_traffic_disruption_cost()
        quality = params['quality']
        
        # Fix potholes based on coverage
        num_to_fix = int(len(segment_potholes) * params['coverage'])
        potholes_fixed = 0
        
        # Prioritize by severity
        sorted_potholes = sorted(segment_potholes, key=lambda p: -p.urgency)
        for i, pothole in enumerate(sorted_potholes[:num_to_fix]):
            self.potholes.remove(pothole)
            segment.remove_pothole(pothole.id)
            potholes_fixed += 1
            
        return cost, quality, potholes_fixed
        
    def _get_pothole_by_id(self, pothole_id: int) -> Pothole:
        """Find pothole by ID"""
        for pothole in self.potholes:
            if pothole.id == pothole_id:
                return pothole
        return None
        
    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        
        # Segment features (normalized)
        for segment in self.road_network.get_all_segments():
            # Condition index (0-100) -> (0-1)
            obs.append(segment.condition_index / 100.0)
            
            # Number of potholes (normalized by max)
            obs.append(min(1.0, len(segment.potholes) / 10.0))
            
            # Traffic volume (normalized)
            obs.append(segment.traffic_volume / 50000.0)
            
            # Average severity (0-4 urgency scale)
            if segment.potholes:
                avg_severity = np.mean([self._get_pothole_by_id(pid).urgency 
                                       for pid in segment.potholes])
                obs.append(avg_severity / 4.0)
            else:
                obs.append(0.0)
                
        # Global features
        obs.append(self.budget_remaining / self.initial_budget)  # budget ratio
        obs.append(self.current_day / self.episode_length)  # time progress
        
        # Weather encoding
        weather_encoding = {'sunny': 0.25, 'rainy': 0.5, 'snowy': 0.75, 'extreme': 1.0}
        obs.append(weather_encoding[self.weather])
        
        return np.array(obs, dtype=np.float32)
        
    def render(self):
        """Render environment state"""
        if self.current_day % 30 == 0:  # Print monthly
            network_stats = self.road_network.get_network_statistics()
            print(f"\n=== Day {self.current_day} ===")
            print(f"Budget: ${self.budget_remaining:,.0f} / ${self.initial_budget:,.0f}")
            print(f"Total Potholes: {len(self.potholes)}")
            print(f"Network Safety: {network_stats['avg_safety_score']:.1f}")
            print(f"Avg Condition: {network_stats['avg_condition']:.1f}")
            print(f"Total Repairs: {self.total_repairs}")
            print(f"Cumulative Reward: {self.cumulative_reward:.2f}")
