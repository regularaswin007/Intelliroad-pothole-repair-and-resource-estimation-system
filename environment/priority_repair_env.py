"""
DRL Prioritization Agent - Learns optimal repair ordering
Agent sees list of potholes and decides priority scores
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct imports without package prefix
import pothole_generator
import road_network

class PriorityRepairEnv(gym.Env):
    """
    DRL Agent learns to assign priority scores to potholes.
    Repairs happen in priority order.
    """
    
    def __init__(self, num_segments=20, max_potholes=50, episode_length=100):
        super().__init__()
        
        self.num_segments = num_segments
        self.max_potholes = max_potholes
        self.episode_length = episode_length
        
        self.road_network = road_network.RoadNetwork(num_segments)
        self.pothole_gen = pothole_generator.PotholeGenerator(num_segments)
        
        # Action: Priority score for NEXT pothole (0.0 to 1.0)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation: Features of current pothole
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )
        
        self.current_step = 0
        self.potholes = []
        self.pothole_priorities = {}
        self.current_pothole_idx = 0
        self.last_repair_segment = 0
        self.total_repaired = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_repaired = 0
        self.current_pothole_idx = 0
        self.last_repair_segment = 0
        self.pothole_priorities = {}
        
        # Generate potholes
        self.potholes = self.pothole_gen.generate_initial_potholes(density=0.6)
        
        # Update segments
        for segment in self.road_network.get_all_segments():
            segment.potholes = []
        for pothole in self.potholes:
            self.road_network.get_segment(pothole.segment_id).add_pothole(pothole.id)
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Agent assigns priority score to current pothole"""
        
        priority_score = float(action[0])
        
        # Assign priority to current pothole
        if self.current_pothole_idx < len(self.potholes):
            current_pothole = self.potholes[self.current_pothole_idx]
            self.pothole_priorities[current_pothole.id] = priority_score
            self.current_pothole_idx += 1
        
        # If all potholes scored, execute repairs
        if self.current_pothole_idx >= len(self.potholes):
            reward = self._execute_repair_phase()
            terminated = True
            
            info = {
                'total_repaired': self.total_repaired,
                'total_potholes': len(self.potholes),
                'repair_efficiency': self.total_repaired / max(1, len(self.potholes))
            }
            
            return self._get_obs(), reward, terminated, False, info
        
        # Still scoring potholes
        self.current_step += 1
        
        return self._get_obs(), 0, False, False, {}
    
    def _execute_repair_phase(self):
        """Execute repairs in priority order"""
        
        # Sort by DRL-assigned priority
        sorted_potholes = sorted(
            self.potholes, 
            key=lambda p: self.pothole_priorities.get(p.id, 0),
            reverse=True
        )
        
        total_reward = 0
        repair_budget = 30
        distance_traveled = 0
        
        for i, pothole in enumerate(sorted_potholes[:repair_budget]):
            # Distance from last repair
            distance = abs(pothole.segment_id - self.last_repair_segment)
            distance_traveled += distance
            
            # Reward components
            urgency_reward = pothole.urgency * 50
            early_bonus = (1 - i / repair_budget) * 20 if pothole.urgency >= 3 else 0
            distance_penalty = distance * 2
            
            repair_reward = urgency_reward + early_bonus - distance_penalty
            total_reward += repair_reward
            
            self.last_repair_segment = pothole.segment_id
            self.total_repaired += 1
            
            # Remove pothole
            segment = self.road_network.get_segment(pothole.segment_id)
            segment.remove_pothole(pothole.id)
        
        # Penalty for unrepaired critical
        remaining_critical = sum(
            1 for p in sorted_potholes[repair_budget:] 
            if p.urgency >= 3
        )
        total_reward -= remaining_critical * 30
        
        print(f"\nRepair Phase: Repaired {self.total_repaired}/{len(self.potholes)}, Reward: {total_reward:.0f}")
        
        return total_reward
    
    def _get_obs(self):
        """Observation of current pothole"""
        
        if self.current_pothole_idx >= len(self.potholes):
            return np.zeros(6, dtype=np.float32)
        
        pothole = self.potholes[self.current_pothole_idx]
        segment = self.road_network.get_segment(pothole.segment_id)
        
        obs = [
            pothole.urgency / 4.0,
            pothole.depth / 20.0,
            pothole.area / 2.0,
            segment.condition_index / 100.0,
            abs(pothole.segment_id - self.last_repair_segment) / self.num_segments,
            self.current_pothole_idx / len(self.potholes)
        ]
        
        return np.array(obs, dtype=np.float32)
