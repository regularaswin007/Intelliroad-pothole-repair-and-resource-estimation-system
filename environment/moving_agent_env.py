"""
Moving Repair Agent Environment
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pothole_generator import PotholeGenerator, Pothole
from road_network import RoadNetwork
import config

class MovingRepairAgentEnv(gym.Env):
    """Agent moves between road segments and repairs potholes (simplified reward shaping)"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, num_segments=15, episode_length=200, speed=5.0):
        super().__init__()
        
        self.num_segments = num_segments
        self.episode_length = episode_length
        self.agent_speed = speed
        
        # Components
        self.road_network = RoadNetwork(num_segments)
        self.pothole_gen = PotholeGenerator(num_segments)
        
        # Agent state (fuel and capacity are effectively unlimited in this simplified version)
        self.agent_position = 0
        self.agent_fuel = 100000.0  # effectively unlimited
        self.repair_capacity = 1000  # effectively unlimited
        
        # Action space: [segment_to_move_to, should_repair]
        self.action_space = spaces.MultiDiscrete([num_segments, 2])
        
        # Observation space
        obs_dim = num_segments + 3  # potholes per segment + position + fuel + capacity
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # State
        self.current_step = 0
        self.potholes = []
        self.total_repaired = 0
        self.total_distance = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset
        self.agent_position = 0
        self.agent_fuel = 100000.0
        self.repair_capacity = 1000
        self.current_step = 0
        self.total_repaired = 0
        self.total_distance = 0
        
        # Generate potholes
        self.potholes = self.pothole_gen.generate_initial_potholes(density=0.5)
        
        # Update segments
        for segment in self.road_network.get_all_segments():
            segment.potholes = []
        for pothole in self.potholes:
            self.road_network.get_segment(pothole.segment_id).add_pothole(pothole.id)
        
        return self._get_obs(), {}
    
    def step(self, action):
        target_segment, should_repair = action
        target_segment = int(target_segment)
        should_repair = int(should_repair)
        
        reward = 0
        info = {}
        
        # 1. MOVEMENT (free: no travel penalty)
        if target_segment != self.agent_position:
            self.agent_position = target_segment
            info['moved'] = True
        
        # 2. REPAIR (large positive reward per pothole)
        segment_potholes = [p for p in self.potholes if p.segment_id == self.agent_position]
        
        if should_repair == 1 and len(segment_potholes) > 0:
            # Repair all potholes at this location
            repairs_made = len(segment_potholes)
            
            for pothole in segment_potholes:
                # Reward scales with urgency
                reward += 50 * pothole.urgency  # 50-200 points per pothole
                
                self.potholes.remove(pothole)
                self.road_network.get_segment(self.agent_position).remove_pothole(pothole.id)
                self.total_repaired += 1
            
            info['repaired'] = repairs_made
            print(f"Step {self.current_step}: Repaired {repairs_made} potholes at segment {self.agent_position}, Reward: +{reward:.0f}")
        
        elif should_repair == 1 and len(segment_potholes) == 0:
            # Small penalty for trying to repair nothing
            reward -= 1
            info['repaired'] = 0
        
        # 3. TIME PROGRESSION
        self.current_step += 1
        
        # Spawn new potholes occasionally
        if self.current_step % 50 == 0 and len(self.potholes) < 30:
            new_potholes = self.pothole_gen.spawn_new_potholes('sunny', self.current_step)
            for p in new_potholes[:2]:  # Max 2 at a time
                self.potholes.append(p)
                self.road_network.get_segment(p.segment_id).add_pothole(p.id)
        
        # 4. Penalty for remaining potholes
        reward -= len(self.potholes) * 0.1
        
        # Termination
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        info['agent_position'] = self.agent_position
        info['total_repaired'] = self.total_repaired
        info['total_potholes'] = len(self.potholes)
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        """Observation: agent position, fuel, capacity and potholes per segment"""
        obs = []
        
        # Agent position (normalized)
        obs.append(self.agent_position / self.num_segments)
        
        # Fuel (always 1.0 - unlimited)
        obs.append(1.0)
        
        # Capacity (always 1.0 - unlimited)
        obs.append(1.0)
        
        # Potholes per segment
        for seg_id in range(self.num_segments):
            segment_potholes = [p for p in self.potholes if p.segment_id == seg_id]
            obs.append(min(1.0, len(segment_potholes) / 5.0))
        
        return np.array(obs, dtype=np.float32)
    
    def render(self):
        if self.current_step % 50 == 0:
            print(f"Step {self.current_step}: Position={self.agent_position}, Repaired={self.total_repaired}, Potholes={len(self.potholes)}")
