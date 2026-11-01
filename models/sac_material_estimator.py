"""
SAC Agent for Material Usage Estimation
Predicts optimal material quantities for pothole repairs
"""

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces


class MaterialEstimationEnv(gym.Env):
    """Environment for learning optimal material usage"""
    
    def __init__(self):
        super(MaterialEstimationEnv, self).__init__()
        
        # Observation: [depth, area, severity, age]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([30, 5, 4, 365]),
            dtype=np.float32
        )
        
        # Action: material multiplier (0.8 to 1.5) - CONTINUOUS
        self.action_space = spaces.Box(
            low=np.array([0.8]),
            high=np.array([1.5]),
            dtype=np.float32
        )
        
        self.current_pothole = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random pothole characteristics
        self.current_pothole = {
            'depth': np.random.uniform(2, 25),      # cm
            'area': np.random.uniform(0.1, 4.0),    # m²
            'severity': np.random.randint(1, 5),    # 1-4
            'age': np.random.randint(0, 365)        # days
        }
        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):
        return np.array([
            self.current_pothole['depth'],
            self.current_pothole['area'],
            self.current_pothole['severity'],
            self.current_pothole['age']
        ], dtype=np.float32)
    
    def optimal_multiplier(self) -> float:
        """Rule-based target multiplier for the current pothole (the reward is measured against this)"""
        optimal_multiplier = 1.0
        
        # Deeper potholes need overfill
        if self.current_pothole['depth'] > 15:
            optimal_multiplier += 0.2
        
        # Larger areas need compaction allowance
        if self.current_pothole['area'] > 2.0:
            optimal_multiplier += 0.15
        
        # Old potholes have crumbled edges
        if self.current_pothole['age'] > 180:
            optimal_multiplier += 0.1
        
        # Critical severity needs extra material
        if self.current_pothole['severity'] == 4:
            optimal_multiplier += 0.15
        
        # Clamp optimal multiplier to valid range
        optimal_multiplier = np.clip(optimal_multiplier, 0.8, 1.5)
        return optimal_multiplier

    def step(self, action):
        multiplier = action[0]  # Extract from array
        
        # Base volume calculation
        base_volume = (self.current_pothole['depth'] / 100) * self.current_pothole['area']
        
        optimal_multiplier = self.optimal_multiplier()
        
        # Reward: closer to optimal = better
        error = abs(multiplier - optimal_multiplier)
        reward = 10.0 - (error * 20)  # Max reward 10, penalty for being off
        
        # Bonus for being within 5% of optimal
        if error < 0.05:
            reward += 5.0
        
        terminated = True
        truncated = False
        
        return self._get_obs(), float(reward), terminated, truncated, {}
    
    def render(self):
        pass


class SACMaterialEstimator:
    """SAC-based material quantity estimator"""
    
    def __init__(self, verbose=0):
        self.env = DummyVecEnv([lambda: MaterialEstimationEnv()])
        self.model = SAC(
            "MlpPolicy",
            self.env,
            learning_rate=0.001,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=128,
            tau=0.01,
            gamma=0.95,
            train_freq=4,
            gradient_steps=1,
            verbose=verbose
        )
    
    def train(self, total_timesteps=100_000):
        """Train SAC estimator"""
        self.model.learn(total_timesteps=total_timesteps)
        return self.model
    
    def save(self, path):
        """Save trained model"""
        self.model.save(path)
    
    def load(self, path):
        """Load trained model"""
        self.model = SAC.load(path, env=self.env)
    
    def estimate_material(self, pothole):
        """Estimate material needed for a pothole"""
        obs = np.array([
            pothole.depth,
            pothole.area,
            pothole.urgency,  # severity level 1-4
            pothole.age
        ], dtype=np.float32)
        
        multiplier, _ = self.model.predict(obs, deterministic=True)
        base_volume = pothole.get_volume()
        estimated_volume = base_volume * multiplier[0]  # Extract from array
        
        return estimated_volume, multiplier[0]
