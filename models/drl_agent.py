"""
DRL Agent for Pothole Repair Scheduling
Uses PPO algorithm from stable-baselines3
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch
import os

class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring"""
    
    def __init__(self, check_freq=1000, save_path='./checkpoints/', verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Save model periodically
            self.model.save(os.path.join(self.save_path, f'model_{self.n_calls}'))
            
            if self.verbose > 0:
                print(f"Saved checkpoint at step {self.n_calls}")
                
        return True


class PotholeRepairAgent:
    """Wrapper for DRL agent"""
    
    def __init__(self, env, algorithm='PPO', learning_rate=3e-4, verbose=1):
        self.env = env
        self.algorithm = algorithm
        self.verbose = verbose
        
        # Create model
        if algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                env,
                learning_rate=learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=verbose,
                tensorboard_log="./tensorboard_logs/"
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'episode_lengths': [],
            'potholes_repaired': []
        }
        
    def train(self, total_timesteps=500000, callback=None):
        """Train the agent"""
        print(f"Starting training for {total_timesteps} timesteps...")
        
        if callback is None:
            callback = TrainingCallback(check_freq=10000, verbose=self.verbose)
            
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        print("Training completed!")
        
    def evaluate(self, n_episodes=10):
        """Evaluate trained agent"""
        print(f"\nEvaluating agent for {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        total_repairs = []
        final_safety_scores = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            repairs_count = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if 'repair_executed' in info and info['repair_executed']:
                    repairs_count += info.get('potholes_fixed', 0)
                    
                done = terminated or truncated
                
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            total_repairs.append(repairs_count)
            final_safety_scores.append(info.get('network_safety', 0))
            
            if self.verbose > 0:
                print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
                      f"Repairs={repairs_count}, Safety={info.get('network_safety', 0):.1f}")
                      
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_repairs': np.mean(total_repairs),
            'mean_final_safety': np.mean(final_safety_scores)
        }
        
        print(f"\n=== Evaluation Results ===")
        print(f"Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"Mean Episode Length: {results['mean_episode_length']:.1f}")
        print(f"Mean Repairs per Episode: {results['mean_repairs']:.1f}")
        print(f"Mean Final Safety Score: {results['mean_final_safety']:.1f}")
        
        return results
        
    def predict(self, observation, deterministic=True):
        """Predict action for given observation"""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
        
    def save(self, path='./models/best_model'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
        
    def load(self, path='./models/best_model'):
        """Load trained model"""
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
