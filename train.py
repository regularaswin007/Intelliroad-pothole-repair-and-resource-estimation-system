"""
Training script for TD3 agent
Member 2 Contribution
"""

import sys
sys.path.append('..')

import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import argparse

from environment.pothole_env import PotholeEnv
from agent.td3_agent import TD3Agent
from agent.replay_buffer import ReplayBuffer
from config import TD3_CONFIG, TRAINING_CONFIG, MODELS_DIR, LOGS_DIR

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--save-freq', type=int, default=50)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment
    env = PotholeEnv(render_mode="human" if args.render else None)
    
    # Create agent
    agent = TD3Agent(
        state_dim=TD3_CONFIG['state_dim'],
        action_dim=TD3_CONFIG['action_dim'],
        max_action=TD3_CONFIG['max_action'],
        config=TD3_CONFIG
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        state_dim=TD3_CONFIG['state_dim'],
        action_dim=TD3_CONFIG['action_dim'],
        max_size=TD3_CONFIG['buffer_size']
    )
    
    # TensorBoard writer
    writer = SummaryWriter(LOGS_DIR / 'td3_training')
    
    # Training loop
    total_timesteps = 0
    episode_rewards = []
    
    for episode in range(args.episodes):
        state, info = env.reset(seed=args.seed + episode)
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Select action with exploration noise
            if total_timesteps < TD3_CONFIG['start_timesteps']:
                action = env.action_space.sample()
            else:
                action = agent.select_action(
                    state,
                    explore=True,
                    noise=TD3_CONFIG['exploration_noise']
                )
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            replay_buffer.add(state, action, next_state, reward, float(done))
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_timesteps += 1
            
            # Train agent
            if total_timesteps >= TD3_CONFIG['start_timesteps']:
                train_info = agent.train(replay_buffer, TD3_CONFIG['batch_size'])
                
                if total_timesteps % 100 == 0:
                    writer.add_scalar('Loss/Critic', train_info['critic_loss'], total_timesteps)
                    writer.add_scalar('Loss/Actor', train_info['actor_loss'], total_timesteps)
            
            if args.render:
                env.render()
        
        episode_rewards.append(episode_reward)
        
        # Logging
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        writer.add_scalar('Info/Potholes_Detected', info['potholes_detected'], episode)
        writer.add_scalar('Info/Potholes_Repaired', info['potholes_repaired'], episode)
        writer.add_scalar('Info/Success_Rate', info['success_rate'], episode)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | "
                  f"Repaired: {info['potholes_repaired']}/{info['total_potholes']} | "
                  f"Steps: {episode_steps}")
        
        # Save model
        if episode % args.save_freq == 0 and episode > 0:
            agent.save(MODELS_DIR / f'td3_episode_{episode}.pth')
    
    # Save final model
    agent.save(MODELS_DIR / 'td3_final.pth')
    writer.close()
    env.close()
    
    print("Training completed!")

if __name__ == "__main__":
    train()
