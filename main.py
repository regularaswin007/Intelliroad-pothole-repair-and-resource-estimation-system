"""
Main script to run DRL Pothole Repair System
"""

import sys
import os
import argparse
import numpy as np

# Add paths
sys.path.append('./data')
sys.path.append('./models')
sys.path.append('./environment')

from repair_env import PotholeRepairEnv
from drl_agent import PotholeRepairAgent, TrainingCallback
import config

def train_agent(timesteps=500000, save_path='./models/best_model'):
    """Train DRL agent"""
    print("="*80)
    print("DRL-BASED POTHOLE REPAIR SCHEDULING SYSTEM")
    print("="*80)
    
    # Create environment
    print("\n[1/4] Creating environment...")
    env = PotholeRepairEnv(
        num_segments=config.ENV_CONFIG['num_road_segments'],
        episode_length=config.ENV_CONFIG['episode_length'],
        initial_budget=config.ENV_CONFIG['initial_budget']
    )
    print(f"✓ Environment created: {env.num_segments} segments, {env.episode_length} days")
    
    # Create agent
    print("\n[2/4] Initializing DRL agent...")
    agent = PotholeRepairAgent(
        env=env,
        algorithm=config.AGENT_CONFIG['algorithm'],
        learning_rate=config.AGENT_CONFIG['learning_rate'],
        verbose=1
    )
    print(f"✓ Agent initialized: {config.AGENT_CONFIG['algorithm']} algorithm")
    
    # Train agent
    print(f"\n[3/4] Training agent for {timesteps} timesteps...")
    print("This may take 10-15 minutes depending on your hardware...\n")
    
    callback = TrainingCallback(check_freq=10000, save_path='./checkpoints/')
    agent.train(total_timesteps=timesteps, callback=callback)
    
    # Save trained model
    print("\n[4/4] Saving trained model...")
    agent.save(save_path)
    print(f"✓ Model saved to {save_path}.zip")
    
    return agent, env

def evaluate_agent(model_path='./models/best_model', n_episodes=10):
    """Evaluate trained agent"""
    print("\n" + "="*80)
    print("EVALUATING TRAINED AGENT")
    print("="*80)
    
    # Create environment
    env = PotholeRepairEnv(
        num_segments=config.ENV_CONFIG['num_road_segments'],
        episode_length=config.ENV_CONFIG['episode_length'],
        initial_budget=config.ENV_CONFIG['initial_budget']
    )
    
    # Create and load agent
    agent = PotholeRepairAgent(env=env, verbose=1)
    agent.load(model_path)
    
    # Evaluate
    results = agent.evaluate(n_episodes=n_episodes)
    
    return results

def demo_run(model_path='./models/best_model'):
    """Run demo with visualization"""
    print("\n" + "="*80)
    print("DEMO RUN WITH VISUALIZATION")
    print("="*80)
    
    # Create environment
    env = PotholeRepairEnv(
        num_segments=config.ENV_CONFIG['num_road_segments'],
        episode_length=config.ENV_CONFIG['episode_length'],
        initial_budget=config.ENV_CONFIG['initial_budget']
    )
    
    # Load agent
    agent = PotholeRepairAgent(env=env, verbose=1)
    
    try:
        agent.load(model_path)
        print("✓ Loaded trained model")
    except:
        print("! No trained model found, using random policy")
        
    # Run episode with rendering
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    print("\nRunning simulation...")
    print("(Progress updates every 30 days)\n")
    
    while not done and step < 365:
        if os.path.exists(model_path + '.zip'):
            action, _ = agent.model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Render every 30 days
        if step % 30 == 0:
            env.render()
            
        done = terminated or truncated
        
    print(f"\n{'='*80}")
    print(f"Simulation Complete!")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Budget: ${info['budget_remaining']:,.0f}")
    print(f"Final Network Safety: {info['network_safety']:.1f}/100")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description='DRL Pothole Repair System')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate', 'demo'],
                       help='Mode: train, evaluate, or demo')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Training timesteps (default: 100000)')
    parser.add_argument('--model', type=str, default='./models/best_model',
                       help='Model path')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Evaluation episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        agent, env = train_agent(timesteps=args.timesteps, save_path=args.model)
        print("\n✓ Training complete! Run with --mode evaluate to test performance")
        
    elif args.mode == 'evaluate':
        results = evaluate_agent(model_path=args.model, n_episodes=args.episodes)
        
    elif args.mode == 'demo':
        demo_run(model_path=args.model)

if __name__ == '__main__':
    main()
