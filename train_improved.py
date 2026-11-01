"""
Improved training script with better hyperparameters
"""

import sys
sys.path.append('./data')
sys.path.append('./models')
sys.path.append('./environment')

from repair_env import PotholeRepairEnv
from drl_agent import PotholeRepairAgent, TrainingCallback
import config

# Override config for better learning
config.ENV_CONFIG['initial_budget'] = 1000000  # More budget for learning
config.ENV_CONFIG['deterioration_rate'] = 0.03  # Slower deterioration

# Better reward weights - prioritize safety more
config.REWARD_WEIGHTS = {
    'safety': 0.6,  # Increased from 0.4
    'cost_efficiency': 0.2,  # Decreased from 0.3
    'traffic_disruption': 0.1,
    'long_term_performance': 0.1,
}

print("="*80)
print("IMPROVED DRL TRAINING")
print("="*80)

# Create environment
print("\n[1/3] Creating environment with improved config...")
env = PotholeRepairEnv(
    num_segments=config.ENV_CONFIG['num_road_segments'],
    episode_length=config.ENV_CONFIG['episode_length'],
    initial_budget=config.ENV_CONFIG['initial_budget']
)
print(f"✓ Budget: ${env.initial_budget:,}, Deterioration: {config.ENV_CONFIG['deterioration_rate']}")

# Create agent with better learning rate
print("\n[2/3] Creating agent with improved hyperparameters...")
agent = PotholeRepairAgent(
    env=env,
    algorithm='PPO',
    learning_rate=5e-4,  # Increased learning rate
    verbose=1
)

# Train longer
print("\n[3/3] Training for 300k timesteps (will take ~20 minutes)...")
callback = TrainingCallback(check_freq=10000, save_path='./checkpoints/')
agent.train(total_timesteps=300000, callback=callback)

# Save
agent.save('./models/improved_model')
print(f"\n✓ Improved model saved!")

# Quick evaluation
print("\nQuick evaluation...")
results = agent.evaluate(n_episodes=5)
