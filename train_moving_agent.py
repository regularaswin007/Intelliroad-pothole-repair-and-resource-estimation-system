"""
Train the moving repair agent (MovingRepairAgentEnv) with PPO
"""

import sys
import os

sys.path.append('./environment')
sys.path.append('./models')
sys.path.append('./data')

from moving_agent_env import MovingRepairAgentEnv
from drl_agent import PotholeRepairAgent, TrainingCallback

print("="*80)
print("TRAINING MOVING REPAIR AGENT")
print("="*80)

# Small environment
env = MovingRepairAgentEnv(
    num_segments=10,
    episode_length=100,
    speed=10.0
)

print("✓ Environment created\n")

# Test environment works
obs, _ = env.reset()
print(f"Initial potholes: {len(env.potholes)}")

# Test a repair action
env.agent_position = 0
# Find a segment with potholes
for seg_id in range(10):
    if any(p.segment_id == seg_id for p in env.potholes):
        env.agent_position = seg_id
        break

action = [env.agent_position, 1]  # Stay and repair
obs, reward, _, _, info = env.step(action)
print(f"Test repair: Reward={reward:.1f}, Repaired={info.get('repaired', 0)}")

if reward > 0:
    print("✓ Repair step gives a positive reward\n")
else:
    print("✗ Repair step did not give a positive reward - check the environment")
    sys.exit(1)

# Reset for training
env.reset()

# Create agent
agent = PotholeRepairAgent(
    env=env,
    algorithm='PPO',
    learning_rate=3e-4,
    verbose=1
)

print("Training for 30k timesteps...\n")

agent.train(total_timesteps=30000)

agent.save('./models/moving_agent')
print("\n✓ Model saved!")

# Evaluate
print("\nEvaluating...")
results = agent.evaluate(n_episodes=5)

print("\n" + "="*80)
print(f"Mean Reward: {results['mean_reward']:.2f}")
print(f"Mean Repairs: {results['mean_repairs']:.1f}")

if results['mean_repairs'] > 10:
    print("\nMean repairs per episode above 10")
elif results['mean_repairs'] > 0:
    print("\nSome repairs made - consider training longer")
else:
    print("\nNo repairs made - check the reward function")
print("="*80)
