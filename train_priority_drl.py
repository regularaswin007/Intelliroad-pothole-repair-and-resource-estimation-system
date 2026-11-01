"""
Train DRL Priority Agent - Learns optimal repair ordering
"""

import sys
sys.path.append('./environment')
sys.path.append('./models')
sys.path.append('./data')

from priority_repair_env import PriorityRepairEnv
from drl_agent import PotholeRepairAgent

print("="*80)
print("DRL PRIORITY AGENT - Learning Optimal Repair Ordering")
print("="*80)

# Create environment
env = PriorityRepairEnv(num_segments=20, max_potholes=50, episode_length=100)

print("✓ Priority environment created")
print("  DRL learns to score potholes")
print("  Repairs execute in learned priority order\n")

# Test environment
obs, _ = env.reset()
print(f"Test: {len(env.potholes)} potholes to prioritize")

# Score all potholes (test)
episode_reward = 0
done = False
while not done:
    action = [0.5]  # Test priority score
    obs, reward, done, _, info = env.step(action)
    episode_reward += reward

print(f"Test episode: Reward={episode_reward:.0f}, Repaired={info.get('total_repaired', 0)}")

if episode_reward > 0:
    print("✅ Reward function works!\n")
else:
    print("⚠️ Testing...\n")

# Reset for training
env.reset()

# Create DRL agent
agent = PotholeRepairAgent(
    env=env,
    algorithm='PPO',
    learning_rate=3e-4,
    verbose=1
)

print("Training DRL prioritization policy for 100k steps (~10 minutes)...\n")

agent.train(total_timesteps=100000)

agent.save('./models/priority_drl')
print("\n✓ DRL model saved!")

# Evaluate
print("\nEvaluating DRL prioritization...")
results = agent.evaluate(n_episodes=10)

print("\n" + "="*80)
print("DRL PRIORITY AGENT RESULTS")
print("="*80)
print(f"Mean Reward: {results['mean_reward']:.2f}")
print(f"Mean Repairs: {results['mean_repairs']:.1f}")

print("\nCompare against baselines with: python evaluate_agents.py")
print("="*80)
