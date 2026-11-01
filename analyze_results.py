"""
Analyze agent behavior in detail
"""

import sys
sys.path.append('./data')
sys.path.append('./models')
sys.path.append('./environment')

from repair_env import PotholeRepairEnv
from drl_agent import PotholeRepairAgent
import config
import matplotlib.pyplot as plt
import numpy as np

print("Analyzing agent behavior...\n")

# Create environment
env = PotholeRepairEnv(
    num_segments=config.ENV_CONFIG['num_road_segments'],
    episode_length=config.ENV_CONFIG['episode_length'],
    initial_budget=config.ENV_CONFIG['initial_budget']
)

# Load agent
agent = PotholeRepairAgent(env=env, verbose=0)
agent.load('./models/best_model')

# Run detailed episode
obs, _ = env.reset()
done = False

# Tracking
daily_safety = []
daily_potholes = []
daily_budget = []
daily_repairs = []
repair_actions = {'do_nothing': 0, 'patch': 0, 'overlay': 0, 'mill_fill': 0}
segment_repair_count = {i: 0 for i in range(50)}

day = 0
while not done and day < 365:
    action, _ = agent.model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Track metrics
    daily_safety.append(info['network_safety'])
    daily_potholes.append(info['total_potholes'])
    daily_budget.append(info['budget_remaining'])
    
    # Track action types
    segment_id, repair_type = action
    repair_types = ['do_nothing', 'patch', 'overlay', 'mill_fill']
    repair_actions[repair_types[repair_type]] += 1
    
    if repair_type > 0:
        segment_repair_count[segment_id] += 1
        daily_repairs.append(day)
    
    day += 1
    done = terminated or truncated

print("="*80)
print("DETAILED ANALYSIS")
print("="*80)

print(f"\n📊 Action Distribution:")
for action_type, count in repair_actions.items():
    percentage = (count / 365) * 100
    print(f"  {action_type:15s}: {count:4d} times ({percentage:5.1f}%)")

print(f"\n🔧 Repair Statistics:")
print(f"  Total repair actions: {sum(repair_actions.values()) - repair_actions['do_nothing']}")
print(f"  Do-nothing actions: {repair_actions['do_nothing']}")
print(f"  Most repaired segment: {max(segment_repair_count.values())} repairs")
print(f"  Segments never repaired: {sum(1 for v in segment_repair_count.values() if v == 0)}")

print(f"\n📈 Performance Metrics:")
print(f"  Initial safety score: {daily_safety[0]:.1f}")
print(f"  Final safety score: {daily_safety[-1]:.1f}")
print(f"  Safety improvement: {daily_safety[-1] - daily_safety[0]:+.1f}")
print(f"  Average safety: {np.mean(daily_safety):.1f}")
print(f"  Initial potholes: {daily_potholes[0]}")
print(f"  Final potholes: {daily_potholes[-1]}")
print(f"  Budget used: ${config.ENV_CONFIG['initial_budget'] - daily_budget[-1]:,.0f}")
print(f"  Budget remaining: ${daily_budget[-1]:,.0f}")

# Save plot
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(daily_safety, color='green', linewidth=2)
plt.title('Network Safety Score Over Time', fontsize=14, fontweight='bold')
plt.ylabel('Safety Score (0-100)')
plt.grid(True, alpha=0.3)
plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Target: 50')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(daily_potholes, color='red', linewidth=2)
plt.title('Total Potholes Over Time', fontsize=14, fontweight='bold')
plt.ylabel('Number of Potholes')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(daily_budget, color='blue', linewidth=2)
if daily_repairs:
    plt.scatter(daily_repairs, [daily_budget[d] for d in daily_repairs], 
               color='orange', s=20, label='Repair Actions', zorder=5)
plt.title('Budget Over Time', fontsize=14, fontweight='bold')
plt.ylabel('Budget Remaining ($)')
plt.xlabel('Day')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('agent_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Visualization saved to: agent_analysis.png")
print("="*80)
