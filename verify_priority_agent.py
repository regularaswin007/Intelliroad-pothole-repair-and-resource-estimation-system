"""
Verify the DRL priority agent is working correctly
"""

import sys
sys.path.append('./environment')
sys.path.append('./models')
sys.path.append('./data')

from priority_repair_env import PriorityRepairEnv
from drl_agent import PotholeRepairAgent
import numpy as np

print("="*80)
print("VERIFYING DRL PRIORITY AGENT")
print("="*80)

# Load trained model
env = PriorityRepairEnv(num_segments=20, max_potholes=50)
agent = PotholeRepairAgent(env=env, verbose=0)
agent.load('./models/priority_drl')

print("✓ Loaded trained DRL model\n")

# Run detailed episode
obs, _ = env.reset()
print(f"Episode started: {len(env.potholes)} potholes to prioritize\n")

# Collect priorities assigned by DRL
drl_priorities = []
done = False
step = 0

while not done:
    action, _ = agent.model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    
    if not done:
        pothole = env.potholes[step]
        priority = float(action[0])
        drl_priorities.append({
            'id': pothole.id,
            'segment': pothole.segment_id,
            'urgency': pothole.urgency,
            'depth': pothole.depth,
            'drl_priority': priority
        })
        step += 1

print("\n" + "="*80)
print("DRL PRIORITY ASSIGNMENTS (First 10)")
print("="*80)
print(f"{'Segment':<10} {'Urgency':<10} {'Depth':<10} {'DRL Priority':<15}")
print("-" * 50)

for p in sorted(drl_priorities, key=lambda x: x['drl_priority'], reverse=True)[:10]:
    print(f"{p['segment']:<10} {p['urgency']:<10.1f} {p['depth']:<10.1f} {p['drl_priority']:<15.3f}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Analyze what DRL learned
high_priority = [p for p in drl_priorities if p['drl_priority'] > 0.7]
medium_priority = [p for p in drl_priorities if 0.4 <= p['drl_priority'] <= 0.7]
low_priority = [p for p in drl_priorities if p['drl_priority'] < 0.4]

print(f"Priority Distribution:")
print(f"  High (>0.7): {len(high_priority)}")
print(f"  Medium (0.4-0.7): {len(medium_priority)}")
print(f"  Low (<0.4): {len(low_priority)}")

if high_priority:
    avg_high_urgency = np.mean([p['urgency'] for p in high_priority])
    print(f"\nHigh priority potholes: Avg urgency = {avg_high_urgency:.2f}")
else:
    avg_high_urgency = 0
    print(f"\nNo high priority assignments!")

if low_priority:
    avg_low_urgency = np.mean([p['urgency'] for p in low_priority])
    print(f"Low priority potholes: Avg urgency = {avg_low_urgency:.2f}")
else:
    avg_low_urgency = 0

# Check urgency correlation
urgencies = [p['urgency'] for p in drl_priorities]
priorities = [p['drl_priority'] for p in drl_priorities]
correlation = np.corrcoef(urgencies, priorities)[0, 1]

print(f"\nCorrelation between urgency and DRL priority: {correlation:.3f}")

# Final stats
print(f"\nTotal Repaired: {info['total_repaired']}")
print(f"Total Potholes: {info['total_potholes']}")
print(f"Efficiency: {info['repair_efficiency']*100:.1f}%")
print(f"Final Reward: {reward:.0f}")

print("\n" + "="*80)
if correlation > 0.3:
    print("Priorities correlate positively with urgency (r > 0.3)")
elif correlation > 0:
    print("Weak positive correlation between priorities and urgency")
else:
    print("No positive correlation between priorities and urgency")
print("="*80)
