"""
Integrate SYNTHETIC report-style data into the DRL environment
(demo of initializing the environment from external pothole records - no real data is used)
"""

import sys
sys.path.append('./data')
sys.path.append('./models')
sys.path.append('./environment')

from synthetic_data import SyntheticDataGenerator, DamageClassMix, create_visualization_report
from repair_env import PotholeRepairEnv
from drl_agent import PotholeRepairAgent
from pothole_generator import Pothole
import config
import pandas as pd
import numpy as np

print("="*80)
print("SYNTHETIC DATA INTEGRATION (demo)")
print("="*80)

# Step 1: Generate synthetic data
print("\n[1/5] Generating synthetic municipal-style data...")
loader = SyntheticDataGenerator()

# Municipal-report-style synthetic data
municipal_df = loader.generate_municipal_style_data(num_records=2000, filename='municipal_style_synthetic.csv')

# Generate UK-style data
uk_df = loader.generate_uk_style_data()

# Step 2: Analyze Data
print("\n[2/5] Analyzing the synthetic data...")
analysis = loader.analyze_data(municipal_df)

print("\n📊 Data Analysis:")
print(f"  Total Records: {analysis['total_records']}")
print(f"  Date Range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")
print(f"  Severity Distribution:")
for sev, count in analysis['severity_distribution'].items():
    print(f"    {sev}: {count}")
print(f"  Average Depth: {analysis['avg_depth_cm']:.2f} cm")
print(f"  Average Area: {analysis['avg_area_sqm']:.2f} sq m")
print(f"  Total Repair Cost: ${analysis['total_repair_cost']:,.2f}")
print(f"  Segments Affected: {analysis['segments_affected']}")

# Step 3: Get Historical Trends
print("\n[3/5] Extracting historical trends...")
trends = loader.get_historical_trends(municipal_df)
print(f"  Seasonal Distribution:")
for season, count in trends['seasonal_distribution'].items():
    print(f"    {season}: {count}")

# Step 4: Create Visualization
print("\n[4/5] Creating visualization report...")
create_visualization_report(municipal_df)

# Step 5: Integrate with DRL System
print("\n[5/5] Integrating with DRL environment...")

# Convert records to simulation format
simulation_potholes = loader.convert_to_simulation_format(municipal_df.head(200))  # Use first 200

print(f"  Converted {len(simulation_potholes)} synthetic records to simulation format")

# Create environment initialized from the records
class DataInitializedPotholeEnv(PotholeRepairEnv):
    """Extended environment that can initialize from external pothole records"""
    
    def __init__(self, initial_potholes=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_pothole_records = initial_potholes or []
    
    def reset(self, seed=None, options=None):
        """Reset using the provided records if available"""
        # Call parent reset but don't use its return yet
        super().reset(seed=seed, options=options)
        
        # Override with the provided records
        if self.initial_pothole_records and len(self.initial_pothole_records) > 0:
            # Clear existing potholes
            self.potholes = []
            
            # Clear all segment potholes first
            for segment in self.road_network.get_all_segments():
                segment.potholes = []
            
            # Add potholes from the records
            for i, data in enumerate(self.initial_pothole_records[:100]):  # Limit to 100 for performance
                pothole = Pothole(
                    pothole_id=i,  # Use sequential IDs
                    segment_id=data['segment_id'],
                    severity=data['severity'],
                    depth=data['depth'],
                    area=data['area'],
                    age=data.get('age', 0)
                )
                self.potholes.append(pothole)
                
                # Update segment with correct pothole ID
                segment = self.road_network.get_segment(data['segment_id'])
                if segment:
                    segment.add_pothole(pothole.id)
            
            print(f"  Initialized with {len(self.potholes)} potholes from records")
        
        return self._get_obs(), {}

# Create environment with the records
env = DataInitializedPotholeEnv(
    initial_potholes=simulation_potholes,
    num_segments=config.ENV_CONFIG['num_road_segments'],
    episode_length=config.ENV_CONFIG['episode_length'],
    initial_budget=config.ENV_CONFIG['initial_budget']
)

print("\n✓ Environment initialized from synthetic records!")

# Test with DRL agent
print("\n📝 Testing DRL agent with synthetic-record initialization...")

# Try to load trained agent
agent = PotholeRepairAgent(env=env, verbose=0)
try:
    agent.load('./models/improved_model')
    print("✓ Loaded trained agent")
    has_model = True
except:
    try:
        agent.load('./models/best_model')
        print("✓ Loaded best_model agent")
        has_model = True
    except:
        print("! No trained model found, using random policy")
        has_model = False

# Run one episode
obs, _ = env.reset()
done = False
total_reward = 0
step = 0
repairs_made = 0

while not done and step < 365:
    if has_model:
        action, _ = agent.model.predict(obs, deterministic=True)
    else:
        action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if 'repair_executed' in info and info['repair_executed']:
        repairs_made += info.get('potholes_fixed', 0)
    
    step += 1
    done = terminated or truncated

print(f"\n📈 Simulation Results:")
print(f"  Total Steps: {step}")
print(f"  Total Reward: {total_reward:.2f}")
print(f"  Repairs Made: {repairs_made}")
print(f"  Final Safety Score: {info['network_safety']:.1f}")
print(f"  Final Budget: ${info['budget_remaining']:,.0f}")
print(f"  Final Potholes: {info['total_potholes']}")

# Save integration summary
summary = {
    'data_source': 'Synthetic (numpy random) municipal-style reports',
    'total_records': len(municipal_df),
    'analysis': analysis,
    'trends': trends,
    'simulation_result': {
        'total_reward': float(total_reward),
        'repairs_made': int(repairs_made),
        'final_safety': float(info['network_safety']),
        'budget_remaining': float(info['budget_remaining']),
        'final_potholes': int(info['total_potholes'])
    }
}

import json
import os
os.makedirs('./reports', exist_ok=True)
with open('./reports/synthetic_data_integration_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n✓ Summary saved to ./reports/synthetic_data_integration_summary.json")

# Damage-type class-mix synthetic samples
print("\n" + "="*80)
print("SYNTHETIC SAMPLES WITH AN APPROXIMATE DAMAGE-TYPE MIX")
print("="*80)

class_mix = DamageClassMix()
class_mix_meta = class_mix.write_class_mix_metadata()
mix_df = class_mix.simulate_from_class_mix(num_samples=1000)

print(f"\n✓ Generated {len(mix_df)} synthetic samples with an approximate damage-type mix")
print(f"  Pothole samples: {len(mix_df[mix_df['damage_type'] == 'pothole'])}")

# Save synthetic class-mix data
mix_df.to_csv('./synthetic_data/damage_class_mix_synthetic.csv', index=False)
print(f"✓ Synthetic class-mix data saved")

print("\n" + "="*80)
print("INTEGRATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  📄 ./synthetic_data/municipal_style_synthetic.csv")
print("  📄 ./synthetic_data/uk_style_synthetic.csv")
print("  📄 ./synthetic_data/damage_class_mix_synthetic.csv")
print("  📄 ./synthetic_data/damage_class_mix.json")
print("  📊 ./reports/synthetic_data_analysis.png")
print("  📋 ./reports/synthetic_data_integration_summary.json")
print("\nDone. All data above is synthetic; load your own CSV with SyntheticDataGenerator.load_csv_data().")
