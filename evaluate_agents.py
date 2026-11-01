"""
Evaluate trained agents against simple baselines
Writes ./reports/evaluation_results.json (read by the Analytics page of complete_pipeline.py)

  Priority:  PPO agent vs urgency-sort vs random    (PriorityRepairEnv)
  Material:  SAC estimator vs fixed 1.0 multiplier  (MaterialEstimationEnv)

All numbers come from the synthetic simulator, not from real roads.
Every policy sees the SAME episodes (seeded), so the comparison is paired.

Usage:
  python evaluate_agents.py
  python evaluate_agents.py --episodes 100 --material-samples 2000
"""

import sys
sys.path.append('./environment')
sys.path.append('./models')
sys.path.append('./data')

import os
import io
import json
import argparse
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np

from priority_repair_env import PriorityRepairEnv

REPAIR_BUDGET = 30  # potholes repaired per episode - matches PriorityRepairEnv._execute_repair_phase


# ========================================
# PRIORITY AGENT
# ========================================
def random_policy(env, obs):
    return env.action_space.sample()


def urgency_policy(env, obs):
    """Baseline: score each pothole by its urgency level (repairs then run most urgent first)"""
    pothole = env.potholes[env.current_pothole_idx]
    return np.array([pothole.urgency / 4.0], dtype=np.float32)


def make_ppo_policy(env, model_path):
    from drl_agent import PotholeRepairAgent
    agent = PotholeRepairAgent(env=env, verbose=0)
    agent.load(model_path)
    return lambda env_, obs: agent.model.predict(obs, deterministic=True)[0]


def run_priority_episode(env, policy, seed):
    """One episode; returns (reward, critical fixed, critical total, travel distance)"""
    np.random.seed(seed)          # pothole generation uses the global RNG
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)

    done, reward = False, 0.0
    with redirect_stdout(io.StringIO()):   # the env prints one line per episode
        while not done:
            action = policy(env, obs)
            obs, reward, done, _, _ = env.step(action)

    # Same ordering the env used for its repair phase
    ranked = sorted(env.potholes, key=lambda p: env.pothole_priorities.get(p.id, 0), reverse=True)
    chosen = ranked[:REPAIR_BUDGET]
    critical_total = sum(1 for p in env.potholes if p.urgency >= 3)
    critical_fixed = sum(1 for p in chosen if p.urgency >= 3)

    distance, last = 0, 0
    for p in chosen:
        distance += abs(p.segment_id - last)
        last = p.segment_id
    return reward, critical_fixed, critical_total, distance


def evaluate_priority(n_episodes, model_path):
    env = PriorityRepairEnv(num_segments=20, max_potholes=50, episode_length=100)
    policies = {'random': random_policy, 'urgency_sort': urgency_policy}

    if os.path.exists(model_path + '.zip'):
        try:
            policies['ppo_agent'] = make_ppo_policy(env, model_path)
        except Exception as e:
            print(f"! Could not load PPO agent ({e}) - evaluating baselines only")
    else:
        print(f"! {model_path}.zip not found - evaluating baselines only")

    results = {}
    for name, policy in policies.items():
        rows = np.array([run_priority_episode(env, policy, seed=1000 + i) for i in range(n_episodes)], dtype=float)
        results[name] = {
            'mean_reward': float(rows[:, 0].mean()),
            'std_reward': float(rows[:, 0].std()),
            'mean_critical_fixed': float(rows[:, 1].mean()),
            'mean_critical_total': float(rows[:, 2].mean()),
            'mean_travel_distance': float(rows[:, 3].mean()),
        }
    return {'repair_budget': REPAIR_BUDGET, 'policies': results}


# ========================================
# MATERIAL ESTIMATOR
# ========================================
def evaluate_material(n_samples, model_path):
    try:
        from sac_material_estimator import MaterialEstimationEnv, SACMaterialEstimator
    except ImportError as e:
        print(f"! Skipping material evaluation ({e})")
        return None

    env = MaterialEstimationEnv()
    estimator = None
    if os.path.exists(model_path + '.zip'):
        try:
            estimator = SACMaterialEstimator(verbose=0)
            estimator.load(model_path)
        except Exception as e:
            print(f"! Could not load SAC estimator ({e}) - evaluating the baseline only")
            estimator = None
    else:
        print(f"! {model_path}.zip not found - evaluating the baseline only")

    errors = {'fixed_1.0': []}
    if estimator is not None:
        errors['sac_agent'] = []

    for i in range(n_samples):
        np.random.seed(20000 + i)
        obs, _ = env.reset()
        target = float(env.optimal_multiplier())
        errors['fixed_1.0'].append(abs(1.0 - target))
        if estimator is not None:
            multiplier, _ = estimator.model.predict(obs, deterministic=True)
            errors['sac_agent'].append(abs(float(np.asarray(multiplier).ravel()[0]) - target))

    policies = {
        name: {'mae': float(np.mean(e)), 'within_0_05': float(np.mean(np.array(e) < 0.05))}
        for name, e in errors.items()
    }
    return {
        'n_samples': n_samples,
        'policies': policies,
        'note': ("The target is the rule-based multiplier defined in MaterialEstimationEnv, so this measures how "
                 "well the SAC agent reproduces that rule on synthetic potholes, not real material usage."),
    }


# ========================================
# MAIN
# ========================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate trained agents against baselines')
    parser.add_argument('--episodes', type=int, default=50, help='Priority-agent episodes per policy')
    parser.add_argument('--material-samples', type=int, default=1000, help='Potholes for the material estimator')
    parser.add_argument('--priority-model', type=str, default='./models/priority_drl')
    parser.add_argument('--material-model', type=str, default='./models/material_estimator_sac')
    parser.add_argument('--out', type=str, default='./reports/evaluation_results.json')
    args = parser.parse_args()

    print("=" * 80)
    print("AGENT EVALUATION (synthetic simulator)")
    print("=" * 80)

    print(f"\n[1/2] Priority agent ({args.episodes} episodes per policy)...")
    priority = evaluate_priority(args.episodes, args.priority_model)
    for name, m in priority['policies'].items():
        print(f"  {name:14s} reward {m['mean_reward']:8.1f} +/- {m['std_reward']:6.1f} | "
              f"critical fixed {m['mean_critical_fixed']:.1f}/{m['mean_critical_total']:.1f} | "
              f"distance {m['mean_travel_distance']:.1f}")

    print(f"\n[2/2] Material estimator ({args.material_samples} potholes)...")
    material = evaluate_material(args.material_samples, args.material_model)
    if material:
        for name, m in material['policies'].items():
            print(f"  {name:14s} MAE {m['mae']:.3f} | within 0.05 of target: {m['within_0_05']:.0%}")

    results = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'n_episodes': args.episodes,
        'priority': priority,
        'material': material,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {args.out}")


if __name__ == '__main__':
    main()
