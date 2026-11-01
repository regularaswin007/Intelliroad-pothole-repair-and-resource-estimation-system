# Configuration file for DRL Pothole Repair System

# Environment Parameters
ENV_CONFIG = {
    'num_road_segments': 50,
    'max_potholes_per_segment': 10,
    'episode_length': 365,  # days in a year
    'initial_budget': 500000,  # dollars
    'repair_cost_range': (100, 2000),  # min/max repair cost
    'deterioration_rate': 0.05,  # daily deterioration factor
}

# DRL Agent Parameters
AGENT_CONFIG = {
    'algorithm': 'PPO',
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'total_timesteps': 500000,
}

# Reward Function Weights
REWARD_WEIGHTS = {
    'safety': 0.4,  # Road safety improvement
    'cost_efficiency': 0.3,  # Cost-effectiveness
    'traffic_disruption': 0.15,  # Minimize disruption
    'long_term_performance': 0.15,  # Pavement longevity
}

# Pothole Severity Levels
SEVERITY_LEVELS = {
    'minor': {'depth_range': (1, 3), 'urgency': 1},
    'moderate': {'depth_range': (3, 6), 'urgency': 2},
    'severe': {'depth_range': (6, 10), 'urgency': 3},
    'critical': {'depth_range': (10, 20), 'urgency': 4},
}

# Road Network Parameters
ROAD_CONFIG = {
    'segment_length_range': (100, 500),  # meters
    'traffic_volume_range': (1000, 50000),  # vehicles/day
    'pavement_types': ['asphalt', 'concrete', 'composite'],
    'weather_conditions': ['sunny', 'rainy', 'snowy', 'extreme'],
}

# Visualization Settings
VIZ_CONFIG = {
    'update_interval': 100,  # training steps
    'plot_metrics': ['reward', 'cost', 'safety_score', 'potholes_repaired'],
    'colors': {
        'minor': '#FFA726',
        'moderate': '#FF7043',
        'severe': '#EF5350',
        'critical': '#D32F2F',
    }
}

# Model Save Paths
MODEL_PATHS = {
    'checkpoint_dir': './checkpoints/',
    'best_model': './best_model.zip',
    'training_logs': './logs/',
}