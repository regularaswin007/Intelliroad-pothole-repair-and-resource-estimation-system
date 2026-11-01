"""Train SAC Material Estimator"""
import sys
sys.path.append('./models')

from sac_material_estimator import SACMaterialEstimator

if __name__ == "__main__":
    estimator = SACMaterialEstimator(verbose=1)
    
    print("Training SAC Material Estimator...")
    estimator.train(total_timesteps=100000)
    
    estimator.save('./models/material_estimator_sac')
    print("✓ SAC material estimator saved to ./models/material_estimator_sac")
