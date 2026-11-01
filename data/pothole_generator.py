"""
Pothole Generator Module
Generates synthetic pothole data for simulation
"""

import numpy as np
from typing import List, Dict, Tuple
import config

class Pothole:
    """Represents a single pothole"""
    
    def __init__(self, pothole_id: int, segment_id: int, severity: str, depth: float, 
                 area: float, age: int = 0):
        self.id = pothole_id
        self.segment_id = segment_id
        self.severity = severity
        self.depth = depth  # cm
        self.area = area  # square meters
        self.age = age  # days since formation
        self.urgency = config.SEVERITY_LEVELS[severity]['urgency']
        
    def deteriorate(self, weather_factor: float = 1.0):
        """Simulate pothole deterioration over time"""
        deterioration = config.ENV_CONFIG['deterioration_rate'] * weather_factor
        self.depth += deterioration * (1 + self.urgency * 0.1)
        self.area += deterioration * 0.5
        self.age += 1
        self._update_severity()
        
    def _update_severity(self):
        """Update severity based on current depth"""
        for severity, params in config.SEVERITY_LEVELS.items():
            min_depth, max_depth = params['depth_range']
            if min_depth <= self.depth < max_depth:
                self.severity = severity
                self.urgency = params['urgency']
                break
        else:
            self.severity = 'critical'
            self.urgency = 4
            
    def get_volume(self) -> float:
        """Calculate pothole volume in cubic meters"""
        return self.area * (self.depth / 100)  # convert cm to m
        
    def get_repair_cost(self) -> float:
        """Estimate repair cost based on volume and severity"""
        base_cost = np.random.uniform(*config.ENV_CONFIG['repair_cost_range'])
        severity_multiplier = {
            'minor': 0.5,
            'moderate': 0.75,
            'severe': 1.0,
            'critical': 1.5
        }
        return base_cost * severity_multiplier[self.severity] * self.get_volume()
        
    def to_dict(self) -> Dict:
        """Convert pothole to dictionary"""
        return {
            'id': self.id,
            'segment_id': self.segment_id,
            'severity': self.severity,
            'depth': self.depth,
            'area': self.area,
            'age': self.age,
            'urgency': self.urgency,
            'volume': self.get_volume(),
            'repair_cost': self.get_repair_cost()
        }


class PotholeGenerator:
    """Generates and manages potholes for simulation"""
    
    def __init__(self, num_segments: int, seed: int = 42):
        self.num_segments = num_segments
        self.pothole_counter = 0
        np.random.seed(seed)
        
    def generate_initial_potholes(self, density: float = 0.3) -> List[Pothole]:
        """Generate initial set of potholes across road network"""
        potholes = []
        max_per_segment = config.ENV_CONFIG['max_potholes_per_segment']
        
        for segment_id in range(self.num_segments):
            # Number of potholes in this segment
            num_potholes = np.random.binomial(max_per_segment, density)
            
            for _ in range(num_potholes):
                pothole = self._create_random_pothole(segment_id)
                potholes.append(pothole)
                
        return potholes
        
    def _create_random_pothole(self, segment_id: int) -> Pothole:
        """Create a random pothole"""
        # Random severity
        severities = list(config.SEVERITY_LEVELS.keys())
        severity = np.random.choice(severities, p=[0.4, 0.3, 0.2, 0.1])
        
        # Depth based on severity
        depth_range = config.SEVERITY_LEVELS[severity]['depth_range']
        depth = np.random.uniform(*depth_range)
        
        # Random area (0.1 to 2.0 square meters)
        area = np.random.uniform(0.1, 2.0)
        
        # Random age (0 to 180 days)
        age = np.random.randint(0, 180)
        
        pothole = Pothole(
            pothole_id=self.pothole_counter,
            segment_id=segment_id,
            severity=severity,
            depth=depth,
            area=area,
            age=age
        )
        
        self.pothole_counter += 1
        return pothole
        
    def spawn_new_potholes(self, weather: str, day: int) -> List[Pothole]:
        """Spawn new potholes based on weather conditions"""
        weather_spawn_rate = {
            'sunny': 0.01,
            'rainy': 0.05,
            'snowy': 0.03,
            'extreme': 0.08
        }
        
        spawn_rate = weather_spawn_rate.get(weather, 0.01)
        new_potholes = []
        
        for segment_id in range(self.num_segments):
            if np.random.random() < spawn_rate:
                pothole = self._create_random_pothole(segment_id)
                new_potholes.append(pothole)
                
        return new_potholes
        
    def get_pothole_statistics(self, potholes: List[Pothole]) -> Dict:
        """Calculate statistics for current potholes"""
        if not potholes:
            return {
                'total': 0,
                'by_severity': {},
                'avg_depth': 0,
                'total_volume': 0,
                'total_repair_cost': 0
            }
            
        severity_counts = {}
        for severity in config.SEVERITY_LEVELS.keys():
            severity_counts[severity] = sum(1 for p in potholes if p.severity == severity)
            
        return {
            'total': len(potholes),
            'by_severity': severity_counts,
            'avg_depth': np.mean([p.depth for p in potholes]),
            'total_volume': sum(p.get_volume() for p in potholes),
            'total_repair_cost': sum(p.get_repair_cost() for p in potholes)
        }