"""
Road Network Module
Manages road segments and their properties
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import config

class RoadSegment:
    """Represents a road segment"""
    
    def __init__(self, segment_id: int, length: float, traffic_volume: int, 
                 pavement_type: str, condition_index: float = 100.0):
        self.id = segment_id
        self.length = length  # meters
        self.traffic_volume = traffic_volume  # vehicles per day
        self.pavement_type = pavement_type
        self.condition_index = condition_index  # 0-100, higher is better
        self.potholes = []  # List of pothole IDs in this segment
        
    def add_pothole(self, pothole_id: int):
        """Add pothole to segment"""
        self.potholes.append(pothole_id)
        
    def remove_pothole(self, pothole_id: int):
        """Remove pothole from segment"""
        if pothole_id in self.potholes:
            self.potholes.remove(pothole_id)
            
    def update_condition(self, repair_quality: float = 0.0):
        """Update segment condition based on potholes and repairs"""
        # Deterioration from potholes
        pothole_impact = len(self.potholes) * 2.0
        self.condition_index = max(0, self.condition_index - pothole_impact + repair_quality)
        self.condition_index = min(100, self.condition_index)
        
    def get_safety_score(self) -> float:
        """Calculate safety score based on condition and traffic"""
        # Inverse relationship: more potholes + high traffic = low safety
        if self.condition_index < 50:
            base_safety = 30 + (self.condition_index * 0.4)
        else:
            base_safety = 50 + (self.condition_index * 0.5)
            
        # Traffic factor
        traffic_penalty = min(30, (self.traffic_volume / 50000) * 30)
        
        return max(0, base_safety - traffic_penalty)
        
    def get_traffic_disruption_cost(self) -> float:
        """Estimate cost of traffic disruption during repair"""
        # Higher traffic = higher disruption cost
        base_disruption = 500  # base cost
        traffic_multiplier = self.traffic_volume / 10000
        return base_disruption * traffic_multiplier
        
    def to_dict(self) -> Dict:
        """Convert segment to dictionary"""
        return {
            'id': self.id,
            'length': self.length,
            'traffic_volume': self.traffic_volume,
            'pavement_type': self.pavement_type,
            'condition_index': self.condition_index,
            'num_potholes': len(self.potholes),
            'safety_score': self.get_safety_score()
        }


class RoadNetwork:
    """Manages the complete road network"""
    
    def __init__(self, num_segments: int, seed: int = 42):
        self.num_segments = num_segments
        np.random.seed(seed)
        self.segments = self._generate_segments()
        self.graph = self._create_network_graph()
        
    def _generate_segments(self) -> Dict[int, RoadSegment]:
        """Generate road segments with random properties"""
        segments = {}
        
        for i in range(self.num_segments):
            length = np.random.uniform(*config.ROAD_CONFIG['segment_length_range'])
            traffic = np.random.randint(*config.ROAD_CONFIG['traffic_volume_range'])
            pavement = np.random.choice(config.ROAD_CONFIG['pavement_types'])
            
            # Initial condition varies
            condition = np.random.uniform(60, 100)
            
            segments[i] = RoadSegment(i, length, traffic, pavement, condition)
            
        return segments
        
    def _create_network_graph(self) -> nx.Graph:
        """Create network graph for routing optimization"""
        G = nx.Graph()
        
        # Add nodes
        for seg_id in self.segments.keys():
            G.add_node(seg_id)
            
        # Add edges (simplified: connect sequential segments + some random connections)
        for i in range(self.num_segments - 1):
            G.add_edge(i, i + 1, weight=self.segments[i].length)
            
        # Add some random connections
        num_random_edges = self.num_segments // 5
        for _ in range(num_random_edges):
            n1, n2 = np.random.choice(self.num_segments, 2, replace=False)
            if not G.has_edge(n1, n2):
                avg_length = (self.segments[n1].length + self.segments[n2].length) / 2
                G.add_edge(n1, n2, weight=avg_length)
                
        return G
        
    def get_segment(self, segment_id: int) -> RoadSegment:
        """Get segment by ID"""
        return self.segments.get(segment_id)
        
    def get_all_segments(self) -> List[RoadSegment]:
        """Get all segments"""
        return list(self.segments.values())
        
    def get_network_statistics(self) -> Dict:
        """Calculate network-wide statistics"""
        avg_condition = np.mean([seg.condition_index for seg in self.segments.values()])
        total_potholes = sum(len(seg.potholes) for seg in self.segments.values())
        avg_safety = np.mean([seg.get_safety_score() for seg in self.segments.values()])
        
        # Segments by condition category
        poor = sum(1 for seg in self.segments.values() if seg.condition_index < 50)
        fair = sum(1 for seg in self.segments.values() if 50 <= seg.condition_index < 75)
        good = sum(1 for seg in self.segments.values() if seg.condition_index >= 75)
        
        return {
            'avg_condition': avg_condition,
            'total_potholes': total_potholes,
            'avg_safety_score': avg_safety,
            'segments_poor': poor,
            'segments_fair': fair,
            'segments_good': good
        }
        
    def find_optimal_route(self, start_segment: int, segments_to_visit: List[int]) -> List[int]:
        """Find optimal route to visit multiple segments (Traveling Salesman approximation)"""
        if not segments_to_visit:
            return []
            
        # Use greedy nearest neighbor
        route = [start_segment]
        remaining = set(segments_to_visit)
        current = start_segment
        
        while remaining:
            # Find nearest unvisited segment
            nearest = min(remaining, key=lambda s: self._distance(current, s))
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest
            
        return route
        
    def _distance(self, seg1: int, seg2: int) -> float:
        """Calculate distance between two segments"""
        try:
            return nx.shortest_path_length(self.graph, seg1, seg2, weight='weight')
        except nx.NetworkXNoPath:
            # If no path exists, return large distance
            return float('inf')