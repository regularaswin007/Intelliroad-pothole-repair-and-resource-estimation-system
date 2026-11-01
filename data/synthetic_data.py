"""
Synthetic Data Generator
Generates SYNTHETIC pothole records (numpy random draws) that mimic the schema of municipal,
UK-council and damage-class-mix data, for testing the pipeline. NOTHING generated here is real data.
Real CSV files can be loaded with load_csv_data().
"""

import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    """Generate and process SYNTHETIC pothole report data (plus CSV loading for real files you provide)"""
    
    def __init__(self, data_dir='./synthetic_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_csv_data(self, filepath: str) -> pd.DataFrame:
        """
        Load pothole data from a CSV file (e.g. real data you provide)
        Expected columns: date, latitude, longitude, severity, depth, area, status, cost
        """
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            return None
    
    def generate_municipal_style_data(self, num_records=1000, filename='municipal_style_synthetic.csv'):
        """
        Generate SYNTHETIC records that mimic the columns of municipal pothole reports.
        All values are random draws (seeded) - this is not real data.
        """
        np.random.seed(42)
        
        # Generate dates (last year)
        start_date = datetime.now() - timedelta(days=365)
        dates = [start_date + timedelta(days=np.random.randint(0, 365)) 
                for _ in range(num_records)]
        
        # Generate locations (sample city coordinates)
        base_lat, base_lon = 40.7128, -74.0060  # NYC coordinates as example
        latitudes = base_lat + np.random.normal(0, 0.1, num_records)
        longitudes = base_lon + np.random.normal(0, 0.1, num_records)
        
        # Severity levels
        severities = np.random.choice(
            ['minor', 'moderate', 'severe', 'critical'],
            num_records,
            p=[0.4, 0.3, 0.2, 0.1]
        )
        
        # Depth (cm) based on severity
        depth_ranges = {
            'minor': (1, 3),
            'moderate': (3, 6),
            'severe': (6, 10),
            'critical': (10, 20)
        }
        depths = [np.random.uniform(*depth_ranges[sev]) for sev in severities]
        
        # Area (square meters)
        areas = np.random.uniform(0.1, 2.5, num_records)
        
        # Status (some repaired, some pending)
        statuses = np.random.choice(
            ['reported', 'inspected', 'scheduled', 'repaired', 'verified'],
            num_records,
            p=[0.3, 0.2, 0.2, 0.2, 0.1]
        )
        
        # Repair costs (only for repaired ones)
        base_costs = areas * depths * 50  # Base calculation
        costs = [
            cost if status == 'repaired' else None 
            for cost, status in zip(base_costs, statuses)
        ]
        
        # Street names
        street_names = [
            'Main St', 'Oak Ave', 'Elm St', 'Maple Dr', 'Washington Blvd',
            'Lincoln Ave', 'Park Rd', 'Broadway', 'Cedar Ln', 'Pine St'
        ]
        streets = np.random.choice(street_names, num_records)
        
        # Road segments (0-49 to match our simulation)
        segments = np.random.randint(0, 50, num_records)
        
        # Traffic volume for that segment
        traffic_volumes = np.random.randint(1000, 50000, num_records)
        
        # Weather at time of report
        weather_conditions = np.random.choice(
            ['sunny', 'rainy', 'snowy', 'cloudy'],
            num_records,
            p=[0.5, 0.3, 0.1, 0.1]
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'report_date': dates,
            'latitude': latitudes,
            'longitude': longitudes,
            'street_name': streets,
            'segment_id': segments,
            'severity': severities,
            'depth_cm': depths,
            'area_sqm': areas,
            'volume_cum': areas * (np.array(depths) / 100),
            'status': statuses,
            'repair_cost': costs,
            'traffic_volume': traffic_volumes,
            'weather': weather_conditions
        })
        
        # Sort by date
        df = df.sort_values('report_date').reset_index(drop=True)
        
        # Save to CSV
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"✓ Generated {num_records} records saved to {filepath}")
        
        return df
    
    def generate_uk_style_data(self):
        """
        Generate SYNTHETIC UK-council-style pothole records.
        Nothing is downloaded from data.gov.uk - all values are random draws.
        """
        print("Generating synthetic UK-style pothole data...")
        
        data = {
            'council': ['Birmingham', 'Manchester', 'Leeds', 'Liverpool', 'Newcastle'] * 200,
            'reference': [f'POT{i:05d}' for i in range(1000)],
            'location': np.random.choice(['A-Road', 'B-Road', 'Minor Road', 'Residential'], 1000),
            'reported_date': pd.date_range(start='2024-01-01', periods=1000, freq='D').tolist()[:1000],
            'completed_date': [None] * 1000,
            'repair_method': np.random.choice(['Patch', 'Permanent', 'Temporary'], 1000),
            'cost_gbp': np.random.uniform(50, 500, 1000)
        }
        
        df = pd.DataFrame(data)
        
        # Mark some as completed
        completed_indices = np.random.choice(1000, 600, replace=False)
        for idx in completed_indices:
            df.loc[idx, 'completed_date'] = df.loc[idx, 'reported_date'] + timedelta(days=np.random.randint(1, 30))
        
        filepath = os.path.join(self.data_dir, 'uk_style_synthetic.csv')
        df.to_csv(filepath, index=False)
        print(f"✓ UK data saved to {filepath}")
        
        return df
    
    def analyze_data(self, df: pd.DataFrame) -> Dict:
        """
        Extract summary statistics from a pothole record DataFrame
        """
        if df is None or len(df) == 0:
            return {}
        
        analysis = {
            'total_records': len(df),
            'date_range': {
                'start': df['report_date'].min() if 'report_date' in df else None,
                'end': df['report_date'].max() if 'report_date' in df else None
            },
            'severity_distribution': df['severity'].value_counts().to_dict() if 'severity' in df else {},
            'status_distribution': df['status'].value_counts().to_dict() if 'status' in df else {},
            'avg_depth_cm': df['depth_cm'].mean() if 'depth_cm' in df else None,
            'avg_area_sqm': df['area_sqm'].mean() if 'area_sqm' in df else None,
            'total_repair_cost': df['repair_cost'].sum() if 'repair_cost' in df else None,
            'avg_repair_cost': df['repair_cost'].mean() if 'repair_cost' in df else None,
            'segments_affected': df['segment_id'].nunique() if 'segment_id' in df else None
        }
        
        return analysis
    
    def convert_to_simulation_format(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert records to the simulation-compatible format
        """
        potholes = []
        
        for idx, row in df.iterrows():
            pothole = {
                'id': idx,
                'segment_id': int(row['segment_id']) if 'segment_id' in row else np.random.randint(0, 50),
                'severity': row['severity'] if 'severity' in row else 'moderate',
                'depth': float(row['depth_cm']) if 'depth_cm' in row else 5.0,
                'area': float(row['area_sqm']) if 'area_sqm' in row else 1.0,
                'age': 0,
                'status': row['status'] if 'status' in row else 'reported',
                'traffic_volume': int(row['traffic_volume']) if 'traffic_volume' in row else 10000
            }
            potholes.append(pothole)
        
        return potholes
    
    def get_historical_trends(self, df: pd.DataFrame) -> Dict:
        """
        Extract historical trends from the records
        """
        if 'report_date' not in df.columns:
            return {}
        
        df['report_date'] = pd.to_datetime(df['report_date'])
        df['month'] = df['report_date'].dt.to_period('M')
        
        monthly_counts = df.groupby('month').size().to_dict()
        monthly_costs = df.groupby('month')['repair_cost'].sum().to_dict() if 'repair_cost' in df else {}
        
        # Seasonal patterns
        df['season'] = df['report_date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        seasonal_distribution = df['season'].value_counts().to_dict()
        
        return {
            'monthly_counts': {str(k): v for k, v in monthly_counts.items()},
            'monthly_costs': {str(k): v for k, v in monthly_costs.items()},
            'seasonal_distribution': seasonal_distribution
        }


class DamageClassMix:
    """
    Synthetic samples that follow an APPROXIMATE road-damage type mix.
    The proportions are hand-entered, not computed from any dataset.
    """
    
    DAMAGE_TYPES = ['longitudinal_crack', 'transverse_crack', 'alligator_crack', 'pothole', 'other']
    TYPE_PROBS = [0.27, 0.24, 0.18, 0.23, 0.08]
    
    def write_class_mix_metadata(self, save_path='./synthetic_data/damage_class_mix.json'):
        """
        Write the hand-entered class mix to a small JSON file (nothing is downloaded)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        metadata = {
            'note': 'Hand-entered approximate values, not computed from any dataset',
            'damage_types': self.DAMAGE_TYPES,
            'proportions': dict(zip(self.DAMAGE_TYPES, self.TYPE_PROBS))
        }
        
        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Class-mix metadata saved to {save_path}")
        return metadata
    
    def simulate_from_class_mix(self, num_samples=1000) -> pd.DataFrame:
        """
        Generate SYNTHETIC detections whose damage-type mix follows the approximate proportions above
        """
        types = np.random.choice(self.DAMAGE_TYPES, num_samples, p=self.TYPE_PROBS)
        
        # Confidence scores
        confidences = np.random.beta(8, 2, num_samples)  # High confidence bias
        
        # Bounding box sizes (normalized)
        bbox_widths = np.random.uniform(0.1, 0.5, num_samples)
        bbox_heights = np.random.uniform(0.1, 0.5, num_samples)
        
        df = pd.DataFrame({
            'damage_type': types,
            'confidence': confidences,
            'bbox_width': bbox_widths,
            'bbox_height': bbox_heights,
            'area_normalized': bbox_widths * bbox_heights
        })
        
        # Map to our severity system
        df['severity'] = df['area_normalized'].apply(lambda x:
            'critical' if x > 0.15 else
            'severe' if x > 0.10 else
            'moderate' if x > 0.05 else
            'minor'
        )
        
        return df


def create_visualization_report(df: pd.DataFrame, output_path='./reports/'):
    """
    Create a visualization report of a pothole record DataFrame
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(output_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Synthetic Pothole Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Severity distribution
    if 'severity' in df.columns:
        severity_counts = df['severity'].value_counts()
        axes[0, 0].bar(severity_counts.index, severity_counts.values, 
                      color=['#2ecc71', '#f39c12', '#e74c3c', '#c0392b'])
        axes[0, 0].set_title('Severity Distribution')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Depth distribution
    if 'depth_cm' in df.columns:
        axes[0, 1].hist(df['depth_cm'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Depth Distribution (cm)')
        axes[0, 1].set_xlabel('Depth (cm)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(df['depth_cm'].mean(), color='red', linestyle='--', label=f'Mean: {df["depth_cm"].mean():.1f}')
        axes[0, 1].legend()
    
    # 3. Status distribution
    if 'status' in df.columns:
        status_counts = df['status'].value_counts()
        axes[0, 2].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                      colors=['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6'])
        axes[0, 2].set_title('Status Distribution')
    
    # 4. Timeline
    if 'report_date' in df.columns:
        df['report_date'] = pd.to_datetime(df['report_date'])
        df_sorted = df.sort_values('report_date')
        monthly = df_sorted.groupby(df_sorted['report_date'].dt.to_period('M')).size()
        axes[1, 0].plot(range(len(monthly)), monthly.values, marker='o', linewidth=2, color='#e74c3c')
        axes[1, 0].set_title('Monthly Reports Timeline')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Number of Reports')
        axes[1, 0].grid(alpha=0.3)
    
    # 5. Area vs Depth scatter
    if 'area_sqm' in df.columns and 'depth_cm' in df.columns:
        scatter = axes[1, 1].scatter(df['area_sqm'], df['depth_cm'], 
                                    c=pd.Categorical(df['severity']).codes if 'severity' in df else 'blue',
                                    cmap='RdYlGn_r', alpha=0.6, s=50)
        axes[1, 1].set_title('Area vs Depth')
        axes[1, 1].set_xlabel('Area (sq m)')
        axes[1, 1].set_ylabel('Depth (cm)')
        axes[1, 1].grid(alpha=0.3)
        if 'severity' in df:
            plt.colorbar(scatter, ax=axes[1, 1], label='Severity')
    
    # 6. Cost analysis
    if 'repair_cost' in df.columns:
        repaired = df[df['repair_cost'].notna()]
        if len(repaired) > 0:
            axes[1, 2].hist(repaired['repair_cost'], bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
            axes[1, 2].set_title('Repair Cost Distribution')
            axes[1, 2].set_xlabel('Cost ($)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].axvline(repaired['repair_cost'].mean(), color='red', linestyle='--',
                             label=f'Mean: ${repaired["repair_cost"].mean():.0f}')
            axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'synthetic_data_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}synthetic_data_analysis.png")
    plt.close()
