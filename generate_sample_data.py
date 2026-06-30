#!/usr/bin/env python3
"""
Generate sample real solar data for testing.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_solar_data(output_path: str, start_date: str = "2024-06-15", 
                               num_days: int = 3, num_nodes: int = 8):
    """
    Genera datos solares realistas de ejemplo.
    
    Args:
        output_path: Path to save CSV
        start_date: Initial date in YYYY-MM-DD format
        num_days: Number of days to generate
        num_nodes: Number of nodes
    """
    rows = []
    start = pd.to_datetime(start_date)
    
    # Parámetros por nodo (simulan diferentes ubicaciones/orientaciones)
    np.random.seed(42)
    battery_capacities = np.random.uniform(2.0, 3.0, num_nodes)
    cfr_values = np.random.uniform(400, 600, num_nodes)
    consumption = np.random.uniform(0.30, 0.40, num_nodes)
    
    for day_offset in range(num_days):
        current_date = start + timedelta(days=day_offset)
        
        for hour in range(24):
            timestamp = current_date + timedelta(hours=hour)
            
            # Irradiancia solar realista: curva sinusoidal con ruido
            # Pico a mediodía, 0 en noche
            if 6 <= hour <= 18:
                base_irradiance = 0.8 * np.sin(np.pi * (hour - 6) / 12.0)
            else:
                base_irradiance = 0.0
            
            # Variar por nodo (diferentes orientaciones/sombreado)
            for node_id in range(num_nodes):
                # Ruido de nube (variación hasta ±30%)
                cloud_factor = np.random.normal(1.0, 0.15)
                cloud_factor = np.clip(cloud_factor, 0.5, 1.3)
                
                # Solape ocasional (5% chance de reducción drástica)
                if np.random.random() < 0.05 and 9 <= hour <= 17:
                    cloud_factor *= np.random.uniform(0.1, 0.5)
                
                solar_irradiance = max(0.0, base_irradiance * cloud_factor)
                
                rows.append({
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M'),
                    'node_id': node_id,
                    'solar_irradiance_kwh': round(solar_irradiance, 4),
                    'battery_capacity_kwh': round(battery_capacities[node_id], 2),
                    'carbon_footprint_gco2eg_kwh': round(cfr_values[node_id], 0),
                    'power_consumption_kwh': round(consumption[node_id], 3),
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df)} records in {output_path}")
    print(f"   - Dates: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   - Nodes: {df['node_id'].nunique()}")
    print(f"   - Records per node: {len(df) // df['node_id'].nunique()}")
    
    return df


if __name__ == "__main__":
    import sys
    
    output_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_solar_data.csv"
    
    df = generate_sample_solar_data(output_path)
    
    print(f"\n📊 Data Preview:")
    print(df.head(10))
