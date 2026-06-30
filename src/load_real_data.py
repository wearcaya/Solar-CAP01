#!/usr/bin/env python3
"""
Load real solar data from CSV and convert to Solar-CAP scenario format.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_real_scenario(csv_path: str, date: str, N: int = 8, T: int = 24, K: int = 3) -> Dict:
    """
    Carga un escenario a partir de datos reales.
    
    Args:
        csv_path: Ruta al CSV con datos reales
        date: Fecha en formato YYYY-MM-DD
        N: Número de nodos
        T: Horas en el período
        K: Quórum requerido
    
    Returns:
        Dict con estructura compatible con generate_scenario()
    
    Expected CSV columns:
        - timestamp: ISO format datetime
        - node_id: Node identifier (0, 1, 2, ...)
        - solar_irradiance_kwh: Solar generation in kWh
        - battery_capacity_kwh: Maximum battery capacity
        - carbon_footprint_gco2eg_kwh: Carbon footprint factor (gCO2eq per kWh)
        - power_consumption_kwh: Typical node consumption
    """
    df = pd.read_csv(csv_path)
    
    # Filtrar por fecha
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_filtered = df[df['timestamp'].dt.date == pd.to_datetime(date).date()]
    
    if df_filtered.empty:
        raise ValueError(f"No data found for {date}")
    
    # Separar por nodo (tomar primeros N)
    nodes = sorted(df_filtered['node_id'].unique())[:N]
    if len(nodes) < N:
        raise ValueError(f"Expected {N} nodes, found {len(nodes)}")
    
    # Crear matriz de irradiancia solar (N x T)
    solar = np.zeros((N, T), dtype=float)
    for i, node_id in enumerate(nodes):
        node_data = df_filtered[df_filtered['node_id'] == node_id].sort_values('timestamp')
        solar_values = node_data['solar_irradiance_kwh'].values[:T]
        if len(solar_values) < T:
            print(f"Warning: Node {node_id} has only {len(solar_values)} hours, padding with zeros")
        solar[i, :len(solar_values)] = solar_values
    
    # Extraer capacidades y parámetros por nodo
    # Usar el primer registro de cada nodo (mismo para todas las horas)
    Bmax = np.zeros(N)
    CFR = np.zeros(N)
    p_cons = np.zeros(N)
    
    for i, node_id in enumerate(nodes):
        node_first = df_filtered[df_filtered['node_id'] == node_id].iloc[0]
        Bmax[i] = node_first['battery_capacity_kwh']
        CFR[i] = node_first['carbon_footprint_gco2eg_kwh'] * 1000  # Convertir a gCO2eq
        p_cons[i] = node_first['power_consumption_kwh']
    
    # Batería inicial: 30-40% de capacidad máxima
    rng = np.random.default_rng(hash(date) % (2**31))
    Binit = rng.uniform(0.30, 0.40) * Bmax
    
    # Costo de migración: 10% del consumo
    p_mig = p_cons * 0.1
    
    return {
        "seed": hash(date) % (2**31),  # Pseudo-seed basado en fecha
        "profile": "real_data",
        "N": N,
        "T": T,
        "K": K,
        "Bmax": Bmax,
        "Binit": Binit,
        "CFR": CFR,
        "p_cons": p_cons,
        "p_mig": p_mig,
        "solar": solar,
        "date": date,
    }


def validate_solar_data(csv_path: str) -> Tuple[bool, str]:
    """
    Valida que el CSV cumpla los requisitos.
    
    Returns:
        (valid: bool, message: str)
    """
    try:
        df = pd.read_csv(csv_path)
        
        required_cols = {'timestamp', 'node_id', 'solar_irradiance_kwh', 
                         'battery_capacity_kwh', 'carbon_footprint_gco2eg_kwh', 
                         'power_consumption_kwh'}
        
        # Verificar columnas
        missing = required_cols - set(df.columns)
        if missing:
            return False, f"❌ Faltan columnas: {missing}"
        
        # Verificar tipos
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Verificar rangos
        if (df['solar_irradiance_kwh'] < 0).any():
            return False, "❌ Irradiancia solar no puede ser negativa"
        
        if (df['battery_capacity_kwh'] <= 0).any():
            return False, "❌ Capacidad de batería debe ser positiva"
        
        if (df['power_consumption_kwh'] <= 0).any():
            return False, "❌ Consumo de potencia debe ser positivo"
        
        # Estadísticas
        num_nodes = df['node_id'].nunique()
        date_min = df['timestamp'].min()
        date_max = df['timestamp'].max()
        num_records = len(df)
        
        msg = f"""✅ Validación exitosa
   - Nodos: {num_nodes}
   - Período: {date_min} a {date_max}
   - Registros: {num_records}
   - Registros por nodo: {num_records // num_nodes}
"""
        return True, msg
        
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_real_data.py <csv_path> [date]")
        print("Example: python load_real_data.py data/solar.csv 2024-06-15")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    date = sys.argv[2] if len(sys.argv) > 2 else "2024-06-15"
    
    # Validar
    valid, msg = validate_solar_data(csv_path)
    print(msg)
    
    if not valid:
        sys.exit(1)
    
    # Cargar
    try:
        scenario = load_real_scenario(csv_path, date)
        print(f"\n✅ Escenario cargado para {date}")
        print(f"   - Nodos: {scenario['N']}")
        print(f"   - Horas: {scenario['T']}")
        print(f"   - Quórum: {scenario['K']}")
        print(f"   - Solar range: {scenario['solar'].min():.4f} - {scenario['solar'].max():.4f} kWh")
        print(f"   - Battery capacity: {scenario['Bmax'].min():.2f} - {scenario['Bmax'].max():.2f} kWh")
        print(f"   - CFR range: {scenario['CFR'].min():.0f} - {scenario['CFR'].max():.0f} gCO2eq/kWh")
    except Exception as e:
        print(f"❌ Error loading scenario: {e}")
        sys.exit(1)
