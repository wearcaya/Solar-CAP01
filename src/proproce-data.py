import pandas as pd
import numpy as np
import requests
import datetime

# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================
N_NODES = 8
T_SLOTS = 24
DATE = "2024-12-21" # Solsticio para máxima variabilidad

# Coordenadas geográficas reales para los perfiles solares
LOCATIONS = {
    "Arequipa_PE": {"lat": -16.40, "lon": -71.53},  # Nodos 0, 1 (Sunny)
    "Seattle_US":  {"lat": 47.60, "lon": -122.33},  # Nodos 2, 3 (Cloudy)
    "Sydney_AU":   {"lat": -33.86, "lon": 151.20},  # Nodos 4, 5 (Intermittent)
    "Munich_DE":   {"lat": 48.13, "lon": 11.58}     # Nodos 6, 7 (Stormy/Variable)
}

# ==========================================
# 1. EXTRACCIÓN DE DATOS SOLARES REALES (Open-Meteo API)
# ==========================================
def fetch_real_solar_data():
    print("Descargando datos de irradiancia solar real...")
    solar_matrix = np.zeros((N_NODES, T_SLOTS))
    
    node_idx = 0
    for loc_name, coords in LOCATIONS.items():
        # Llamada a la API de Open-Meteo (Archivo histórico, sin API Key)
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={coords['lat']}&longitude={coords['lon']}&start_date={DATE}&end_date={DATE}&hourly=shortwave_radiation"
        response = requests.get(url).json()
        
        # Irradiancia en W/m2
        irradiance_w_m2 = np.array(response['hourly']['shortwave_radiation'])
        
        # Preprocesamiento: Convertir W/m2 a factor de cosecha por slot (kWh)
        # Asumiendo un panel de ~1m2 con 20% de eficiencia
        harvest_kwh = (irradiance_w_m2 * 0.20) / 1000.0
        
        # Asignar a dos nodos por ubicación para redundancia
        solar_matrix[node_idx] = harvest_kwh
        solar_matrix[node_idx + 1] = harvest_kwh * np.random.uniform(0.95, 1.05, T_SLOTS) # Ligero ruido local
        node_idx += 2
        
    return pd.DataFrame(solar_matrix, index=[f"Node_{i}" for i in range(N_NODES)])

# ==========================================
# 2. GENERACIÓN DE INTENSIDAD DE CARBONO BASADA EN DATOS REGIONALES
# ==========================================
def generate_regional_carbon_data():
    print("Generando perfiles de carbono basados en Electricity Maps...")
    carbon_matrix = np.zeros((N_NODES, T_SLOTS))
    
    # Base real aproximada (gCO2e/kWh) por región
    base_carbon = [
        250, 250, # N0, N1: Perú (Mixto: Gas/Hidro)
        40,  40,  # N2, N3: Seattle (Limpio: Mayoría Hidro)
        700, 700, # N4, N5: Sídney (Fósil: Mayoría Carbón)
        350, 350  # N6, N7: Múnich (Variable: Eólico/Solar vs Gas)
    ]
    
    for i in range(N_NODES):
        # Añadir fluctuación horaria realista (por ejemplo, picos nocturnos cuando baja la solar)
        fluctuation = np.sin(np.linspace(0, 2 * np.pi, T_SLOTS)) * 50
        carbon_matrix[i] = base_carbon[i] + fluctuation + np.random.normal(0, 10, T_SLOTS)
        
    # Limitar para no tener valores negativos en zonas limpias
    carbon_matrix = np.clip(carbon_matrix, a_min=10, a_max=1000)
    return pd.DataFrame(carbon_matrix, index=[f"Node_{i}" for i in range(N_NODES)])

# ==========================================
# 3. GENERACIÓN DE DEMANDA EN RÁFAGA (Simulación tipo Alibaba)
# ==========================================
def generate_bursty_workload():
    print("Generando trazas de carga en ráfaga (Alibaba style)...")
    # Los contenedores de Alibaba suelen estar en idle (20-30%) con picos esporádicos al 90-100%
    demand_matrix = np.zeros((N_NODES, T_SLOTS))
    
    for i in range(N_NODES):
        # Carga base simulando el consumo de 0.28 kWh mínimo del paper
        base_demand = np.full(T_SLOTS, 0.28)
        
        # Inyectar picos Poisson (bursts de cómputo)
        spikes = np.random.poisson(lam=0.5, size=T_SLOTS) * 0.10
        demand = base_demand + spikes
        
        # Limitar al máximo teórico del nodo en el paper (0.50 kWh)
        demand_matrix[i] = np.clip(demand, 0.28, 0.50)
        
    return pd.DataFrame(demand_matrix, index=[f"Node_{i}" for i in range(N_NODES)])

# ==========================================
# EJECUCIÓN PRINCIPAL Y EXPORTACIÓN
# ==========================================
if __name__ == "__main__":
    # 1. Obtener datos
    df_solar = fetch_real_solar_data()
    df_carbon = generate_regional_carbon_data()
    df_demand = generate_bursty_workload()
    
    # 2. Exportar a CSV para el simulador Solar-CAP
    df_solar.to_csv("dataset_solar_real.csv", header=False, index=False)
    df_carbon.to_csv("dataset_carbon_real.csv", header=False, index=False)
    df_demand.to_csv("dataset_demand_bursty.csv", header=False, index=False)
    
    print("\n¡Preprocesamiento completado! Se han generado los siguientes archivos:")
    print("- dataset_solar_real.csv   (Dimensiones: 8x24)")
    print("- dataset_carbon_real.csv  (Dimensiones: 8x24)")
    print("- dataset_demand_bursty.csv (Dimensiones: 8x24)")
    print("\nEstos archivos pueden ser leídos por numpy.loadtxt() en tu repositorio de GitHub.")