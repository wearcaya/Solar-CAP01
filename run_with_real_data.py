#!/usr/bin/env python3
"""
Run Solar-CAP evaluation using real solar data instead of synthetic data.

This script demonstrates how to:
1. Load real solar data from CSV
2. Run all policies (Always-on, Static-K, Greedy, Solar-CAP)
3. Compare results with synthetic baseline
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from run_experiments import (
    solar_cap_policy, simulate_activation, baseline_policy, 
    summarize, hourly_trace, DEFAULT_CONFIG
)
from load_real_data import load_real_scenario, validate_solar_data


def run_evaluation_with_real_data(csv_path: str, dates: list, output_dir: Path, config: dict):
    """
    Ejecuta evaluación completa con datos reales.
    
    Args:
        csv_path: Path to CSV with real solar data
        dates: List of dates in YYYY-MM-DD format
        output_dir: Output directory for results
        config: Configuration dict with N, K, T, etc.
    """
    
    print("=" * 80)
    print("SOLAR-CAP EVALUATION WITH REAL DATA")
    print("=" * 80)
    
    # Validate data
    valid, msg = validate_solar_data(csv_path)
    print(f"\n{msg}")
    if not valid:
        return None
    
    POLICIES = ["Always-on", "Static-K", "Greedy carbon-aware", "Solar-CAP"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_rows = []
    hourly_rows = []
    
    print(f"\n📊 Processing {len(dates)} dates...\n")
    
    for date in dates:
        print(f"  📅 {date}...", end=" ", flush=True)
        
        try:
            # Load real scenario
            scenario = load_real_scenario(csv_path, date, 
                                         N=config.get('N', 8),
                                         T=config.get('T', 24),
                                         K=config.get('K', 3))
            
            # Evaluate each policy
            for policy_name in POLICIES:
                if policy_name == "Solar-CAP":
                    result = solar_cap_policy(
                        scenario,
                        lookahead=config.get('lookahead', 6),
                        lambda_switch=config.get('lambda_switch', 0.03),
                        lambda_safety=config.get('lambda_safety', 0.20),
                        hysteresis=config.get('hysteresis', 0.10)
                    )
                else:
                    activation = baseline_policy(scenario, policy_name)
                    result = simulate_activation(scenario, activation)
                
                # Record summary
                row = summarize(scenario, result, policy_name)
                summary_rows.append(row)
                
                # Record hourly traces
                hourly_rows.append(hourly_trace(scenario, result, policy_name))
            
            print("✅")
            
        except Exception as e:
            print(f"❌ {str(e)}")
            continue
    
    # Combine results
    summary = pd.concat([pd.DataFrame([r]) for r in summary_rows], ignore_index=True)
    hourly = pd.concat(hourly_rows, ignore_index=True)
    
    # Save results
    summary.to_csv(output_dir / "real_data_summary.csv", index=False)
    hourly.to_csv(output_dir / "real_data_hourly.csv", index=False)
    
    print(f"\n✅ Results saved to {output_dir}/")
    print(f"   - Summary: {len(summary)} rows × {len(summary.columns)} columns")
    print(f"   - Hourly: {len(hourly)} rows × {len(hourly.columns)} columns")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("SUMMARY BY POLICY")
    print("=" * 80)
    
    policy_comparison = summary.groupby('policy').agg({
        'emissions_kgco2eq': ['mean', 'std'],
        'brown_kwh': ['mean', 'std'],
        'migrations': ['mean', 'std'],
    }).round(3)
    
    print(policy_comparison)
    
    return summary, hourly


if __name__ == "__main__":
    
    # Configuration
    CSV_PATH = ROOT / "data" / "sample_solar_data.csv"
    DATES = ["2024-06-15", "2024-06-16", "2024-06-17"]
    OUTPUT_DIR = ROOT / "results_real_data"
    
    config = DEFAULT_CONFIG.copy()
    
    # Run evaluation
    results = run_evaluation_with_real_data(
        str(CSV_PATH), 
        DATES, 
        OUTPUT_DIR, 
        config
    )
    
    if results:
        summary, hourly = results
        
        # Show top insights
        print("\n" + "=" * 80)
        print("KEY INSIGHTS")
        print("=" * 80)
        
        solar_cap = summary[summary['policy'] == 'Solar-CAP']
        always_on = summary[summary['policy'] == 'Always-on']
        static_k = summary[summary['policy'] == 'Static-K']
        
        reduction_vs_alwayson = 100 * (1 - solar_cap['emissions_kgco2eq'].mean() / always_on['emissions_kgco2eq'].mean())
        reduction_vs_statick = 100 * (1 - solar_cap['emissions_kgco2eq'].mean() / static_k['emissions_kgco2eq'].mean())
        
        print(f"\n🎯 Solar-CAP Performance:")
        print(f"   - Avg emissions: {solar_cap['emissions_kgco2eq'].mean():.2f} ± {solar_cap['emissions_kgco2eq'].std():.2f} kgCO2eq")
        print(f"   - Reduction vs Always-on: {reduction_vs_alwayson:.1f}%")
        print(f"   - Reduction vs Static-K: {reduction_vs_statick:.1f}%")
        print(f"   - Avg migrations: {solar_cap['migrations'].mean():.1f}")
        print(f"   - K-satisfaction: {solar_cap['k_satisfaction_pct'].mean():.1f}%")
