#!/usr/bin/env python3
"""
Run Solar-CAP evaluation using preprocessed datasets produced by proproce-data.py.

This script reads the CSVs:
 - dataset_solar_real.csv   (N x T, per-node per-hour solar harvest in kWh)
 - dataset_carbon_real.csv  (N x T, per-node per-hour carbon intensity in gCO2eq/kWh)
 - dataset_demand_bursty.csv(N x T, per-node per-hour observed demand in kWh)

It builds a scenario dict compatible with run_experiments.* functions and evaluates
the standard policies: Always-on, Static-K, Greedy carbon-aware, and Solar-CAP.
Results (summary and hourly traces) are saved under results/preprocessed_csv/.

Notes:
- Bmax and Binit are inferred with sensible defaults but can be adjusted.
- p_cons is set to the per-node mean demand; p_mig = 0.1 * p_cons by default.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Ensure repository src is importable (this file lives in src/)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_experiments import (
    solar_cap_policy, baseline_policy, simulate_activation,
    summarize, hourly_trace, DEFAULT_CONFIG
)

# Configurable defaults
DEFAULT_K = 3
DEFAULT_BMAX = 2.5  # kWh per node if no better info
BINIT_FRACTION = 0.35
MIGRATION_FRACTION = 0.10


def load_matrix(csv_path: Path) -> np.ndarray:
    """Load an N x T matrix saved without headers (as in proproce-data.py).
    Returns ndarray shape (N, T).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    return df.values


def build_scenario_from_preprocessed(solar_csv: Path, carbon_csv: Path, demand_csv: Path,
                                     K: int = DEFAULT_K) -> dict:
    """Create a scenario dict compatible with run_experiments functions.

    solar_csv, carbon_csv, demand_csv: paths to matrices (N x T) produced by proproce-data.py
    """
    solar = load_matrix(solar_csv)
    carbon = load_matrix(carbon_csv)
    demand = load_matrix(demand_csv)

    if solar.shape != carbon.shape or solar.shape != demand.shape:
        raise ValueError("Input matrices must have identical shapes (N x T)")

    N, T = solar.shape

    # Derive per-node parameters
    # p_cons: use mean observed demand per node (kWh)
    p_cons = np.mean(demand, axis=1)

    # p_mig: fraction of consumption
    p_mig = p_cons * MIGRATION_FRACTION

    # Bmax: use DEFAULT_BMAX vector unless demand suggests larger battery
    # Heuristic: ensure battery can cover ~4 hours of mean demand
    heuristic_bmax = np.maximum(DEFAULT_BMAX, 4.0 * p_cons)

    Bmax = heuristic_bmax
    Binit = np.clip(BINIT_FRACTION * Bmax, a_min=0.01, a_max=None)

    # CFR: use per-node hourly carbon; reduce to a representative per-node CFR
    # The run_experiments code expects CFR per node (gCO2eq/kWh); use daily mean
    CFR = np.mean(carbon, axis=1)

    scenario = {
        "seed": 0,
        "profile": "preprocessed_real",
        "N": int(N),
        "T": int(T),
        "K": int(K),
        "Bmax": Bmax,
        "Binit": Binit,
        "CFR": CFR,
        "p_cons": p_cons,
        "p_mig": p_mig,
        "solar": solar,
    }

    return scenario


def run_preprocessed(solar_csv: Path, carbon_csv: Path, demand_csv: Path,
                     out_dir: Path, config: dict):
    """Run evaluation for a single preprocessed day and save outputs.

    Produces:
    - out_dir / "preprocessed_summary.csv"
    - out_dir / "preprocessed_hourly.csv"
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = build_scenario_from_preprocessed(solar_csv, carbon_csv, demand_csv, K=config.get('K', DEFAULT_K))

    POLICIES = ["Always-on", "Static-K", "Greedy carbon-aware", "Solar-CAP"]

    summary_rows = []
    hourly_rows = []

    for pol in POLICIES:
        if pol == "Solar-CAP":
            result = solar_cap_policy(
                scenario,
                lookahead=config.get('lookahead', DEFAULT_CONFIG.get('lookahead', 6)),
                lambda_switch=config.get('lambda_switch', DEFAULT_CONFIG.get('lambda_switch', 0.03)),
                lambda_safety=config.get('lambda_safety', DEFAULT_CONFIG.get('lambda_safety', 0.20)),
                hysteresis=config.get('hysteresis', DEFAULT_CONFIG.get('hysteresis', 0.10))
            )
        else:
            activation = baseline_policy(scenario, pol)
            result = simulate_activation(scenario, activation)

        row = summarize(scenario, result, pol)
        summary_rows.append(row)
        hourly_rows.append(hourly_trace(scenario, result, pol))

    summary = pd.concat([pd.DataFrame([r]) for r in summary_rows], ignore_index=True)
    hourly = pd.concat(hourly_rows, ignore_index=True)

    summary.to_csv(out_dir / "preprocessed_summary.csv", index=False)
    hourly.to_csv(out_dir / "preprocessed_hourly.csv", index=False)

    print(f"Results saved to {out_dir}")
    print(f" - Summary: {summary.shape[0]} rows × {summary.shape[1]} cols")
    print(f" - Hourly: {hourly.shape[0]} rows × {hourly.shape[1]} cols")

    return summary, hourly


if __name__ == "__main__":
    # Paths to preprocessed matrices (created by proproce-data.py)
    solar_csv = ROOT / "dataset_solar_real.csv"
    carbon_csv = ROOT / "dataset_carbon_real.csv"
    demand_csv = ROOT / "dataset_demand_bursty.csv"

    out_dir = ROOT.parent / "results" / "preprocessed_csv"

    cfg = DEFAULT_CONFIG.copy()

    # Allow overriding K via environment variable or small CLI parse
    import argparse
    parser = argparse.ArgumentParser(description="Run Solar-CAP on preprocessed datasets")
    parser.add_argument("--K", type=int, default=cfg.get('K', 3), help="Quorum K")
    parser.add_argument("--out", type=str, default=str(out_dir), help="Output directory")
    args = parser.parse_args()

    cfg['K'] = args.K
    out_dir = Path(args.out)

    run_preprocessed(solar_csv, carbon_csv, demand_csv, out_dir, cfg)
