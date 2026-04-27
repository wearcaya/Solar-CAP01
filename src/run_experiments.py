#!/usr/bin/env python3
"""
Reproducible synthetic evaluation for Solar-CAP.

This script generates deterministic daily solar-edge scenarios, evaluates
baselines and the Solar-CAP scheduler, and exports CSV tables and PDF/PNG
figures used by the IEEE manuscript.

The manuscript intentionally avoids low-level implementation details. This
artifact is provided only for reproducibility.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 180
plt.rcParams["savefig.dpi"] = 600

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
PROFILES = ["sunny", "cloudy", "intermittent", "stormy"]

DEFAULT_CONFIG = {
    "N": 8,
    "K": 3,
    "T": 24,
    "lookahead": 6,
    "lambda_switch": 0.03,
    "lambda_safety": 0.20,
    "hysteresis": 0.10,
    "energy_partition_threshold": 0.05,
}


def generate_scenario(seed: int, profile: str, N: int = 8, T: int = 24, K: int = 3) -> Dict:
    """Generate one deterministic daily renewable-powered edge scenario."""
    rng = np.random.default_rng(seed)
    hours = np.arange(T)
    hod = hours % 24

    # Energy values are expressed in kWh per time slot.
    Bmax = rng.uniform(1.2, 2.8, N)
    Binit = rng.uniform(0.15, 0.40, N) * Bmax
    CFR = rng.uniform(300, 700, N)  # gCO2eq/kWh
    p_cons = rng.uniform(0.28, 0.50, N)
    p_mig = rng.uniform(0.04, 0.12, N)
    solar_capacity = rng.uniform(0.25, 0.85, N)

    params = {
        "sunny": (0.90, 1.15, 0.03, 0.02),
        "cloudy": (0.45, 0.75, 0.08, 0.10),
        "intermittent": (0.60, 1.00, 0.10, 0.22),
        "stormy": (0.15, 0.42, 0.08, 0.35),
    }
    lo, hi, noise, dip_prob = params[profile]
    daily_factor = rng.uniform(lo, hi)

    solar = np.zeros((N, T), dtype=float)
    for t, h in enumerate(hod):
        if 6 <= h <= 18:
            base_curve = math.sin(math.pi * (h - 6) / 12.0)
        else:
            base_curve = 0.0

        for i in range(N):
            value = solar_capacity[i] * daily_factor * base_curve
            value += rng.normal(0.0, noise * solar_capacity[i])
            if rng.random() < dip_prob and 9 <= h <= 17:
                value *= rng.uniform(0.03, 0.50)
            solar[i, t] = max(0.0, value)

    return {
        "seed": seed,
        "profile": profile,
        "N": N,
        "T": T,
        "K": K,
        "Bmax": Bmax,
        "Binit": Binit,
        "CFR": CFR,
        "p_cons": p_cons,
        "p_mig": p_mig,
        "solar": solar,
    }


def simulate_activation(scenario: Dict, activation: np.ndarray) -> Dict:
    """Evaluate a fixed activation matrix under battery and brown-energy accounting."""
    N, T = scenario["N"], scenario["T"]
    Bmax = scenario["Bmax"]
    B = scenario["Binit"].copy()
    solar = scenario["solar"]
    p_cons = scenario["p_cons"]
    p_mig = scenario["p_mig"]

    battery = np.zeros((N, T), dtype=float)
    brown = np.zeros((N, T), dtype=float)
    migration = np.zeros((N, T), dtype=int)
    prev = np.zeros(N, dtype=int)

    for t in range(T):
        curr = activation[:, t].astype(int)
        migration[:, t] = (curr != prev).astype(int)
        demand = curr * p_cons + migration[:, t] * p_mig
        available = B + solar[:, t]
        brown[:, t] = np.maximum(0.0, demand - available)
        B = np.minimum(Bmax, available + brown[:, t] - demand)
        battery[:, t] = B
        prev = curr

    return {
        "x": activation.astype(int),
        "z": migration,
        "B": battery,
        "E_brown": brown,
    }


def baseline_policy(scenario: Dict, name: str) -> np.ndarray:
    """Return activation matrices for the reference baselines."""
    N, T, K = scenario["N"], scenario["T"], scenario["K"]
    CFR = scenario["CFR"]
    p_cons = scenario["p_cons"]
    p_mig = scenario["p_mig"]
    Bmax = scenario["Bmax"]
    solar = scenario["solar"]

    if name == "Always-on":
        return np.ones((N, T), dtype=int)

    if name == "Static-K":
        chosen = np.argsort(CFR)[:K]
        x = np.zeros((N, T), dtype=int)
        x[chosen, :] = 1
        return x

    if name == "Greedy carbon-aware":
        x = np.zeros((N, T), dtype=int)
        B = scenario["Binit"].copy()
        prev = np.zeros(N, dtype=int)

        for t in range(T):
            demand_if_on = p_cons + (prev == 0) * p_mig
            available = B + solar[:, t]
            brown = np.maximum(0.0, demand_if_on - available)
            score = brown * CFR + 0.10 * (prev == 0) * p_mig * CFR + 0.02 * CFR
            chosen = np.argsort(score)[:K]

            curr = np.zeros(N, dtype=int)
            curr[chosen] = 1
            z = (curr != prev).astype(int)
            demand = curr * p_cons + z * p_mig
            brown = np.maximum(0.0, demand - (B + solar[:, t]))
            B = np.minimum(Bmax, B + solar[:, t] + brown - demand)

            x[:, t] = curr
            prev = curr

        return x

    raise ValueError(f"unknown policy: {name}")


def solar_cap_policy(scenario: Dict,
                     lookahead: int = 6,
                     lambda_switch: float = 0.03,
                     lambda_safety: float = 0.20,
                     hysteresis: float = 0.10) -> Dict:
    """
    Solar-CAP deterministic scheduler.

    For every time slot, all K-node quorums are scored over a short horizon using
    brown-energy emissions, switching cost, and low-battery risk. Hysteresis keeps
    the previous quorum when its score remains close to the best one.
    """
    N, T, K = scenario["N"], scenario["T"], scenario["K"]
    Bmax = scenario["Bmax"]
    B = scenario["Binit"].copy()
    solar = scenario["solar"]
    p_cons = scenario["p_cons"]
    p_mig = scenario["p_mig"]
    CFR = scenario["CFR"]

    subsets = list(itertools.combinations(range(N), K))
    x = np.zeros((N, T), dtype=int)
    z = np.zeros((N, T), dtype=int)
    battery = np.zeros((N, T), dtype=float)
    brown = np.zeros((N, T), dtype=float)
    prev = np.zeros(N, dtype=int)
    prev_subset = None

    def evaluate(subset: Tuple[int, ...], t0: int, B0: np.ndarray, prev_vec: np.ndarray) -> float:
        curr = np.zeros(N, dtype=int)
        curr[list(subset)] = 1
        Btmp = B0.copy()
        pvec = prev_vec.copy()
        score = 0.0

        for tau in range(t0, min(T, t0 + lookahead)):
            zz = (curr != pvec).astype(int) if tau == t0 else np.zeros(N, dtype=int)
            demand = curr * p_cons + zz * p_mig
            available = Btmp + solar[:, tau]
            brown_tau = np.maximum(0.0, demand - available)
            Btmp = np.minimum(Bmax, available + brown_tau - demand)

            score += np.sum(brown_tau * CFR) / 1000.0
            if tau == t0:
                score += lambda_switch * np.sum(zz)

            low = np.maximum(0.0, 0.15 * Bmax - Btmp) / Bmax
            score += lambda_safety * np.sum(low * curr)

            pvec = curr.copy()

        return score

    for t in range(T):
        scored = [(evaluate(subset, t, B, prev), subset) for subset in subsets]
        scored.sort(key=lambda item: item[0])
        best_score, best_subset = scored[0]

        if prev_subset is not None:
            prev_score = evaluate(prev_subset, t, B, prev)
            if prev_score <= best_score * (1.0 + hysteresis) + 1e-12:
                best_subset = prev_subset

        curr = np.zeros(N, dtype=int)
        curr[list(best_subset)] = 1
        z[:, t] = (curr != prev).astype(int)

        demand = curr * p_cons + z[:, t] * p_mig
        available = B + solar[:, t]
        brown[:, t] = np.maximum(0.0, demand - available)
        B = np.minimum(Bmax, available + brown[:, t] - demand)

        x[:, t] = curr
        battery[:, t] = B
        prev = curr
        prev_subset = best_subset

    return {
        "x": x,
        "z": z,
        "B": battery,
        "E_brown": brown,
    }


def summarize(scenario: Dict, result: Dict, policy: str) -> Dict:
    x = result["x"]
    z = result["z"]
    B = result["B"]
    E = result["E_brown"]
    CFR = scenario["CFR"][:, None]

    return {
        "policy": policy,
        "profile": scenario["profile"],
        "seed": scenario["seed"],
        "N": scenario["N"],
        "T": scenario["T"],
        "K": scenario["K"],
        "brown_kwh": float(np.sum(E)),
        "emissions_kgco2eq": float(np.sum(E * CFR) / 1000.0),
        "migrations": int(np.sum(z)),
        "avg_active": float(np.mean(np.sum(x, axis=0))),
        "min_active": int(np.min(np.sum(x, axis=0))),
        "k_satisfaction_pct": float(np.mean(np.sum(x, axis=0) >= scenario["K"]) * 100.0),
        "battery_depletion_events": int(np.sum(B <= (0.02 * scenario["Bmax"][:, None]))),
    }


def hourly_trace(scenario: Dict, result: Dict, policy: str) -> pd.DataFrame:
    rows = []
    for t in range(scenario["T"]):
        rows.append({
            "policy": policy,
            "profile": scenario["profile"],
            "seed": scenario["seed"],
            "hour": t,
            "solar_mean_kwh": float(np.mean(scenario["solar"][:, t])),
            "solar_total_kwh": float(np.sum(scenario["solar"][:, t])),
            "battery_mean_kwh": float(np.mean(result["B"][:, t])),
            "brown_total_kwh": float(np.sum(result["E_brown"][:, t])),
            "emissions_kgco2eq": float(np.sum(result["E_brown"][:, t] * scenario["CFR"]) / 1000.0),
            "active_nodes": int(np.sum(result["x"][:, t])),
            "migrations": int(np.sum(result["z"][:, t])),
        })
    return pd.DataFrame(rows)


def run_all(out_dir: Path, config: Dict = DEFAULT_CONFIG) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "csv").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)

    rows: List[Dict] = []
    hourly_rows: List[pd.DataFrame] = []
    detailed_rows: List[pd.DataFrame] = []

    policies = ["Always-on", "Static-K", "Greedy carbon-aware", "Solar-CAP"]

    for profile in PROFILES:
        for seed in SEEDS:
            scenario = generate_scenario(seed, profile, N=config["N"], T=config["T"], K=config["K"])

            for policy in policies:
                if policy == "Solar-CAP":
                    result = solar_cap_policy(
                        scenario,
                        lookahead=config["lookahead"],
                        lambda_switch=config["lambda_switch"],
                        lambda_safety=config["lambda_safety"],
                        hysteresis=config["hysteresis"],
                    )
                else:
                    result = simulate_activation(scenario, baseline_policy(scenario, policy))

                rows.append(summarize(scenario, result, policy))
                hourly_rows.append(hourly_trace(scenario, result, policy))

                if seed == 42 and profile == "intermittent":
                    # Detailed node-level trace for the representative scenario.
                    for i in range(scenario["N"]):
                        for t in range(scenario["T"]):
                            detailed_rows.append(pd.DataFrame([{
                                "policy": policy,
                                "node": i,
                                "hour": t,
                                "active": int(result["x"][i, t]),
                                "migration": int(result["z"][i, t]),
                                "battery_kwh": float(result["B"][i, t]),
                                "brown_kwh": float(result["E_brown"][i, t]),
                                "solar_kwh": float(scenario["solar"][i, t]),
                                "cfr_gco2eq_per_kwh": float(scenario["CFR"][i]),
                            }]))

    summary = pd.DataFrame(rows)
    hourly = pd.concat(hourly_rows, ignore_index=True)
    detailed = pd.concat(detailed_rows, ignore_index=True)

    # Add reductions relative to baselines per profile/seed.
    pivot = summary.pivot_table(index=["profile", "seed"], columns="policy", values="emissions_kgco2eq")
    for baseline in ["Always-on", "Static-K", "Greedy carbon-aware"]:
        key = f"reduction_vs_{baseline.replace(' ', '_').replace('-', '').lower()}_pct"
        summary[key] = summary.apply(
            lambda r: 100.0 * (1.0 - r["emissions_kgco2eq"] / pivot.loc[(r["profile"], r["seed"]), baseline])
            if pivot.loc[(r["profile"], r["seed"]), baseline] > 0 else np.nan,
            axis=1,
        )

    summary.to_csv(out_dir / "csv" / "scenario_policy_summary.csv", index=False)
    hourly.to_csv(out_dir / "csv" / "hourly_traces.csv", index=False)
    detailed.to_csv(out_dir / "csv" / "representative_node_trace.csv", index=False)

    aggregate = summary.groupby(["profile", "policy"], as_index=False).agg(
        emissions_mean=("emissions_kgco2eq", "mean"),
        emissions_std=("emissions_kgco2eq", "std"),
        brown_mean=("brown_kwh", "mean"),
        brown_std=("brown_kwh", "std"),
        migrations_mean=("migrations", "mean"),
        migrations_std=("migrations", "std"),
        k_satisfaction_mean=("k_satisfaction_pct", "mean"),
        depletion_mean=("battery_depletion_events", "mean"),
    )
    aggregate.to_csv(out_dir / "csv" / "aggregate_by_profile_policy.csv", index=False)

    # Overall aggregate used in the paper.
    overall = summary.groupby("policy", as_index=False).agg(
        emissions_mean=("emissions_kgco2eq", "mean"),
        emissions_std=("emissions_kgco2eq", "std"),
        brown_mean=("brown_kwh", "mean"),
        brown_std=("brown_kwh", "std"),
        migrations_mean=("migrations", "mean"),
        migrations_std=("migrations", "std"),
        k_satisfaction_mean=("k_satisfaction_pct", "mean"),
    )
    overall.to_csv(out_dir / "csv" / "overall_policy_comparison.csv", index=False)

    # Reduction table for Solar-CAP only.
    solar = summary[summary["policy"] == "Solar-CAP"].copy()
    red = solar.groupby("profile", as_index=False).agg(
        reduction_vs_always_mean=("reduction_vs_alwayson_pct", "mean"),
        reduction_vs_static_mean=("reduction_vs_statick_pct", "mean"),
        reduction_vs_greedy_mean=("reduction_vs_greedy_carbonaware_pct", "mean"),
        reduction_vs_always_std=("reduction_vs_alwayson_pct", "std"),
        reduction_vs_static_std=("reduction_vs_statick_pct", "std"),
        reduction_vs_greedy_std=("reduction_vs_greedy_carbonaware_pct", "std"),
    )
    red.to_csv(out_dir / "csv" / "solar_cap_reductions.csv", index=False)

    # Sensitivity over the switching weight on intermittent traces.
    sens_rows = []
    for beta in [0.00, 0.01, 0.03, 0.06, 0.10]:
        for seed in SEEDS:
            scenario = generate_scenario(seed, "intermittent", N=config["N"], T=config["T"], K=config["K"])
            result = solar_cap_policy(
                scenario,
                lookahead=config["lookahead"],
                lambda_switch=beta,
                lambda_safety=config["lambda_safety"],
                hysteresis=config["hysteresis"],
            )
            row = summarize(scenario, result, "Solar-CAP")
            row["switch_weight"] = beta
            sens_rows.append(row)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(out_dir / "csv" / "sensitivity_switch_weight.csv", index=False)

    make_latex_tables(summary, aggregate, overall, red, sens, out_dir / "tables")
    make_figures(summary, hourly, aggregate, red, sens, out_dir / "figures", config)


def pm(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def make_latex_tables(summary: pd.DataFrame,
                      aggregate: pd.DataFrame,
                      overall: pd.DataFrame,
                      reductions: pd.DataFrame,
                      sensitivity: pd.DataFrame,
                      out_dir: Path) -> None:
    order = ["Always-on", "Static-K", "Greedy carbon-aware", "Solar-CAP"]
    overall = overall.set_index("policy").loc[order].reset_index()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Overall comparison across 40 deterministic daily traces (10 seeds $\times$ 4 solar profiles).}",
        r"\label{tab:overall_results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Policy} & \textbf{CO$_2$e kg} & \textbf{Brown kWh} & \textbf{Migrations} \\",
        r"\midrule",
    ]
    for _, r in overall.iterrows():
        lines.append(f"{r['policy']} & {pm(r['emissions_mean'], r['emissions_std'])} & {pm(r['brown_mean'], r['brown_std'])} & {pm(r['migrations_mean'], r['migrations_std'], 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / "table_overall_results.tex").write_text("\n".join(lines), encoding="utf-8")

    # Solar-CAP reductions by profile.
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean CO$_2$e reduction of Solar-CAP relative to each baseline.}",
        r"\label{tab:reductions}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Profile} & \textbf{vs. Always-on} & \textbf{vs. Static-K} & \textbf{vs. Greedy} \\",
        r"\midrule",
    ]
    for _, r in reductions.iterrows():
        lines.append(
            f"{r['profile'].capitalize()} & "
            f"{pm(r['reduction_vs_always_mean'], r['reduction_vs_always_std'], 1)}\\% & "
            f"{pm(r['reduction_vs_static_mean'], r['reduction_vs_static_std'], 1)}\\% & "
            f"{pm(r['reduction_vs_greedy_mean'], r['reduction_vs_greedy_std'], 1)}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / "table_reductions.tex").write_text("\n".join(lines), encoding="utf-8")

    # Sensitivity.
    sensagg = sensitivity.groupby("switch_weight", as_index=False).agg(
        emissions_mean=("emissions_kgco2eq", "mean"),
        emissions_std=("emissions_kgco2eq", "std"),
        migrations_mean=("migrations", "mean"),
        migrations_std=("migrations", "std"),
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity of Solar-CAP to the switching penalty on intermittent traces.}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{ccc}",
        r"\toprule",
        r"\textbf{Switching weight} & \textbf{CO$_2$e kg} & \textbf{Migrations} \\",
        r"\midrule",
    ]
    for _, r in sensagg.iterrows():
        lines.append(f"{r['switch_weight']:.2f} & {pm(r['emissions_mean'], r['emissions_std'])} & {pm(r['migrations_mean'], r['migrations_std'], 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / "table_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")


def make_figures(summary: pd.DataFrame,
                 hourly: pd.DataFrame,
                 aggregate: pd.DataFrame,
                 reductions: pd.DataFrame,
                 sensitivity: pd.DataFrame,
                 out_dir: Path,
                 config: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = ["Always-on", "Static-K", "Greedy carbon-aware", "Solar-CAP"]

    # Representative energy dynamics.
    rep = hourly[(hourly["profile"] == "intermittent") & (hourly["seed"] == 42) & (hourly["policy"] == "Solar-CAP")].copy()
    fig, ax1 = plt.subplots(figsize=(7.2, 3.6))
    ax1.plot(rep["hour"], rep["solar_mean_kwh"], marker="o", linewidth=2, label="Mean solar harvest")
    ax1.plot(rep["hour"], rep["battery_mean_kwh"], marker="s", linewidth=2, label="Mean battery level")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("kWh per node")
    ax1.set_xticks(range(0, 24, 2))
    ax2 = ax1.twinx()
    ax2.bar(rep["hour"], rep["brown_total_kwh"], alpha=0.25, label="Brown energy")
    ax2.set_ylabel("Brown energy (kWh)")
    # Solar-defined nighttime / partition windows.
    norm_solar = rep["solar_mean_kwh"] / max(rep["solar_mean_kwh"].max(), 1e-12)
    for h, val in zip(rep["hour"], norm_solar):
        if val < config["energy_partition_threshold"]:
            ax1.axvspan(h - 0.5, h + 0.5, alpha=0.08)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title("Solar-CAP energy dynamics under intermittent solar supply")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_energy_dynamics.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_energy_dynamics.png", bbox_inches="tight")
    plt.close(fig)

    # Activation/migration timeline.
    fig, ax1 = plt.subplots(figsize=(7.2, 3.2))
    ax1.step(rep["hour"], rep["active_nodes"], where="mid", linewidth=2, label="Active nodes")
    ax1.axhline(config["K"], linestyle="--", linewidth=1.5, label=f"K-resilience boundary ({config['K']})")
    ax1.set_ylim(0, config["N"] + 0.5)
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Active nodes")
    ax1.set_xticks(range(0, 24, 2))
    ax2 = ax1.twinx()
    ax2.bar(rep["hour"], rep["migrations"], alpha=0.25, label="Migrations")
    ax2.set_ylabel("Migrations")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8)
    ax1.set_title("Activation and migration decisions")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_activation_migration.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_activation_migration.png", bbox_inches="tight")
    plt.close(fig)

    # Aggregate emissions by policy and profile.
    ag = aggregate.copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(PROFILES))
    width = 0.19
    for idx, pol in enumerate(policies):
        vals = [ag[(ag["profile"] == pr) & (ag["policy"] == pol)]["emissions_mean"].values[0] for pr in PROFILES]
        ax.bar(x + (idx - 1.5) * width, vals, width, label=pol)
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PROFILES])
    ax.set_ylabel("Mean CO$_2$e (kg/day)")
    ax.set_title("Carbon emissions by solar profile")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_emissions_by_profile.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_emissions_by_profile.png", bbox_inches="tight")
    plt.close(fig)

    # Migrations by policy.
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for idx, pol in enumerate(policies):
        vals = [ag[(ag["profile"] == pr) & (ag["policy"] == pol)]["migrations_mean"].values[0] for pr in PROFILES]
        ax.bar(x + (idx - 1.5) * width, vals, width, label=pol)
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PROFILES])
    ax.set_ylabel("Mean migrations/day")
    ax.set_title("Migration overhead by solar profile")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_migrations_by_profile.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_migrations_by_profile.png", bbox_inches="tight")
    plt.close(fig)

    # Heatmap of reduction vs baselines.
    heat = reductions.set_index("profile")[["reduction_vs_always_mean", "reduction_vs_static_mean", "reduction_vs_greedy_mean"]]
    heat.columns = ["Always-on", "Static-K", "Greedy"]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    im = ax.imshow(heat.values, aspect="auto")
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels([i.capitalize() for i in heat.index])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    ax.set_title("Solar-CAP CO$_2$e reduction by baseline")
    fig.colorbar(im, ax=ax, label="Reduction (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_reduction_heatmap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_reduction_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # Sensitivity Pareto.
    sensagg = sensitivity.groupby("switch_weight", as_index=False).agg(
        emissions_mean=("emissions_kgco2eq", "mean"),
        emissions_std=("emissions_kgco2eq", "std"),
        migrations_mean=("migrations", "mean"),
        migrations_std=("migrations", "std"),
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.errorbar(
        sensagg["migrations_mean"], sensagg["emissions_mean"],
        xerr=sensagg["migrations_std"], yerr=sensagg["emissions_std"],
        marker="o", linewidth=1.8, capsize=3,
    )
    for _, r in sensagg.iterrows():
        ax.annotate(f"{r['switch_weight']:.2f}", (r["migrations_mean"], r["emissions_mean"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Mean migrations/day")
    ax.set_ylabel("Mean CO$_2$e (kg/day)")
    ax.set_title("Carbon-migration sensitivity on intermittent traces")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_sensitivity_pareto.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_sensitivity_pareto.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "results"
    run_all(out, DEFAULT_CONFIG)
    print(f"Results written to {out}")
