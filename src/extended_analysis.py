#!/usr/bin/env python3
"""Additional deterministic analysis for the 10-page CLEI version.

This script does not change the core protocol. It creates secondary tables and
figures from the deterministic traces and a small scale sweep over network size
and resilience threshold. It is part of the reproducibility artifact, but the
paper describes only the scientific protocol and not this code.
"""
from pathlib import Path
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 180
plt.rcParams["savefig.dpi"] = 600

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_experiments import (SEEDS, PROFILES, generate_scenario, solar_cap_policy,
                             baseline_policy, simulate_activation, summarize)

POLICIES = ["Always-on", "Static-K", "Greedy carbon-aware", "Solar-CAP"]

def pm(mean, std, digits=2):
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def make_extended_outputs(root: Path):
    csv_dir = root / "results" / "csv"
    fig_dir = root / "results" / "figures"
    tab_dir = root / "results" / "tables"
    paper_fig_dir = root / "paper" / "figures"
    paper_tab_dir = root / "paper" / "tables"
    for d in [fig_dir, tab_dir, paper_fig_dir, paper_tab_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(csv_dir / "scenario_policy_summary.csv")
    hourly = pd.read_csv(csv_dir / "hourly_traces.csv")
    reductions = pd.read_csv(csv_dir / "solar_cap_reductions.csv")
    sens = pd.read_csv(csv_dir / "sensitivity_switch_weight.csv")

    # 1) Boxplot distribution of reductions vs baselines.
    solar = summary[summary.policy == "Solar-CAP"].copy()
    reduction_cols = [
        ("reduction_vs_alwayson_pct", "vs. Always-on"),
        ("reduction_vs_statick_pct", "vs. Static-K"),
        ("reduction_vs_greedy_carbonaware_pct", "vs. Greedy")
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    data = [solar[c].values for c, _ in reduction_cols]
    ax.boxplot(data, labels=[lab for _, lab in reduction_cols], showmeans=True)
    ax.set_ylabel("CO$_2$e reduction (%)")
    ax.set_title("Distribution of Solar-CAP reductions across deterministic traces")
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    for d in [fig_dir, paper_fig_dir]:
        fig.savefig(d / "fig_reduction_distribution.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_reduction_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 2) Hourly brown energy comparison for representative intermittent trace.
    rep = hourly[(hourly.profile == "intermittent") & (hourly.seed == 42)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for pol in POLICIES:
        r = rep[rep.policy == pol]
        ax.plot(r.hour, r.brown_total_kwh, marker='o', linewidth=1.7, label=pol)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Brown energy (kWh)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Hourly brown-energy demand under intermittent solar supply")
    ax.legend(fontsize=7, ncols=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for d in [fig_dir, paper_fig_dir]:
        fig.savefig(d / "fig_hourly_brown_by_policy.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_hourly_brown_by_policy.png", bbox_inches="tight")
    plt.close(fig)

    # 3) Empirical CDF of emissions by policy.
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    for pol in POLICIES:
        vals = np.sort(summary[summary.policy == pol].emissions_kgco2eq.values)
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where='post', label=pol)
    ax.set_xlabel("Daily CO$_2$e (kg)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Emission distribution across solar profiles and seeds")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for d in [fig_dir, paper_fig_dir]:
        fig.savefig(d / "fig_emission_cdf.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_emission_cdf.png", bbox_inches="tight")
    plt.close(fig)

    # 4) Scale sweep: network size and resilience threshold. Use a subset of seeds
    # to bound computational cost while maintaining deterministic replicates.
    scale_rows = []
    scales = [(6, 2), (8, 3), (10, 3), (10, 4), (12, 4)]
    for N, K in scales:
        for profile in PROFILES:
            for seed in SEEDS:
                scen = generate_scenario(seed, profile, N=N, T=24, K=K)
                for pol in POLICIES:
                    if pol == "Solar-CAP":
                        res = solar_cap_policy(scen, lookahead=6, lambda_switch=0.03, lambda_safety=0.20, hysteresis=0.10)
                    else:
                        res = simulate_activation(scen, baseline_policy(scen, pol))
                    row = summarize(scen, res, pol)
                    row["scale"] = f"N={N},K={K}"
                    scale_rows.append(row)
    scale = pd.DataFrame(scale_rows)
    scale.to_csv(csv_dir / "scale_sweep_summary.csv", index=False)

    piv = scale.pivot_table(index=["scale", "profile", "seed"], columns="policy", values="emissions_kgco2eq")
    sc_solar = scale[scale.policy == "Solar-CAP"].copy()
    sc_solar["reduction_vs_static_pct"] = sc_solar.apply(lambda r: 100*(1-r.emissions_kgco2eq/piv.loc[(r.scale, r.profile, r.seed), "Static-K"]), axis=1)
    scale_agg = sc_solar.groupby("scale", as_index=False).agg(
        emission_mean=("emissions_kgco2eq", "mean"),
        emission_std=("emissions_kgco2eq", "std"),
        reduction_static_mean=("reduction_vs_static_pct", "mean"),
        reduction_static_std=("reduction_vs_static_pct", "std"),
        migrations_mean=("migrations", "mean"),
        migrations_std=("migrations", "std"),
        k_sat_mean=("k_satisfaction_pct", "mean"),
    )
    scale_agg.to_csv(csv_dir / "scale_sweep_aggregate.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7.2, 3.3))
    xs = np.arange(len(scale_agg))
    ax1.bar(xs - 0.18, scale_agg.reduction_static_mean, 0.36, yerr=scale_agg.reduction_static_std, capsize=3, label="Reduction vs. Static-K")
    ax1.set_ylabel("CO$_2$e reduction (%)")
    ax1.set_ylim(0, max(5, scale_agg.reduction_static_mean.max() + scale_agg.reduction_static_std.max() + 8))
    ax1.set_xticks(xs)
    ax1.set_xticklabels(scale_agg.scale, rotation=20)
    ax2 = ax1.twinx()
    ax2.plot(xs + 0.18, scale_agg.migrations_mean, marker='o', linewidth=1.8, label="Migrations")
    ax2.set_ylabel("Mean migrations/day")
    ax1.set_title("Scale sensitivity of Solar-CAP")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    for d in [fig_dir, paper_fig_dir]:
        fig.savefig(d / "fig_scale_sensitivity.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_scale_sensitivity.png", bbox_inches="tight")
    plt.close(fig)

    # 5) Table: scale sensitivity.
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Scale sensitivity of Solar-CAP across network size and resilience thresholds.}",
        r"\label{tab:scale_sensitivity}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Scale} & \textbf{CO$_2$e kg} & \textbf{Reduction vs. Static-K} & \textbf{Migrations} \\",
        r"\midrule",
    ]
    for _, r in scale_agg.iterrows():
        lines.append(f"{r['scale']} & {pm(r['emission_mean'], r['emission_std'])} & {pm(r['reduction_static_mean'], r['reduction_static_std'], 1)}\\% & {pm(r['migrations_mean'], r['migrations_std'], 1)} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    for d in [tab_dir, paper_tab_dir]:
        (d / "table_scale_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")

    # 6) Table: solar profile definitions.
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Solar-profile factors used to stress the scheduler under reproducible energy-partition regimes.}",
        r"\label{tab:solar_profiles}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Profile} & \textbf{Irradiance range} & \textbf{Noise} & \textbf{Midday dip prob.} \\",
        r"\midrule",
        r"Sunny & $[0.90,1.15]$ & $0.03$ & $0.02$ \\",
        r"Cloudy & $[0.45,0.75]$ & $0.08$ & $0.10$ \\",
        r"Intermittent & $[0.60,1.00]$ & $0.10$ & $0.22$ \\",
        r"Stormy & $[0.15,0.42]$ & $0.08$ & $0.35$ \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    for d in [tab_dir, paper_tab_dir]:
        (d / "table_solar_profiles.tex").write_text("\n".join(lines), encoding="utf-8")

    # 7) Table: extended comparison/positioning dimensions.
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Positioning of Solar-CAP relative to representative scheduling and carbon-aware systems literature.}",
        r"\label{tab:positioning_extended}",
        r"\begin{tabular}{p{2.7cm}p{2.7cm}ccccp{3.3cm}}",
        r"\toprule",
        r"\textbf{Work} & \textbf{Primary target} & \textbf{Edge} & \textbf{Carbon} & \textbf{Renewables} & \textbf{Resilience} & \textbf{Main gap addressed here} \\",
        r"\midrule",
        r"Luo et al. \cite{luo2021resource} & Resource scheduling survey & \checkmark & -- & -- & partial & Taxonomy does not model renewable intermittency as a consistency/availability stressor. \\",
        r"Lee et al. \cite{lee2023consistency} & CAL theory for real-time systems & \checkmark & -- & -- & \checkmark & Does not optimize energy-source selection or brown-energy exposure. \\",
        r"CASPER \cite{casper2024} & Geo-distributed web services & -- & \checkmark & partial & SLO & Cloud-region scheduling without battery-backed edge quorum constraints. \\",
        r"Cucumber \cite{wiesner2022cucumber} & Renewable-aware admission control & partial & partial & \checkmark & -- & Delay-tolerant admission rather than resilience-preserving activation. \\",
        r"Carbon-aware edge studies \cite{asadov2025carbon,zhang2025energyefficiency} & Spatio-temporal workload shifting & \checkmark & \checkmark & partial & partial & Often treats energy as a cost signal, not as an operational partition. \\",
        r"Solar-CAP & Solar-powered edge quorum scheduling & \checkmark & \checkmark & \checkmark & \checkmark & Jointly models carbon, battery dynamics, energy partitions, and K-resilient activation. \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    for d in [tab_dir, paper_tab_dir]:
        (d / "table_positioning_extended.tex").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    make_extended_outputs(ROOT)
    print("Extended outputs generated.")
