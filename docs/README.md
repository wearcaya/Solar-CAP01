# Solar-CAP CLEI 2026 Package

This package contains an expanded 10-page IEEE-style version of the paper, a reproducible synthetic evaluation, generated figures, tables, and raw CSV outputs.

## Main paper

- `paper/main.tex`: IEEEtran manuscript.
- `paper/content/`: section files.
- `paper/references.bib`: bibliographic database retained for editing; the compiled version uses an IEEE-style reference list in `main.tex` for portability.
- `paper/figures/`: vector PDF figures and PNG copies.
- `paper/tables/`: LaTeX tables generated from the experiment.

## Reproducibility

The experiment uses deterministic seeds:

`42, 43, 44, 45, 46, 47, 48, 49, 50, 51`

Profiles:

`sunny, cloudy, intermittent, stormy`

Protocol:

- `N = 8` edge nodes.
- `K = 3` active nodes.
- `T = 24` hourly slots.
- 40 daily traces: 10 seeds x 4 solar profiles.
- Baselines: Always-on, Static-K, Greedy carbon-aware, Solar-CAP.

To regenerate all CSVs, figures, and tables:

```bash
pip install -r requirements.txt
python src/run_experiments.py
```

Or:

```bash
./reproduce.sh
```

## Outputs

- `results/csv/scenario_policy_summary.csv`
- `results/csv/hourly_traces.csv`
- `results/csv/aggregate_by_profile_policy.csv`
- `results/csv/overall_policy_comparison.csv`
- `results/csv/solar_cap_reductions.csv`
- `results/csv/sensitivity_switch_weight.csv`
- `results/figures/*.pdf`
- `results/tables/*.tex`

## Notes

The manuscript avoids low-level implementation details. The code is provided as a reproducibility artifact only.


## 50+ reference CLEI revision

This revision expands the scholarly basis from 16 to 54 bibliography entries, adding recent 2024--2026 literature on renewable-aware edge orchestration, carbon-aware serverless/AI systems, software carbon accounting, and data-center energy demand. Figures are exported as vector PDF and high-resolution PNG (600 dpi).
