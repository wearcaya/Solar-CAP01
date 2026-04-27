# Solar-CAP

**Carbon-Aware Resilient Scheduling for Solar-Powered Edge Computing under Energy Partition Events**

Solar-CAP is a scheduling model for solar-powered edge infrastructures where energy availability is not always reliable. Instead of treating energy only as a cost, this project models severe renewable scarcity as an **energy partition**: a situation where an edge node may still be reachable through the network, but cannot safely participate in service execution without relying on brown energy.

The goal is simple: keep the service resilient while using as little brown energy as possible.

## What Solar-CAP does

Solar-CAP decides which edge nodes should remain active over time by considering:

- local solar generation,
- battery state,
- brown-energy emissions,
- node activation and switching,
- and a minimum `K`-active-node quorum for service availability.

It aims to reduce emissions without breaking the required level of service continuity.

## Evaluation

The evaluation uses a deterministic and reproducible setup with:

- 40 daily traces,
- 10 fixed random seeds,
- 4 solar regimes:
  - sunny,
  - cloudy,
  - intermittent,
  - stormy.

Solar-CAP is compared against three baselines:

- **Always-on**
- **Static-K**
- **Greedy carbon-aware**

## Main result

Across all evaluated traces, Solar-CAP reduced CO₂e emissions by:

- **87.1%** compared with Always-on,
- **52.7%** compared with Static-K,
- **6.8%** compared with Greedy carbon-aware scheduling.

At the same time, it preserved the required active quorum in all evaluated time slots and avoided unnecessary switching compared with the greedy policy.

## Why it matters

Edge computing is moving closer to renewable-powered and geographically distributed deployments. However, solar energy is intermittent, and a node with low battery may become operationally unavailable even if the network link is still alive.

Solar-CAP captures this behavior directly and provides a practical way to reason about sustainability and resilience together.

## Repository contents

```text
.
├── configs/
│   └── solar_cap_protocol.json
│
├── docs/
│   └── README.md
│
├── notebooks/
│   ├── PLI_Hyrarchical_FLIpynb
│   ├── notebook_solar_aware_pulp_col...
│   └── solar_cap_reproducibility.ipynb
│
├── src/
│   └── ...
│
├── .gitignore
├── reproduce.sh
└── requirements.txt

