# TeleopWithRL

Reinforcement learning workspace for bilateral pneumatic teleoperation.

**Read this repo notebook-first. The most important files are the Jupyter
notebooks in `notebooks/`.** They are the main project record: they explain the
experiment flow, compare runs, collect results, and make the RL work easier to
review. The Python and MATLAB folders are supporting source code for those
notebooks.

## Repository Layout

- `notebooks/`
  - **Primary project workspace. Start here.**
  - Divided by topic: repo map, MATLAB parity, baselines, ablations,
    generalization, policy-gradient methods, and results cataloging.
- `matlab_env/`
  - Actual MATLAB/Simulink reference environment.
  - Contains the focused `SimuOriginal.slx` model used as the MATLAB reference.
- `matlab_env_python_replica/`
  - Python environment that mirrors the MATLAB/Simulink plant as literally as
    possible.
  - Contains the SimuOriginal replica, Gymnasium environment wrapper, shared
    utilities, algorithm packages, and notebook-called experiment launchers.
- `results_index/`
  - Organized, notebook-friendly result catalog.
  - Raw generated result folders are kept out of Git; this folder tracks the
    curated index that notebooks can read.
- Replica source packages
  - `matlab_env_python_replica/environment/` contains the plant and RL wrapper.
  - `matlab_env_python_replica/config/` contains shared constants.
  - `matlab_env_python_replica/dqn/`, `ql/`, and `policy_gradient/` contain
    algorithm-specific agents, training functions, and scripts.
  - `matlab_env_python_replica/common/` contains shared orchestration,
    evaluation, reward, and result utilities.

## Notebook Map

The notebooks are intentionally separated into their own top-level folder.

- `notebooks/00_repo_map.ipynb`: quick map of the repo and active experiment
  roots.
- `notebooks/10_matlab_literal_env/10_io_parity.ipynb`: MATLAB-literal parity
  and plant I/O inspection.
- `notebooks/20_baselines/20_dqn_legacy_baseline.ipynb`: DQN baseline record.
- `notebooks/20_baselines/21_ql_workspace.ipynb`: Q-learning workspace.
- `notebooks/30_ablations/30_reward_state_ablation.ipynb`: reward/state
  ablation review.
- `notebooks/40_generalization/40_waveform_generalization.ipynb`: waveform and
  input generalization.
- `notebooks/50_policy_gradient/`: PPO, TD3, SAC, and policy-gradient planning.
- `notebooks/90_results/90_results_catalog.ipynb`: organized results catalog.

## Current Results Snapshot

The latest comparable protocol is the fair, force-bias-15 PPO study: 500,000
training steps, 32 test episodes, one training/evaluation signal, and 25
focused evaluation scenarios. The tracked tables use focused-evaluation
metrics consistently; lower tracking RMSE is better, while a transparency ratio
near `1.0` is better.

| Study | Candidate | Focused tracking RMSE | Ratio statistic | Status |
|---|---|---:|---:|---|
| Reward ablation | `R5_second_order` | 8.386 mm | 1.009 | balanced reward candidate |
| Physics-informed | `F4_accel_state` | 3.535 mm | 0.776 | strongest tracking candidate |
| Temporal observation | `T3_posvel_current` | 5.177 mm | 1.075 | best tracking/transparency balance |
| Auxiliary GRU | none selected | 76.529–123.245 mm | 0.000–0.366 | not ready for selection |

These are not yet multi-seed conclusions. The exact 25-variant tables and all
comparison graphs are reproduced below. The machine-readable copy is
[`results_index/runs.csv`](results_index/runs.csv).

### Result figures

![Reward ablation comparison](results_index/figures/reward_ablation_summary.png)

![Reward ablation training curves](results_index/figures/reward_ablation_training_curves.png)

![Reward ablation group heatmap](results_index/figures/reward_ablation_group_heatmap.png)

![Physics-informed comparison](results_index/figures/physics_informed_summary_bars.png)

![Physics-informed learning curves](results_index/figures/physics_informed_learning_curves.png)

![Physics-informed transparency ratio rollouts](results_index/figures/physics_informed_transparency_ratio_rollouts.png)

![Temporal observation comparison](results_index/figures/temporal_summary_bars.png)

![Temporal observation learning curves](results_index/figures/temporal_learning_curves.png)

![Auxiliary GRU comparison](results_index/figures/gru_auxiliary_summary.png)

The physics-informed and temporal bar charts use each study's overall test
aggregate fields; the detailed report tables use the focused 25-scenario
metrics. The figures are therefore comparison views, while the tables are the
canonical values for the current catalog.


## Exact Current Result Tables

The common protocol is 500,000 PPO training steps, 32 test episodes, one
training/evaluation signal, and 25 focused evaluation scenarios per variant.
The tables use focused-evaluation aggregates. Tracking and post-contact RMSE
are in millimetres; transparency RMSE is in watts; the ratio should be close to
`1.0`. Ratio validity and within-20% coverage must be read alongside the ratio
because the velocity denominator can approach zero.

### Reward ablation: R0-R8

| ID | Formulation | Track RMSE [mm] | Post-contact [mm] | Transp. RMSE [W] | Ratio | Ratio error RMSE | Valid | Within +/-20% | RMS u [V] | Mean \|du\| [V] | Mean \|d2u\| [V] | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | e only | 7.333 | 6.527 | 1.677 | 0.044 | 241.511 | 66.1% | 3.6% | 3.576 | 6.490 | 12.966 | 0.0% |
| R1 | e + edot | 9.659 | 8.867 | 1.235 | 1.227 | 366.462 | 56.9% | 8.1% | 0.268 | 0.076 | 0.131 | 0.0% |
| R2 | Sliding | 9.276 | 8.334 | 1.198 | 1.226 | 669.090 | 56.8% | 8.6% | 0.274 | 0.109 | 0.200 | 0.0% |
| R3 | Sliding + du | 8.664 | 7.670 | 1.236 | 1.251 | 256.399 | 56.7% | 8.4% | 0.255 | 0.030 | 0.027 | 0.0% |
| R4 | Sliding + du + ddu | 8.718 | 7.691 | 1.232 | 1.261 | 356.875 | 56.7% | 8.2% | 0.248 | 0.027 | 0.020 | 0.0% |
| R5 | Second order | 8.386 | 7.162 | 1.272 | **1.009** | 688.738 | 55.5% | 8.2% | 0.239 | 0.025 | 0.015 | 0.0% |
| R6 | Lyapunov | 13.865 | 13.203 | 1.174 | 0.579 | 28.710 | 52.7% | 2.1% | 0.137 | 0.024 | 0.032 | 0.0% |
| R7 | Phase + direction | 12.093 | 11.262 | 1.156 | 0.593 | 184.424 | 53.1% | 1.7% | 0.148 | 0.029 | 0.040 | 0.0% |
| R8 | HF + deadzone | 11.952 | 11.454 | 1.189 | 1.099 | 71.228 | 53.4% | 1.8% | 0.158 | 0.031 | 0.043 | 0.0% |

### Physics-informed formulations: F0-F6

The physics-informed bar and learning-curve graphs use overall test aggregate
fields; this table uses the focused battery for cross-family comparison.

| ID | Formulation | Track RMSE [mm] | Post-contact [mm] | Transp. RMSE [W] | Ratio | Ratio error RMSE | Valid | Within +/-20% | RMS u [V] | Mean \|du\| [V] | Mean \|d2u\| [V] | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 | Baseline | 4.543 | 2.880 | 1.389 | 0.621 | 88.307 | 64.4% | 8.0% | 0.762 | 0.780 | 1.517 | 0.0% |
| F1 | Add Error | 4.667 | 3.198 | 1.435 | 0.455 | 1136.514 | 64.3% | 6.9% | 0.974 | 1.013 | 1.952 | 0.0% |
| F2 | Add Error Dot | 4.907 | 3.359 | 1.326 | 0.322 | 240.544 | 67.9% | 5.5% | 1.318 | 1.487 | 2.873 | 0.0% |
| F3 | Add Error DDot | 4.693 | 2.820 | 1.277 | 1.190 | 950.544 | 56.2% | 10.6% | 0.264 | 0.077 | 0.130 | 0.0% |
| F4 | Accel State | **3.535** | 2.411 | 1.348 | 0.776 | 144.719 | 64.8% | 9.6% | 0.922 | 1.099 | 2.090 | 0.0% |
| F5 | Accel State + Reward | 15.893 | 14.588 | 1.618 | 0.595 | 1946.017 | 57.0% | 13.0% | 0.466 | 0.075 | 0.084 | 0.0% |
| F6 | Effort + Delta U | 5.002 | 3.288 | 1.330 | **0.917** | 683.580 | 62.7% | 8.4% | 0.336 | 0.211 | 0.380 | 0.0% |

### Temporal observations: T0-T4

The temporal bar and learning-curve graphs use overall test aggregate fields;
this table uses the focused battery for cross-family comparison.

| ID | Formulation | Track RMSE [mm] | Post-contact [mm] | Transp. RMSE [W] | Ratio | Ratio error RMSE | Valid | Within +/-20% | RMS u [V] | Mean \|du\| [V] | Mean \|d2u\| [V] | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | Position Current | 6.487 | 5.028 | 1.664 | **1.045** | 18.305 | 55.4% | 6.6% | 0.302 | 0.103 | 0.183 | 0.0% |
| T1 | Position Stack 3 | 6.488 | 4.806 | 1.412 | **1.045** | 20.668 | 55.7% | 10.9% | 0.243 | 0.033 | 0.034 | 0.0% |
| T2 | Position Stack 5 | 8.071 | 6.521 | 1.873 | 0.003 | 77.438 | 77.2% | 3.7% | 4.831 | 9.606 | 19.217 | 0.0% |
| T3 | Position Velocity Current | 5.177 | 3.348 | 1.357 | 1.075 | 417.006 | 57.0% | 10.4% | 0.404 | 0.341 | 0.666 | 0.0% |
| T4 | Position Velocity Stack 3 | 4.967 | 2.898 | 1.393 | 0.505 | 319.730 | 64.2% | 6.1% | 1.194 | 1.324 | 2.576 | 0.0% |

### Auxiliary GRU-PPO: G0-G3

| ID | Formulation | Track RMSE [mm] | Post-contact [mm] | Transp. RMSE [W] | Ratio | Ratio error RMSE | Valid | Within +/-20% | RMS u [V] | Mean \|du\| [V] | Mean \|d2u\| [V] | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | GRU-PPO | 76.529 | 180.343 | 1.007 | 0.012 | 4.824 | 9.5% | 0.8% | 0.122 | 0.000 | 0.000 | 100.0% |
| G1 | GRU + prediction | 85.159 | 102.897 | 0.699 | 0.366 | 29.979 | 28.0% | 9.1% | 0.033 | 0.000 | 0.000 | 8.0% |
| G2 | GRU + hidden state | 123.245 | 158.799 | 0.647 | 0.290 | 1139.181 | 12.4% | 7.5% | 0.051 | 0.000 | 0.000 | 96.0% |
| G3 | GRU + both auxiliary heads | 118.588 | n/a* | 4.551 | 0.000 | 1.392 | 17.9% | 6.2% | 1.098 | 0.001 | 0.001 | 100.0% |

\* G3 had no valid post-contact segment; the source summary stores this as
zero, so it is represented as `n/a` rather than as a measured zero.

## MATLAB Assets

The MATLAB pieces are split deliberately:

- `matlab_env/` is for the actual MATLAB/Simulink environment.
- `matlab_env_python_replica/` is the Python reproduction of the MATLAB dynamics.

The active MATLAB-parity work is centered on `matlab_env/SimuOriginal.slx`.
The replica constants live in `matlab_env_python_replica/config/config.py`, and
the plant implementation lives in
`matlab_env_python_replica/environment/simuoriginal_replica.py`.

Current parity notes:

- The standalone Python replica reproduces the saved open-loop singularity at
  about `33.793 s`, consistent with the MATLAB observation that true open loop
  blows up around `34 s`.
- With reduced input `F_h(t) = 5 + 5 sin(0.5 t)`, the standalone replica stays
  bounded through `40 s`.
- Against MATLAB-side exported `gui_*` signals over `30 s`, the current replica
  tracks strongly in sign and shape:
  - `x_m` correlation `0.99895`
  - `x_s` correlation `0.99863`
  - `Fe` correlation `0.99751`

## Results Organization

Generated run folders are intentionally not committed. They can be large, noisy,
and machine-specific.

Tracked result organization lives in `results_index/`:

- `results_index/runs.csv` is the normalized flat catalog for the current
  comparable runs.
- `results_index/all_results.md` is the human-readable report containing every
  current variant and its graphs.
- `results_index/figures/` contains the small, tracked figures used by the
  documentation.
- `results_index/manifests/` is reserved for per-run JSON manifests.

The old MATLAB/DQN/Q-learning rows were removed from the current catalog because
their raw result roots are not present in this checkout. Some executed notebooks
still contain embedded historical tables or images; those are archive evidence,
not reproducible current rows, and remain identified as unavailable in the
relevant README files.

Raw generated outputs, when present locally, usually live under:

- `matlab_env_python_replica/results/`
- `matlab_env_python_replica/dqn_experiments/results/`
- `matlab_env_python_replica/ql_experiments/results/`
- `matlab_env_python_replica/policy_gradient_experiments/results/`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Open the notebooks with:

```powershell
jupyter lab notebooks
```

Run smoke tests with:

```powershell
python -m pytest
```

## Notes

- Keep notebooks in `notebooks/`; they are the main deliverable.
- Keep MATLAB/Simulink source in `matlab_env/`.
- Keep replica source and notebook-called experiment launchers in
  `matlab_env_python_replica/`.
- Keep generated results out of Git and index them through `results_index/`.
