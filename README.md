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

These are not yet multi-seed conclusions. The complete 25-variant table,
validity checks, and figure gallery are in
[`results_index/all_results.md`](results_index/all_results.md).

### Result figures

<details>
<summary>Open the current comparison graphs</summary>

![Reward ablation comparison](results_index/figures/reward_ablation_summary.png)

![Physics-informed comparison](results_index/figures/physics_informed_summary_bars.png)

![Temporal observation comparison](results_index/figures/temporal_summary_bars.png)

![Auxiliary GRU comparison](results_index/figures/gru_auxiliary_summary.png)

The physics-informed and temporal bar charts use each study's overall test
aggregate fields; the detailed report tables use the focused 25-scenario
metrics. The figures are therefore comparison views, while the tables are the
canonical values for the current catalog.

</details>

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
