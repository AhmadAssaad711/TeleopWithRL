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
  - Contains the SimuOriginal replica, Gym-style environment wrapper, study
    code, and notebook-called experiment launchers.
- `results_index/`
  - Organized, notebook-friendly result catalog.
  - Raw generated result folders are kept out of Git; this folder tracks the
    curated index that notebooks can read.
- Root Python modules
  - `config.py`, `dqn_agent.py`, and `q_learning_agent.py` are kept because the
    MATLAB-literal notebook workflow imports them.

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

## MATLAB Assets

The MATLAB pieces are split deliberately:

- `matlab_env/` is for the actual MATLAB/Simulink environment.
- `matlab_env_python_replica/` is the Python reproduction of the MATLAB dynamics.

The active MATLAB-parity work is centered on `matlab_env/SimuOriginal.slx`.
The replica constants live in `matlab_env_python_replica/simuoriginal_replica.py`.

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

- `results_index/runs.csv` is the flat catalog used by notebooks.
- `results_index/manifests/` is reserved for per-run JSON manifests.

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
