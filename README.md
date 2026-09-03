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

Only the selected models are reported here. Selection uses the final saved
training telemetry checkpoint, then the focused evaluation result. The common
protocol is 500,000 PPO training steps, 32 test episodes, one signal pair, and
25 focused evaluation scenarios.

| Model | Why it is kept | Training endpoint track RMSE [mm] | Training endpoint transp. RMSE [W] | Focused eval track [mm] | Focused eval post-contact [mm] | Focused eval transp. [W] | Ratio | Ratio error RMSE | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `F4_accel_state` | best training and focused tracking | 3.377 | 1.566 | **3.535** | 2.411 | 1.348 | 0.776 | 144.719 | 64.8% |
| `R5_second_order` | closest focused ratio to 1.0 | 6.594 | 2.249 | 8.386 | 7.162 | 1.272 | **1.009** | 688.738 | 55.5% |
| `T3_posvel_current` | strongest temporal compromise | 5.440 | 2.335 | 5.177 | 3.348 | 1.357 | 1.075 | 417.006 | 57.0% |

`F4` is the primary candidate. `R5` and `T3` are retained as reward-design
and temporal-observation alternatives. The GRU models are omitted because
their training telemetry and evaluation behavior did not meet the usable-model
gate. These are single-study results, not multi-seed conclusions.

### Training results

The training graph uses the saved `train.npz` export of the callback scalars
(`teleop_train` and `teleop_eval`) for the three selected runs. No TensorBoard
event files were captured in the current checkout, so this is the available
training-log export rather than a fabricated event-file chart.

![Selected-model training telemetry](results_index/figures/selected_models_training.png)

### Evaluation results

Evaluation is shown only with bar graphs. The bars use focused-evaluation
values; lower RMSE is better and the transparency ratio should be near `1.0`.

![Selected-model evaluation bars](results_index/figures/selected_models_evaluation_bars.png)

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
- `results_index/all_results.md` is the human-readable report for the selected
  models and their graphs.
- `results_index/figures/` contains the small, tracked training and evaluation
  figures used by the documentation.
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
