# Notebook Workspace

**This is the most important folder in the repo.** The Teleop-with-RL work
should be read notebook-first: notebooks are the project record, the experiment
map, and the main review surface. The source-code folders exist to support these
notebooks.

Notebook execution contract:

- `notebooks/_teleop_nb.py` locates the repository and selects the project
  Python interpreter.
- `run_notebook_command(...)` executes a configured repository command and
  fails the cell if the experiment fails.
- `run_module(...)` is available when a notebook wants to launch a Python
  module directly.
- Training and evaluation logic lives under
  `matlab_env_python_replica/{dqn,ql,policy_gradient,common}`; notebooks pass
  settings to those entry points and then read their artifacts.

Current sections:

- `00_repo_map.ipynb`
  - quick orientation to the repo and active experiment roots
- `10_matlab_literal_env/`
  - MATLAB-parity and plant-level inspection
  - includes `10_io_parity.ipynb` for viewing parity bundles
- `20_baselines/`
  - trusted baseline runs and first-pass comparisons
- `30_ablations/`
  - reward/state ablations
- `40_generalization/`
  - waveform and input-signal generalization
- `50_policy_gradient/`
  - policy-gradient problem formulation and implementation planning
  - includes `51_ppo_continuous_baseline.ipynb`
  - includes `52_td3_baseline.ipynb`
  - includes `53_sac_baseline.ipynb`
  - includes `54_ppo_discrete_baseline.ipynb`
- `90_results/`
  - results cataloging and indexing

The raw experiment outputs still live in:

- `matlab_env_python_replica/results/`
- `matlab_env_python_replica/dqn_experiments/results/`
- `matlab_env_python_replica/ql_experiments/results/`
- `matlab_env_python_replica/policy_gradient_experiments/results/`

The normalization target for those outputs is documented in:

- `results_index/README.md`

The current tracked result tables and figures are in:

- `results_index/all_results.md`
- `results_index/figures/`

The result READMEs are intentionally scoped to their experiment family. The
canonical report is the source for the complete 25-variant table, so a metric
does not need to be copied independently into every notebook description.
