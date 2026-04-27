# Results Index Workspace

This folder is the landing zone for a normalized, notebook-friendly view of the
repo's experiment artifacts.

Raw generated result trees are intentionally kept out of Git:

- `python_env/results/`
- `matlab_literal_env/results/`
- `matlab_literal_env/dqn_experiments/results/`
- `matlab_literal_env/ql_experiments/results/`
- `matlab_literal_env/policy_gradient_experiments/results/`

Why this folder exists:

- result naming is currently inconsistent across study families
- many paths are long enough to trigger Windows path-length failures
- notebooks need short, stable references that are easier to query

Target model for each indexed run:

- `run_id`
- `family`
- `source_root`
- `raw_path`
- `env_mode`
- `fe_mode`
- `agent`
- `state_variant`
- `reward_variant`
- `episode_duration`
- `switch_time`
- `train_episodes`
- `test_episodes`
- `summary_path`
- `plots_path`
- `model_path`

Current contents:

- `manifests/`
  - placeholder for one JSON manifest per normalized run
- `runs.csv`
  - generated flat catalog used by the notebooks

The notebook entry point for this workspace is:

- `notebooks/90_results/90_results_catalog.ipynb`
