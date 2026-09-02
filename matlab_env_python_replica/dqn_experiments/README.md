# DQN Experiments

This workspace is for DQN-only work on `matlab_env_python_replica`.

It is organized into three clear areas:

- runner scripts in this folder
  - focused launchers that save into `dqn_experiments/results/...`
- `results/`
  - all isolated DQN experiment outputs
  - `results/dyn/` for `switched_dynamics`
  - `results/gui/` for `gui_skin_locked`

Main entry points:

```powershell
python -m TeleopWithRL.matlab_env_python_replica.dqn_experiments.run_dqn_experiments
python -m TeleopWithRL.matlab_env_python_replica.dqn_experiments.run_dqn_baselines_both_fe
```

The old copied baseline snapshot was removed because it duplicated the live
environment. Use the shared code in `../studies/` and the
top-level `matlab_env_python_replica` modules instead.
