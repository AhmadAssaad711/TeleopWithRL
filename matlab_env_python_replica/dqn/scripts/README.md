# DQN scripts

These are executable entry points, not a second copy of DQN training logic.
They parse experiment arguments and delegate to `dqn.training` and the shared
orchestration code.

```powershell
python -m TeleopWithRL.matlab_env_python_replica.dqn.scripts.run_experiments
python -m TeleopWithRL.matlab_env_python_replica.dqn.scripts.run_baselines_both_fe
```

The scripts write artifacts to the established
`matlab_env_python_replica/dqn_experiments/results/` location so existing
notebooks and saved runs remain readable.
