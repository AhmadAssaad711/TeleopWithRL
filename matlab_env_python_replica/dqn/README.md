# DQN

This package contains the DQN agent, state encoders, training functions, and
notebook/CLI entry points for `matlab_env_python_replica`.

- `agent.py`: reusable DQN implementation
- `state_variants.py`: named observation encoders
- `training.py`: environment factories, training, evaluation, and artifact writing
- `scripts/`: executable study launchers

Main entry points:

```powershell
python -m TeleopWithRL.matlab_env_python_replica.dqn.scripts.run_experiments
python -m TeleopWithRL.matlab_env_python_replica.dqn.scripts.run_baselines_both_fe
```

Generated files remain under `../dqn_experiments/results/` for compatibility;
the old directory is data storage only. Shared orchestration and evaluation
helpers are in `../common/`.

## Result status

No portable DQN result artifacts are present in this checkout. Executed
notebooks may retain embedded historical output, but the old result-index rows
pointed to missing paths and are therefore not reported as reproducible current
numbers. When DQN runs are restored, add only the selected baseline summary and
its evaluation bar graph using the same conventions as the current report.
