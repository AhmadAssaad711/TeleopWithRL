# Q-learning

This package contains the tabular Q-learning agent, state encoders, training
functions, and executable study entry points for `matlab_env_python_replica`.

- `agent.py`: reusable Q-learning implementation
- `state_variants.py`: named tabular state encoders
- `training.py`: training/evaluation functions and artifact writing
- `scripts/`: executable notebook/CLI launchers

Main notebook entry points:

```powershell
python -m TeleopWithRL.matlab_env_python_replica.ql.scripts.run_experiments
python -m TeleopWithRL.matlab_env_python_replica.ql.scripts.run_baselines_both_fe
```

Generated files remain under `../ql_experiments/results/` for compatibility;
the old directory is data storage only.

## Result status

No portable Q-learning result artifacts are present in this checkout. Executed
notebooks may retain embedded historical output, but the old result-index rows
pointed to missing paths and are therefore not reported as reproducible current
numbers. When Q-learning runs are restored, add their normalized summaries and
graphs to [`../../results_index/all_results.md`](../../results_index/all_results.md).
