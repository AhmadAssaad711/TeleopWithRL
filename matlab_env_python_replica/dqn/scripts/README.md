# DQN scripts

These are executable entry points, not a second copy of DQN training logic.
They parse experiment arguments and delegate to `dqn.training` and the shared
orchestration code.

See [`../../CLI.md`](../../CLI.md) for the complete option and working-directory
contract. From the repository root, use the `matlab_env_python_replica...`
module prefix; notebooks use the fully qualified prefix from the repository's
parent directory.

```powershell
python -m TeleopWithRL.matlab_env_python_replica.dqn.scripts.run_experiments
python -m TeleopWithRL.matlab_env_python_replica.dqn.scripts.run_baselines_both_fe
```

The scripts write artifacts to the established
`matlab_env_python_replica/dqn_experiments/results/` location so existing
notebooks and saved runs remain readable.
