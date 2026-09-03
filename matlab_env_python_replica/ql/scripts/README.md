# Q-learning scripts

These entry points parse Q-learning study settings and delegate to
`ql.training` plus the shared runner/evaluation utilities.

See [`../../CLI.md`](../../CLI.md) for the complete option and working-directory
contract. From the repository root, use the `matlab_env_python_replica...`
module prefix; notebooks use the fully qualified prefix from the repository's
parent directory.

```powershell
python -m TeleopWithRL.matlab_env_python_replica.ql.scripts.run_experiments
python -m TeleopWithRL.matlab_env_python_replica.ql.scripts.run_baselines_both_fe
```

Artifacts continue to use the established
`matlab_env_python_replica/ql_experiments/results/` location.
