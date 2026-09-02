# Policy Gradient

This package contains policy-gradient training functions, stable result-path
helpers, and executable study entry points for `matlab_env_python_replica`.

Supported algorithms:

- `ppo_continuous`
- `td3`
- `sac`
- `ppo_discrete`

Main entry points:

```powershell
python -m TeleopWithRL.matlab_env_python_replica.policy_gradient.scripts.run_experiments
python -m TeleopWithRL.matlab_env_python_replica.policy_gradient.scripts.run_baselines_both_fe
```

The notebooks under `notebooks/50_policy_gradient/` call the `both_fe` runner
with the exact same environment and reward formulation used in the DQN baseline
notebook, then render the saved summaries and plots inline.

Generated files remain under `../policy_gradient_experiments/results/` for
compatibility; the old directory is data storage only.
