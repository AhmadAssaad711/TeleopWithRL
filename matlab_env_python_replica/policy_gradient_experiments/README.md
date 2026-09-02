# Policy-Gradient Experiments

This workspace mirrors the DQN notebook workflow for policy-gradient methods on
`matlab_env_python_replica`.

Supported algorithms:

- `ppo_continuous`
- `td3`
- `sac`
- `ppo_discrete`

Main entry points:

```powershell
python -m TeleopWithRL.matlab_env_python_replica.policy_gradient_experiments.run_policy_gradient_experiments
python -m TeleopWithRL.matlab_env_python_replica.policy_gradient_experiments.run_policy_gradient_baselines_both_fe
```

The notebooks under `notebooks/50_policy_gradient/` call the `both_fe` runner
with the exact same environment and reward formulation used in the DQN baseline
notebook, then render the saved summaries and plots inline.
