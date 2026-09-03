# Policy-gradient scripts

These entry points launch policy-gradient training, formulation studies, and
focused evaluation. Reusable PPO/TD3/SAC logic remains in
`policy_gradient.training`; cross-algorithm evaluation remains in `common/`.

See [`../../CLI.md`](../../CLI.md) for the complete option and working-directory
contract. From the repository root, use the `matlab_env_python_replica...`
module prefix; notebooks use the fully qualified prefix from the repository's
parent directory.

Main entry points include:

```powershell
python -m TeleopWithRL.matlab_env_python_replica.policy_gradient.scripts.run_experiments
python -m TeleopWithRL.matlab_env_python_replica.policy_gradient.scripts.run_baselines_both_fe
python -m TeleopWithRL.matlab_env_python_replica.policy_gradient.scripts.run_physics_informed_formulations
python -m TeleopWithRL.matlab_env_python_replica.policy_gradient.scripts.run_physics_reward_ablation_basic_obs
python -m TeleopWithRL.matlab_env_python_replica.policy_gradient.scripts.run_temporal_observation_stack
python -m TeleopWithRL.matlab_env_python_replica.policy_gradient.scripts.run_auxiliary_gru_ppo
```

The scripts write to the established
`matlab_env_python_replica/policy_gradient_experiments/results/` location.
