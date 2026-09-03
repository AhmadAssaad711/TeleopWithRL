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

## Current result summary

The current tracked policy-gradient result set contains 25 variants from four
fair-bias-15 PPO studies. The best current tracking candidate is
`F4_accel_state` at `3.535 mm` focused tracking RMSE. The closest reported
transparency-ratio statistic is `R5_second_order` at `1.009`. These are
single-study results and should be followed by multi-seed validation.

![Physics-informed summary](../../results_index/figures/physics_informed_summary_bars.png)

![Temporal observation summary](../../results_index/figures/temporal_summary_bars.png)

![Auxiliary GRU summary](../../results_index/figures/gru_auxiliary_summary.png)

The complete variant tables and additional learning/diagnostic graphs are in
[`../../results_index/all_results.md`](../../results_index/all_results.md).

The notebooks under `notebooks/50_policy_gradient/` call the same reusable
training and evaluation code and render the saved summaries and plots inline.

Generated files remain under `../policy_gradient_experiments/results/` for
compatibility; the old directory is data storage only. Tracked summary figures
are copied to `../../results_index/figures/` so documentation does not depend
on ignored local output trees.
