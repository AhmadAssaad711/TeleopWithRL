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

## Selected results

The current report promotes only three models from the fair-bias-15 PPO
protocol: one primary tracking candidate and two useful alternatives.

| Model | Selection reason | Training endpoint track RMSE [mm] | Training endpoint transp. RMSE [W] | Focused eval track [mm] | Focused eval post-contact [mm] | Focused eval transp. [W] | Ratio | Ratio error RMSE | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `F4_accel_state` | best training and focused tracking | 3.377 | 1.566 | **3.535** | 2.411 | 1.348 | 0.776 | 144.719 | 64.8% |
| `R5_second_order` | closest focused ratio to 1.0 | 6.594 | 2.249 | 8.386 | 7.162 | 1.272 | **1.009** | 688.738 | 55.5% |
| `T3_posvel_current` | strongest temporal compromise | 5.440 | 2.335 | 5.177 | 3.348 | 1.357 | 1.075 | 417.006 | 57.0% |

`F4_accel_state` is the primary candidate. GRU models are omitted because
their training telemetry and evaluation behavior did not meet the usable-model
gate. These are single-study results and require multi-seed validation.

## Training and evaluation graphs

The training graph uses the saved `train.npz` export of the callback scalars
(`teleop_train` and `teleop_eval`). The current checkout has no TensorBoard
event files in the `ppo/tb` folders. Evaluation is shown only with bar graphs.

![Selected-model training telemetry](../../results_index/figures/selected_models_training.png)

![Selected-model evaluation bars](../../results_index/figures/selected_models_evaluation_bars.png)

Generated files remain under `../policy_gradient_experiments/results/` for
compatibility; the old directory is data storage only. The tracked figures are
copied to `../../results_index/figures/` so the README does not depend on
ignored local output trees.
