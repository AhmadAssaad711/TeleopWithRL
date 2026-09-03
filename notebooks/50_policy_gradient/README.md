# Policy-gradient studies

This section contains the PPO, TD3, SAC, and recurrent policy-gradient
notebooks. The README records only the selected continuous-PPO models from the
current fair-bias-15 protocol.

## Selected models

`F4_accel_state` is the primary tracking candidate. `T3_posvel_current` is
retained as the temporal-observation alternative because it gives the best
tracking/transparency compromise among the temporal candidates.

| Model | Selection reason | Training endpoint track RMSE [mm] | Training endpoint transp. RMSE [W] | Focused eval track [mm] | Focused eval post-contact [mm] | Focused eval transp. [W] | Ratio | Ratio error RMSE | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `F4_accel_state` | best training and focused tracking | 3.377 | 1.566 | **3.535** | 2.411 | 1.348 | 0.776 | 144.719 | 64.8% |
| `T3_posvel_current` | strongest temporal compromise | 5.440 | 2.335 | 5.177 | 3.348 | 1.357 | 1.075 | 417.006 | 57.0% |

## Training and evaluation graphs

The training graph uses the saved `train.npz` export of the callback scalars
(`teleop_train` and `teleop_eval`). The current checkout has no TensorBoard
event files in the `ppo/tb` folders. Evaluation is shown only with bar graphs.

![Selected-model training telemetry](../../results_index/figures/selected_models_training.png)

![Selected-model evaluation bars](../../results_index/figures/selected_models_evaluation_bars.png)

The reward alternative `R5_second_order` is documented in
[`../30_ablations/README.md`](../30_ablations/README.md). All selected results
are single-study results and still require multi-seed validation.
