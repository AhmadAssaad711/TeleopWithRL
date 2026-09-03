# Selected Results Report

This report intentionally presents only the selected models. The selection is
based on the final saved training telemetry checkpoint and then the focused
evaluation result.

## Protocol

- PPO on the Python SimuOriginal replica
- fair force-bias-15 evaluation protocol
- 500,000 training steps and 32 test episodes
- one training signal and one evaluation signal
- 25 focused evaluation scenarios per model

<a id="reward-ablation-r0-r8"></a>
<a id="physics-informed-formulations-f0-f6"></a>
<a id="temporal-observations-t0-t4"></a>
<a id="auxiliary-gru-ppo-g0-g3"></a>

## Selected models

| Model | Selection reason | Training endpoint track RMSE [mm] | Training endpoint transp. RMSE [W] | Focused eval track [mm] | Focused eval post-contact [mm] | Focused eval transp. [W] | Ratio | Ratio error RMSE | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `F4_accel_state` | best training and focused tracking | 3.377 | 1.566 | **3.535** | 2.411 | 1.348 | 0.776 | 144.719 | 64.8% |
| `R5_second_order` | closest focused ratio to 1.0 | 6.594 | 2.249 | 8.386 | 7.162 | 1.272 | **1.009** | 688.738 | 55.5% |
| `T3_posvel_current` | strongest temporal compromise | 5.440 | 2.335 | 5.177 | 3.348 | 1.357 | 1.075 | 417.006 | 57.0% |

`F4` is the primary candidate. `R5` and `T3` are retained as reward-design
and temporal-observation alternatives. GRU models are omitted because their
training telemetry and evaluation behavior did not meet the usable-model gate.

## Training results

The graph uses the saved `train.npz` export of the callback scalars
(`teleop_train` and `teleop_eval`) for the three selected runs. No TensorBoard
event files were captured in the current checkout, so this is the available
training-log export rather than a fabricated event-file chart.

![Selected-model training telemetry](figures/selected_models_training.png)

## Evaluation results

Evaluation is shown only with bar graphs. The bars use focused-evaluation
values; lower RMSE is better and the transparency ratio should be near `1.0`.

![Selected-model evaluation bars](figures/selected_models_evaluation_bars.png)

## Interpretation and limitations

The ratio-error values can be much larger than the ratio statistic because the
force/velocity ratio is ill-conditioned near zero velocity. Validity is shown
alongside the ratio for that reason. Repeat the selected configurations over
multiple seeds and signals before treating a candidate as final.

The machine-readable audit catalog remains in [`runs.csv`](runs.csv); the
README and this report intentionally promote only the selected models.
