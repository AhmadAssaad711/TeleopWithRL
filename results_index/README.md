# Results index

This README intentionally reports only the selected current models. It is the
short review surface for the experiment results; the raw training trees remain
local because they contain large model and history artifacts.

## Protocol

- PPO on the Python SimuOriginal replica
- fair force-bias-15 evaluation protocol
- 500,000 training steps and 32 test episodes
- one training signal and one evaluation signal
- 25 focused evaluation scenarios per selected model

## Selected results

| Model | Selection reason | Training endpoint track RMSE [mm] | Training endpoint transp. RMSE [W] | Focused eval track [mm] | Focused eval post-contact [mm] | Focused eval transp. [W] | Ratio | Ratio error RMSE | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `F4_accel_state` | best training and focused tracking | 3.377 | 1.566 | **3.535** | 2.411 | 1.348 | 0.776 | 144.719 | 64.8% |
| `R5_second_order` | closest focused ratio to 1.0 | 6.594 | 2.249 | 8.386 | 7.162 | 1.272 | **1.009** | 688.738 | 55.5% |
| `T3_posvel_current` | strongest temporal compromise | 5.440 | 2.335 | 5.177 | 3.348 | 1.357 | 1.075 | 417.006 | 57.0% |

`F4` is the primary candidate. `R5` and `T3` are alternatives. GRU models
are omitted because their training telemetry and evaluation behavior did not
meet the usable-model gate.

## Training graph

The graph uses the saved `train.npz` export of the training callback scalars
(`teleop_train` and `teleop_eval`). No TensorBoard event files were captured in
the current checkout, so the saved scalar export is the available training
record.

![Selected-model training telemetry](figures/selected_models_training.png)

## Evaluation graph

Evaluation is shown only with bar graphs. Lower RMSE is better and the
transparency ratio should be near `1.0`.

![Selected-model evaluation bars](figures/selected_models_evaluation_bars.png)

## Data record

The machine-readable [`runs.csv`](runs.csv) remains an audit catalog of the
underlying comparable runs. This README promotes only the selected models.

When raw outputs are present locally, they are under
`../matlab_env_python_replica/policy_gradient_experiments/results/`. Some
executed notebooks retain historical embedded outputs, but those are not used
for current model selection.
