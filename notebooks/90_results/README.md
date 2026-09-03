# Results catalog

This README is the short review surface for the selected current results. It
contains the exact selected-model table and the two graphs used for reporting.
The companion `90_results_catalog.ipynb` displays the same tracked assets.

## Selected models

| Model | Selection reason | Training endpoint track RMSE [mm] | Training endpoint transp. RMSE [W] | Focused eval track [mm] | Focused eval post-contact [mm] | Focused eval transp. [W] | Ratio | Ratio error RMSE | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `F4_accel_state` | best training and focused tracking | 3.377 | 1.566 | **3.535** | 2.411 | 1.348 | 0.776 | 144.719 | 64.8% |
| `R5_second_order` | closest focused ratio to 1.0 | 6.594 | 2.249 | 8.386 | 7.162 | 1.272 | **1.009** | 688.738 | 55.5% |
| `T3_posvel_current` | strongest temporal compromise | 5.440 | 2.335 | 5.177 | 3.348 | 1.357 | 1.075 | 417.006 | 57.0% |

## Training graph

The training graph uses the saved `train.npz` export of the callback scalars
(`teleop_train` and `teleop_eval`). No TensorBoard event files were captured in
the current checkout, so the saved scalar export is the available training
record.

![Selected-model training telemetry](../../results_index/figures/selected_models_training.png)

## Evaluation graph

Evaluation is shown only with bar graphs. Lower RMSE is better and the
transparency ratio should be near `1.0`.

![Selected-model evaluation bars](../../results_index/figures/selected_models_evaluation_bars.png)

The machine-readable audit catalog remains in
[`../../results_index/runs.csv`](../../results_index/runs.csv); this README
promotes only the selected models.
