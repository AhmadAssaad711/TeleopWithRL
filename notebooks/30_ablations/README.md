# Ablations

`30_reward_state_ablation.ipynb` compares reward and observation variants. The
README records only the selected reward model rather than every ablation row.

## Selected reward model: R5

`R5_second_order` is retained because its focused-evaluation transparency ratio
is closest to the target value of `1.0`.

| Model | Training endpoint track RMSE [mm] | Training endpoint transp. RMSE [W] | Focused eval track [mm] | Focused eval post-contact [mm] | Focused eval transp. [W] | Ratio | Ratio error RMSE | Valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `R5_second_order` | 6.594 | 2.249 | 8.386 | 7.162 | 1.272 | **1.009** | 688.738 | 55.5% |

## Training and evaluation graphs

Training is represented by the saved callback telemetry export. Evaluation is
represented only by the selected-model bar graph.

![Selected-model training telemetry](../../results_index/figures/selected_models_training.png)

![Selected-model evaluation bars](../../results_index/figures/selected_models_evaluation_bars.png)

The reward ablation is not a multi-seed result. The ratio statistic must be
read with ratio error and validity because velocity-denominator singularities
can make a ratio appear deceptively close to one.
