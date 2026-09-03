# Policy-gradient studies

This section contains baseline, formulation, and results notebooks for PPO,
TD3, SAC, and recurrent policy-gradient studies. The executable entry points
are under `matlab_env_python_replica/policy_gradient/scripts/`; notebooks
configure runs and present their outputs.

The current comparable result set is the fair-bias-15 PPO protocol: 500,000
training steps, 32 test episodes, one train/evaluation signal, and 25 focused
scenarios per variant.

## Physics-informed formulations

| ID | Formulation | Track RMSE [mm] | Post-contact [mm] | Transp. RMSE [W] | Ratio | Ratio error RMSE | Valid | Within ±20% | RMS u [V] | Mean \|Δu\| [V] | Mean \|Δ²u\| [V] | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 | Baseline | 4.543 | 2.880 | 1.389 | 0.621 | 88.307 | 64.4% | 8.0% | 0.762 | 0.780 | 1.517 | 0.0% |
| F1 | Add Error | 4.667 | 3.198 | 1.435 | 0.455 | 1136.514 | 64.3% | 6.9% | 0.974 | 1.013 | 1.952 | 0.0% |
| F2 | Add Error Dot | 4.907 | 3.359 | 1.326 | 0.322 | 240.544 | 67.9% | 5.5% | 1.318 | 1.487 | 2.873 | 0.0% |
| F3 | Add Error DDot | 4.693 | 2.820 | 1.277 | 1.190 | 950.544 | 56.2% | 10.6% | 0.264 | 0.077 | 0.130 | 0.0% |
| F4 | Accel State | **3.535** | 2.411 | 1.348 | 0.776 | 144.719 | 64.8% | 9.6% | 0.922 | 1.099 | 2.090 | 0.0% |
| F5 | Accel State + Reward | 15.893 | 14.588 | 1.618 | 0.595 | 1946.017 | 57.0% | 13.0% | 0.466 | 0.075 | 0.084 | 0.0% |
| F6 | Effort + Delta U | 5.002 | 3.288 | 1.330 | **0.917** | 683.580 | 62.7% | 8.4% | 0.336 | 0.211 | 0.380 | 0.0% |

The physics-informed bars and learning curves are the study-generated overall
test aggregates; the tables above use focused-evaluation aggregates.

![Physics-informed comparison](../../results_index/figures/physics_informed_summary_bars.png)

![Physics-informed learning curves](../../results_index/figures/physics_informed_learning_curves.png)

![Physics-informed transparency ratio rollouts](../../results_index/figures/physics_informed_transparency_ratio_rollouts.png)

## Temporal observations

| ID | Formulation | Track RMSE [mm] | Post-contact [mm] | Transp. RMSE [W] | Ratio | Ratio error RMSE | Valid | Within ±20% | RMS u [V] | Mean \|Δu\| [V] | Mean \|Δ²u\| [V] | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | Position Current | 6.487 | 5.028 | 1.664 | **1.045** | 18.305 | 55.4% | 6.6% | 0.302 | 0.103 | 0.183 | 0.0% |
| T1 | Position Stack 3 | 6.488 | 4.806 | 1.412 | **1.045** | 20.668 | 55.7% | 10.9% | 0.243 | 0.033 | 0.034 | 0.0% |
| T2 | Position Stack 5 | 8.071 | 6.521 | 1.873 | 0.003 | 77.438 | 77.2% | 3.7% | 4.831 | 9.606 | 19.217 | 0.0% |
| T3 | Position Velocity Current | 5.177 | 3.348 | 1.357 | 1.075 | 417.006 | 57.0% | 10.4% | 0.404 | 0.341 | 0.666 | 0.0% |
| T4 | Position Velocity Stack 3 | 4.967 | 2.898 | 1.393 | 0.505 | 319.730 | 64.2% | 6.1% | 1.194 | 1.324 | 2.576 | 0.0% |

The temporal bars and learning curves are the study-generated overall test
aggregates; the tables above use focused-evaluation aggregates.

![Temporal observation comparison](../../results_index/figures/temporal_summary_bars.png)

![Temporal observation learning curves](../../results_index/figures/temporal_learning_curves.png)

## Auxiliary GRU-PPO

| ID | Formulation | Track RMSE [mm] | Post-contact [mm] | Transp. RMSE [W] | Ratio | Ratio error RMSE | Valid | Within ±20% | RMS u [V] | Mean \|Δu\| [V] | Mean \|Δ²u\| [V] | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | GRU-PPO | 76.529 | 180.343 | 1.007 | 0.012 | 4.824 | 9.5% | 0.8% | 0.122 | 0.000 | 0.000 | 100.0% |
| G1 | GRU + prediction | 85.159 | 102.897 | 0.699 | 0.366 | 29.979 | 28.0% | 9.1% | 0.033 | 0.000 | 0.000 | 8.0% |
| G2 | GRU + hidden state | 123.245 | 158.799 | 0.647 | 0.290 | 1139.181 | 12.4% | 7.5% | 0.051 | 0.000 | 0.000 | 96.0% |
| G3 | GRU + both auxiliary heads | 118.588 | n/a* | 4.551 | 0.000 | 1.392 | 17.9% | 6.2% | 1.098 | 0.001 | 0.001 | 100.0% |

![Auxiliary GRU comparison](../../results_index/figures/gru_auxiliary_summary.png)

\* G3 has no valid post-contact segment; the source summary stores the value
as zero, but it is not treated as a measured zero here.

The tables above are the exact focused-evaluation values for the current
physics-informed, temporal, and auxiliary-GRU studies. The reward ablation
table is reproduced in [`../30_ablations/README.md`](../30_ablations/README.md).

## Notebooks

- `51_ppo_continuous_baseline.ipynb`
- `52_td3_baseline.ipynb`
- `53_sac_baseline.ipynb`
- `54_ppo_discrete_baseline.ipynb`
- `55_physics_informed_formulations_results.ipynb`

The reward ablation table is maintained with the ablation notebook in
[`../30_ablations/README.md`](../30_ablations/README.md).
