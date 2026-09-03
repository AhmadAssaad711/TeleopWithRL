# Ablations

`30_reward_state_ablation.ipynb` compares controlled reward and observation
variants. The executable implementation is in
`matlab_env_python_replica/policy_gradient/`; the current tracked result
snapshot is the fair-bias-15, 500,000-step study
`physics_reward_ablation_basic_obs_04_fair_bias15_500k`.

## Current reward results

The table below uses the 25-scenario focused evaluation. Tracking and
post-contact RMSE are in millimetres; the ratio should be near `1.0`.

| ID | Formulation | Track [mm] | Post-contact [mm] | Transp. [W] | Ratio | Ratio error | Valid | Within ±20% | RMS u [V] | Mean \|Δu\| [V] | Mean \|Δ²u\| [V] | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | e only | 7.333 | 6.527 | 1.677 | 0.044 | 241.511 | 66.1% | 3.6% | 3.576 | 6.490 | 12.966 | 0.0% |
| R1 | e + edot | 9.659 | 8.867 | 1.235 | 1.227 | 366.462 | 56.9% | 8.1% | 0.268 | 0.076 | 0.131 | 0.0% |
| R2 | Sliding | 9.276 | 8.334 | 1.198 | 1.226 | 669.090 | 56.8% | 8.6% | 0.274 | 0.109 | 0.200 | 0.0% |
| R3 | Sliding + du | 8.664 | 7.670 | 1.236 | 1.251 | 256.399 | 56.7% | 8.4% | 0.255 | 0.030 | 0.027 | 0.0% |
| R4 | Sliding + du + ddu | 8.718 | 7.691 | 1.232 | 1.261 | 356.875 | 56.7% | 8.2% | 0.248 | 0.027 | 0.020 | 0.0% |
| R5 | Second order | 8.386 | 7.162 | 1.272 | **1.009** | 688.738 | 55.5% | 8.2% | 0.239 | 0.025 | 0.015 | 0.0% |
| R6 | Lyapunov | 13.865 | 13.203 | 1.174 | 0.579 | 28.710 | 52.7% | 2.1% | 0.137 | 0.024 | 0.032 | 0.0% |
| R7 | Phase + direction | 12.093 | 11.262 | 1.156 | 0.593 | 184.424 | 53.1% | 1.7% | 0.148 | 0.029 | 0.040 | 0.0% |
| R8 | HF + deadzone | 11.952 | 11.454 | 1.189 | 1.099 | 71.228 | 53.4% | 1.8% | 0.158 | 0.031 | 0.043 | 0.0% |

`R5_second_order` is the current reward candidate because its ratio statistic
is closest to one. `R0_e_only` has lower tracking error than R5 but fails the
transparency objective.

![Reward ablation comparison](../../results_index/figures/reward_ablation_summary.png)

![Reward ablation training curves](../../results_index/figures/reward_ablation_training_curves.png)

![Reward ablation group heatmap](../../results_index/figures/reward_ablation_group_heatmap.png)

This README contains the exact R0-R8 results table and all three ablation
graphs. The machine-readable rows are in
[`results_index/runs.csv`](../../results_index/runs.csv).
