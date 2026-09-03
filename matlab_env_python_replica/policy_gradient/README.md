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

## Current result tables and graphs

The current tracked policy-gradient result set contains 25 variants from four
fair-bias-15 PPO studies. The protocol is 500,000 training steps, 32 test
episodes, one training/evaluation signal, and 25 focused scenarios per variant.
These are single-study results and should be followed by multi-seed validation.
The tables use focused-evaluation aggregates. A good transparency ratio is
close to `1.0`; ratio validity and within-20% coverage must be read alongside
it.

### Reward ablation: R0-R8

![Reward ablation comparison](../../results_index/figures/reward_ablation_summary.png)

![Reward ablation training curves](../../results_index/figures/reward_ablation_training_curves.png)

![Reward ablation group heatmap](../../results_index/figures/reward_ablation_group_heatmap.png)

| ID | Formulation | Track RMSE [mm] | Post-contact [mm] | Transp. RMSE [W] | Ratio | Ratio error RMSE | Valid | Within +/-20% | RMS u [V] | Mean \|du\| [V] | Mean \|d2u\| [V] | Failure |
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

### Physics-informed formulations: F0-F6

The physics-informed bars and learning curves use overall test aggregate
fields; this table uses the focused battery for cross-family comparison.

![Physics-informed comparison](../../results_index/figures/physics_informed_summary_bars.png)

![Physics-informed learning curves](../../results_index/figures/physics_informed_learning_curves.png)

![Physics-informed transparency ratio rollouts](../../results_index/figures/physics_informed_transparency_ratio_rollouts.png)

| ID | Formulation | Track RMSE [mm] | Post-contact [mm] | Transp. RMSE [W] | Ratio | Ratio error RMSE | Valid | Within +/-20% | RMS u [V] | Mean \|du\| [V] | Mean \|d2u\| [V] | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 | Baseline | 4.543 | 2.880 | 1.389 | 0.621 | 88.307 | 64.4% | 8.0% | 0.762 | 0.780 | 1.517 | 0.0% |
| F1 | Add Error | 4.667 | 3.198 | 1.435 | 0.455 | 1136.514 | 64.3% | 6.9% | 0.974 | 1.013 | 1.952 | 0.0% |
| F2 | Add Error Dot | 4.907 | 3.359 | 1.326 | 0.322 | 240.544 | 67.9% | 5.5% | 1.318 | 1.487 | 2.873 | 0.0% |
| F3 | Add Error DDot | 4.693 | 2.820 | 1.277 | 1.190 | 950.544 | 56.2% | 10.6% | 0.264 | 0.077 | 0.130 | 0.0% |
| F4 | Accel State | **3.535** | 2.411 | 1.348 | 0.776 | 144.719 | 64.8% | 9.6% | 0.922 | 1.099 | 2.090 | 0.0% |
| F5 | Accel State + Reward | 15.893 | 14.588 | 1.618 | 0.595 | 1946.017 | 57.0% | 13.0% | 0.466 | 0.075 | 0.084 | 0.0% |
| F6 | Effort + Delta U | 5.002 | 3.288 | 1.330 | **0.917** | 683.580 | 62.7% | 8.4% | 0.336 | 0.211 | 0.380 | 0.0% |

### Temporal observations: T0-T4

The temporal bars and learning curves use overall test aggregate fields; this
table uses the focused battery for cross-family comparison.

![Temporal observation comparison](../../results_index/figures/temporal_summary_bars.png)

![Temporal observation learning curves](../../results_index/figures/temporal_learning_curves.png)

| ID | Formulation | Track RMSE [mm] | Post-contact [mm] | Transp. RMSE [W] | Ratio | Ratio error RMSE | Valid | Within +/-20% | RMS u [V] | Mean \|du\| [V] | Mean \|d2u\| [V] | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | Position Current | 6.487 | 5.028 | 1.664 | **1.045** | 18.305 | 55.4% | 6.6% | 0.302 | 0.103 | 0.183 | 0.0% |
| T1 | Position Stack 3 | 6.488 | 4.806 | 1.412 | **1.045** | 20.668 | 55.7% | 10.9% | 0.243 | 0.033 | 0.034 | 0.0% |
| T2 | Position Stack 5 | 8.071 | 6.521 | 1.873 | 0.003 | 77.438 | 77.2% | 3.7% | 4.831 | 9.606 | 19.217 | 0.0% |
| T3 | Position Velocity Current | 5.177 | 3.348 | 1.357 | 1.075 | 417.006 | 57.0% | 10.4% | 0.404 | 0.341 | 0.666 | 0.0% |
| T4 | Position Velocity Stack 3 | 4.967 | 2.898 | 1.393 | 0.505 | 319.730 | 64.2% | 6.1% | 1.194 | 1.324 | 2.576 | 0.0% |

### Auxiliary GRU-PPO: G0-G3

![Auxiliary GRU comparison](../../results_index/figures/gru_auxiliary_summary.png)

| ID | Formulation | Track RMSE [mm] | Post-contact [mm] | Transp. RMSE [W] | Ratio | Ratio error RMSE | Valid | Within +/-20% | RMS u [V] | Mean \|du\| [V] | Mean \|d2u\| [V] | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | GRU-PPO | 76.529 | 180.343 | 1.007 | 0.012 | 4.824 | 9.5% | 0.8% | 0.122 | 0.000 | 0.000 | 100.0% |
| G1 | GRU + prediction | 85.159 | 102.897 | 0.699 | 0.366 | 29.979 | 28.0% | 9.1% | 0.033 | 0.000 | 0.000 | 8.0% |
| G2 | GRU + hidden state | 123.245 | 158.799 | 0.647 | 0.290 | 1139.181 | 12.4% | 7.5% | 0.051 | 0.000 | 0.000 | 96.0% |
| G3 | GRU + both auxiliary heads | 118.588 | n/a* | 4.551 | 0.000 | 1.392 | 17.9% | 6.2% | 1.098 | 0.001 | 0.001 | 100.0% |

\* G3 had no valid post-contact segment; the source summary stores this as
zero, so it is represented as `n/a` rather than as a measured zero.

The machine-readable copy of these exact values is
[`../../results_index/runs.csv`](../../results_index/runs.csv).

The notebooks under `notebooks/50_policy_gradient/` call the same reusable
training and evaluation code and render the saved summaries and plots inline.

Generated files remain under `../policy_gradient_experiments/results/` for
compatibility; the old directory is data storage only. Tracked summary figures
are copied to `../../results_index/figures/` so documentation does not depend
on ignored local output trees.
