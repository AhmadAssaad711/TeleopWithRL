# Python Replica Environment

This folder contains the Python implementation that mirrors the
SimuOriginal MATLAB/Simulink plant and the reusable RL code built on top of it.
The source is organized by responsibility so notebooks can configure runs and
display results without carrying the training implementation themselves.

## Source layout

```text
matlab_env_python_replica/
├── environment/
│   ├── simuoriginal_replica.py   # nonlinear plant and fixed-step integration
│   └── simuoriginal_env.py       # Gymnasium reset/step wrapper
├── config/
│   └── config.py                 # shared physical, RL, and normalization constants
├── common/
│   ├── cli.py                    # argument-to-environment normalization
│   ├── runner.py                 # study orchestration
│   ├── rewarding.py              # reward variants and reward wrapper
│   ├── study_utils.py            # metrics, files, plots, and result helpers
│   ├── focused_evaluation.py     # reusable evaluation batteries
│   └── saved_policy_eval.py      # evaluation of saved policies
├── dqn/
│   ├── agent.py
│   ├── state_variants.py
│   ├── training.py
│   └── scripts/                  # notebook/CLI entry points
├── ql/
│   ├── agent.py
│   ├── state_variants.py
│   ├── training.py
│   └── scripts/                  # notebook/CLI entry points
└── policy_gradient/
    ├── paths.py
    ├── training.py
    └── scripts/                  # PPO, TD3, SAC, and auxiliary studies
```

Every Python source file has a module description, and the environment module
documents its inputs, outputs, observation order, timing, and termination
behavior in detail.

Detailed contracts are collected in:

- [`API.md`](API.md): environment, plant, agents, training, rewards, and
  evaluation interfaces.
- [`CLI.md`](CLI.md): every executable launcher, option groups, and output
  contract.

## Notebook-to-code contract

The notebooks are the presentation and analysis layer. A training notebook:

1. builds a configuration and command;
2. calls a script entry point through `notebooks._teleop_nb`;
3. reads the generated manifests, summaries, histories, and plots.

For example:

```powershell
python -m TeleopWithRL.matlab_env_python_replica.dqn.scripts.run_experiments
python -m TeleopWithRL.matlab_env_python_replica.ql.scripts.run_experiments
python -m TeleopWithRL.matlab_env_python_replica.policy_gradient.scripts.run_experiments
```

Those fully qualified commands are the notebook form and should be run from
the repository's parent directory. From the repository root, use the shorter
form documented in [`CLI.md`](CLI.md), for example:

```powershell
python -m matlab_env_python_replica.dqn.scripts.run_experiments --help
```

The functions used by those entry points live in `training.py`, `agent.py`,
the state-variant modules, and `common/`; the notebooks do not need to repeat
those functions.

## Generated results

Generated outputs are local artifacts and are ignored by Git. Existing result
locations are intentionally kept stable while source code is reorganized:

- `results/` for shared replica studies
- `dqn_experiments/results/` for DQN artifacts
- `ql_experiments/results/` for Q-learning artifacts
- `policy_gradient_experiments/results/` for policy-gradient artifacts

The old `*_experiments` directories now serve only as compatibility locations
for local results; their source code has moved into the packages above. Use
`../results_index/` and `../notebooks/90_results/` to curate and review outputs.

## Current results

Only the selected models are reported here. `F4_accel_state` is the primary
tracking candidate; `R5_second_order` and `T3_posvel_current` are retained as
reward-design and temporal-observation alternatives.

| Model | Training endpoint track RMSE [mm] | Training endpoint transp. RMSE [W] | Focused eval track [mm] | Focused eval post-contact [mm] | Focused eval transp. [W] | Ratio | Valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| `F4_accel_state` | **3.377** | **1.566** | **3.535** | 2.411 | 1.348 | 0.776 | 64.8% |
| `R5_second_order` | 6.594 | 2.249 | 8.386 | 7.162 | **1.272** | **1.009** | 55.5% |
| `T3_posvel_current` | 5.440 | 2.335 | 5.177 | 3.348 | 1.357 | 1.075 | 57.0% |

![Selected-model training telemetry](../results_index/figures/selected_models_training.png)

![Selected-model evaluation bars](../results_index/figures/selected_models_evaluation_bars.png)

The training graph uses the saved `train.npz` callback-scalar export because
the current checkout contains no TensorBoard event files. Evaluation is shown
only with bar graphs. The complete model-selection notes are recorded in the
selected results report under `../results_index/`.
