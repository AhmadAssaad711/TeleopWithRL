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
