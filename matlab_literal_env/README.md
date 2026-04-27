# MATLAB-Literal Environment

This folder groups the Python work that mirrors the Simulink/MATLAB plant as
literally as possible.

Important pieces:

- `simuoriginal_replica.py`: standalone nonlinear Python replica of `SimuOriginal.slx`
- `simuoriginal_env.py`: Gym-style wrapper that lets MRAC, Q-learning, and DQN run on the replica plant
- `scripts/run_simuoriginal_replica.py`: runner for exporting replica outputs
- `scripts/run_mrac.py`: MRAC on the replica env
- `scripts/train_q_learning.py`: Q-learning on the replica env
- `scripts/train_dqn.py`: DQN on the replica env
- `scripts/run_all_agents.py`: sequential runner for open-loop plant (`u=0`), MRAC, Q-learning, and DQN on the replica env
- `dqn_experiments/`, `ql_experiments/`, `policy_gradient_experiments/`:
  algorithm-specific launchers and study scripts
- `studies/`: reusable study helpers shared by the MATLAB-literal experiments

Generated outputs are local artifacts and are ignored by Git. Use
`../results_index/` and `../notebooks/90_results/` to organize and review them.

Notebook-first workspace:

- `../notebooks/`: top-level Jupyter workspace; this is the most important
  project-facing folder
- `../results_index/`: normalization target for indexed experiment artifacts
