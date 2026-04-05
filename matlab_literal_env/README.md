# MATLAB-Literal Area

This folder groups the Python work that mirrors the Simulink/MATLAB plant as
literally as possible.

Important pieces:

- `simuoriginal_replica.py`: standalone nonlinear Python replica of `SimuOriginal.slx`
- `simuoriginal_env.py`: Gym-style wrapper that lets MRAC, Q-learning, and DQN run on the replica plant
- `scripts/run_simuoriginal_replica.py`: runner for exporting replica outputs
- `scripts/run_mrac.py`: MRAC on the replica env
- `scripts/train_q_learning.py`: Q-learning on the replica env
- `scripts/train_dqn.py`: DQN on the replica env
- `scripts/run_all_agents.py`: sequential runner for MRAC, Q-learning, and DQN on the replica env
- `results/`: MATLAB-literal outputs and side-by-side comparison artifacts
