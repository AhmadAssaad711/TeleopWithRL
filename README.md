# TeleopWithRL Layout

The repo now has two working areas:

- `python_env/`: operational scripts and results for the active `TeleopEnv`-based Python environment
- `matlab_literal_env/`: scripts and results for the SimuOriginal/Simulink-literal parity work

Core implementation still lives at the package root:

- `teleop_env.py`: nonlinear pneumatic environment used by RL and MRAC runs
- `config.py`: plant, reward, and experiment constants
- `mrac_controller.py`, `dqn_agent.py`, `q_learning_agent.py`: controllers and agents
- `train_dqn.py`, `train_q_learning.py`: training implementations
- `benchmark_agents.py`: standard non-ablated DQN/Q-learning/MRAC comparison flow

Organized entry points:

- `python_env/scripts/run_mrac.py`
- `python_env/scripts/train_q_learning.py`
- `python_env/scripts/train_dqn.py`
- `python_env/scripts/run_benchmark.py`
- `matlab_literal_env/scripts/run_simuoriginal_replica.py`

Generated outputs live under:

- `python_env/results/`
- `matlab_literal_env/results/`

## Current SimuOriginal Parity Study

The active MATLAB-parity work is centered on the nonlinear `SimuOriginal.slx`
plant, using `ParmsOriginal.m` only as the parameter source.

Important source files:

- `matlab/reference/SimuOriginal.slx`
- `matlab/reference/ParmsOriginal.m`
- `matlab_literal_env/simuoriginal_replica.py`
- `matlab_literal_env/scripts/run_simuoriginal_replica.py`

Important result folders:

- `matlab_literal_env/results/simuoriginal_replica_saved_open_loop`
- `matlab_literal_env/results/simuoriginal_replica_amp5_bias5`
- `matlab_literal_env/results/matlab_vs_simuoriginal_amp5_bias5_compare_30s`

Current findings:

- The standalone Python replica reproduces the saved open-loop singularity at about `33.793 s`, which is consistent with the MATLAB note that the true open loop blows up around `34 s`.
- With the reduced input `F_h(t) = 5 + 5 sin(0.5 t)`, the standalone replica stays bounded through `40 s`.
- Against the MATLAB-side exported `gui_*` signals over `30 s`, the new replica tracks strongly in sign and shape:
  - `x_m` correlation `0.99895`
  - `x_s` correlation `0.99863`
  - `Fe` correlation `0.99751`

The main remaining parity work is to bring `teleop_env.py` in line with the
standalone `matlab_literal_env/simuoriginal_replica.py` dynamics.
