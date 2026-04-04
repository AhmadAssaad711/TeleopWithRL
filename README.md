# TeleopWithRL Layout

Core model and agent code lives at the package root:

- `teleop_env.py`: nonlinear pneumatic environment
- `config.py`: plant, reward, and experiment constants
- `mrac_controller.py`, `dqn_agent.py`, `q_learning_agent.py`: controllers and agents
- `train_dqn.py`, `train_q_learning.py`: primary training entry points

Runner-style scripts are grouped under `experiments/`:

- `open_loop_io.py`: open-loop plant I/O analysis
- `paper_replica.py`: paper-style MRAC replay
- `benchmark.py`: benchmark runner
- `long_study.py`: long reward-ablation study
- `state_ablation.py`: state-ablation study
- `saved_dqn_eval.py`: saved-model evaluation utilities

Compatibility wrappers remain at the old top-level paths so existing commands still work.

Tests live in `tests/`, and `results_fh/` is treated as generated output rather than source.
