# Python Replica Environment

This folder groups the Python work that mirrors the Simulink/MATLAB plant.

Important pieces:

- `simuoriginal_replica.py`: standalone nonlinear Python replica of `SimuOriginal.slx`
- `simuoriginal_env.py`: Gym-style wrapper for the replica plant
- `dqn_experiments/`, `ql_experiments/`, `policy_gradient_experiments/`:
  notebook-called algorithm launchers and study scripts
- `studies/`: reusable study helpers shared by the MATLAB-literal experiments
- `scripts/run_replica_studies.py` and `scripts/_common.py`: shared runner
  utilities used by the experiment launchers

Generated outputs are local artifacts and are ignored by Git. Use
`../results_index/` and `../notebooks/90_results/` to organize and review them.

Notebook-first workspace:

- `../notebooks/`: top-level Jupyter workspace; this is the most important
  project-facing folder
- `../results_index/`: normalization target for indexed experiment artifacts
