# Environment

This package is the physics-facing boundary for the Python replica.

- `simuoriginal_replica.py` contains the nonlinear 12-state plant, saved
  SimuOriginal parameters/profile, RK4 integration, derived signals, and CSV
  result export.
- `simuoriginal_env.py` wraps that plant as a Gymnasium environment for RL.

The main class is `SimuOriginalReplicaEnv`:

```python
from matlab_env_python_replica.environment import SimuOriginalReplicaEnv

env = SimuOriginalReplicaEnv(env_mode="changing_skin_fat")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step_voltage(0.0)
```

The observation is a normalized 10-element `float32` vector. `reset` returns
`(observation, info)` and `step`/`step_voltage` return the standard Gymnasium
5-tuple. See the module docstring in `simuoriginal_env.py` for the exact
observation order, accepted action forms, reward terms, and termination rules.
