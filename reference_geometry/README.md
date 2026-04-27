# Reference Geometry

This folder holds the reference parameter and geometry sources for the
teleoperation plant.

Current contents:

- `ParmsOriginal.m`
  - MATLAB parameter script from the original SimuOriginal setup.
  - Defines cylinder, valve, pressure, mass, and damping constants used by the
    Simulink model.
  - The Python constants in `../config.py` mirror these values where the RL
    environments need them.

Keep this folder for plant constants, geometry definitions, reference input
definitions, and small metadata files. Generated experiment results belong in
local result folders and should be indexed through `../results_index/` instead.
