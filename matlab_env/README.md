# MATLAB Environment

This folder is only for the actual MATLAB/Simulink reference environment.

- `SimuOriginal.slx`

Reference parameter and geometry definitions live separately in:

- `../reference_geometry/ParmsOriginal.m`

This `SimuOriginal.slx` copy is the edited version with the control-law path removed at the top level:

- `Plant/F_h` is driven by the model input path
- `Plant/u` is driven by a constant zero input

Current top-level input settings in the saved reference model:

- `Sine Wave` amplitude = `10`
- `Sine Wave` bias = `5`
- `Sine Wave` frequency = `0.5`

Observed behavior:

- the model is stable enough to export bounded signals up to about `30 s`
- in true open loop, it develops the same pressure-dynamics singularity around `34 s`

Historical duplicate `.slx` files and Simulink cache artifacts were removed so
this folder stays focused on the single reference model.
