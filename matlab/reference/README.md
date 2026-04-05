# MATLAB Reference Models

Current open-loop ground-truth pair for plant debugging:

- `SimuOriginal.slx`
- `ParmsOriginal.m`

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

Other `.slx` files in this folder are retained only as historical reference.
