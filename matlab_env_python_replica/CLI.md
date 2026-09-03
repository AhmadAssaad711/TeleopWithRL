# Command-line interface

Run commands from the repository root with the short module prefix shown below.
The fully qualified `TeleopWithRL...` form used by notebooks is also supported
when the directory containing this repository is on `PYTHONPATH` (the notebook
helper does this automatically):

```powershell
python -m matlab_env_python_replica.<module>
```

Every launcher accepts `--help`. Training commands create local artifacts
(models, histories, summaries, plots, and TensorBoard logs); these output trees
are ignored by Git.

Policy-gradient training additionally requires the optional
`stable-baselines3` dependency. Its `--help` output is available without that
dependency; the dependency is checked only after arguments are parsed.

## Entry points

| Command | Purpose | Main output location |
|---|---|---|
| `common.runner` | Combined Q-learning/DQN staged study | `matlab_env_python_replica/results/...` |
| `dqn.scripts.run_experiments` | DQN baselines, reward/state studies, evaluation | `dqn_experiments/results/...` |
| `dqn.scripts.run_baselines_both_fe` | Run the DQN baseline in both FE modes | `dqn_experiments/results/...` |
| `ql.scripts.run_experiments` | Q-learning baselines, state/reward studies, evaluation | `ql_experiments/results/...` |
| `ql.scripts.run_baselines_both_fe` | Run the Q-learning baseline in both FE modes | `ql_experiments/results/...` |
| `policy_gradient.scripts.run_experiments` | PPO/TD3/SAC/PPO-discrete runs | `policy_gradient_experiments/results/...` |
| `policy_gradient.scripts.run_baselines_both_fe` | Policy-gradient baseline in both FE modes | `policy_gradient_experiments/results/...` |
| `policy_gradient.scripts.run_focused_evaluation` | Evaluate a saved continuous policy on the shared battery | beside the selected model |
| `policy_gradient.scripts.run_physics_informed_formulations` | Compare F0-F6 state/reward formulations | `policy_gradient_experiments/results/...` |
| `policy_gradient.scripts.run_physics_reward_ablation_basic_obs` | Compare reward terms with fixed basic observations | `policy_gradient_experiments/results/...` |
| `policy_gradient.scripts.run_temporal_observation_stack` | Compare temporal observation windows | `policy_gradient_experiments/results/...` |
| `policy_gradient.scripts.run_auxiliary_gru_ppo` | Train/evaluate the auxiliary GRU-PPO study | `policy_gradient_experiments/results/...` |

From the repository's parent directory, the notebook-compatible form is:

```powershell
python -m TeleopWithRL.matlab_env_python_replica.policy_gradient.scripts.run_experiments --help
```

## Common experiment options

The exact defaults are printed by `--help`; the shared options have these
meanings:

- `--study-name`: result-folder/study identifier.
- `--stage`: run all stages or one named stage such as `baselines`, a state or
  reward ablation, or `eval`.
- `--env-mode`: `constant_skin` or `changing_skin_fat`.
- `--fe-mode`: exported environment-force convention, normally
  `gui_skin_locked` or `switched_dynamics`.
- `--episode-duration` and `--env-switch-time`: seconds per episode and the
  skin-to-fat switch time.
- `--force-amp`, `--force-bias`, `--force-freq`, `--force-freq-rad`,
  `--force-phase`, and `--force-waveform`: human-force input settings. Use
  either the frequency in Hz or the explicit angular frequency in rad/s as
  supported by the launcher.
- `--reset-position-mode`: `midpoint` is the RL-safe default; `zero` is the
  legacy origin-centered initialization.
- `--stroke-limit-mode`: `terminate` or `clamp`.
- `--test-episodes`, `--seed`, and `--noise-std`: evaluation count,
  reproducibility seed, and force-noise scale.
- `--skip-existing`: do not rerun a completed stage.
- `--resume`: resume a compatible partial training artifact where supported.
- `--disable-terminate-on-error` and `--disable-stroke-limit`: explicit
  safety-behavior overrides; use only for diagnostic experiments.

## Algorithm-specific options

`dqn.scripts.run_experiments` additionally accepts `--dqn-episodes`,
`--dqn-parallel-envs`, `--reward-variant`, `--full-grid`,
`--parallel-workers`, and `--worker-torch-threads`.

`ql.scripts.run_experiments` additionally accepts `--q-episodes`,
`--reward-variant`, `--parallel-workers`, and `--worker-torch-threads`.

`policy_gradient.scripts.run_experiments` requires `--algo` with one of
`ppo_continuous`, `td3`, `sac`, or `ppo_discrete`. It also accepts
`--state-variant`, `--reward-variant`, optional `--state-spec-json` and
`--reward-spec-json`, train/evaluation reset-option JSON files, and PPO vector
environment settings (`--parallel-envs`, `--vec-env`, `--ppo-n-steps`,
`--ppo-batch-size`, `--ppo-n-epochs`, and `--ppo-device`).

The three formulation-study launchers use additional study-specific settings
such as `--calibration-study`, `--train-episodes`, `--total-timesteps`,
`--eval-every-episodes`, and `--focused-seed`. Run their individual `--help`
commands before a long run because their defaults are intentionally different.

## Examples

Run a short policy-gradient smoke test into an isolated study directory:

```powershell
python -m matlab_env_python_replica.policy_gradient.scripts.run_experiments `
  --algo ppo_continuous `
  --study-name local_check `
  --train-episodes 2 `
  --test-episodes 2 `
  --seed 42
```

Run the reusable focused evaluation against a saved policy directory:

```powershell
python -m matlab_env_python_replica.policy_gradient.scripts.run_focused_evaluation `
  --model-path matlab_env_python_replica/policy_gradient_experiments/results/dyn/<study>/<variant>/ppo `
  --out-dir matlab_env_python_replica/policy_gradient_experiments/results/dyn/<study>/<variant>/focused_eval
```

## Output contract

A completed training run normally contains:

- `m/`: saved model (`*.pt`, `*.zip`, or `q_table.npy`);
- `l/`: machine-readable `summary.json` and human-readable summary files;
- `plots/`: learning curves, rollouts, action/state diagnostics;
- `tb/`: TensorBoard event files;
- optional `focused_eval/`: scenario-level metrics and plots.

The later results page should select only completed runs with a manifest,
finite summary metrics, expected scenario counts, and the required diagnostic
plots. Development probes, smoke tests, and partial runs should remain local
only or be removed during curation.
