# Python replica API

This document describes the stable interfaces used by the notebooks and the
experiment launchers. The notebooks are consumers of these interfaces; they
should configure a run, call an entry point, and analyse the returned data
without copying the implementation into a notebook cell.

## Environment boundary

### `environment.simuoriginal_env.SimuOriginalReplicaEnv`

The Gymnasium environment accepts physical and experiment settings in its
constructor and returns normalized observations plus diagnostic metadata.

```python
from matlab_env_python_replica.environment import SimuOriginalReplicaEnv

env = SimuOriginalReplicaEnv(
    env_mode="changing_skin_fat",
    master_input_mode="force",
    episode_duration=60.0,
    env_switch_time=30.0,
)
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step_voltage(0.0)
history = env.render()
```

Constructor inputs include `env_mode`, `episode_duration`, `env_switch_time`,
termination/stroke-limit settings, optional voltage `action_levels`, and
optional `ParmsOriginal`/`SimuOriginalProfile` objects.

- `reset(seed, options)` returns `(observation, info)`.
- `step(action)` accepts a discrete action index, scalar voltage, or a
  one-element array and returns the Gymnasium five-tuple.
- `step_voltage(u_v)` applies a scalar voltage directly and clips it to the
  configured action range.
- `render()` returns the current history dictionary; it does not open a GUI.
- The observation is a normalized `float32` vector. The exact ten-element
  order and scales are documented in `environment/simuoriginal_env.py`.
- `info` contains physical signals, environment labels, termination reason,
  force inputs, and transparency diagnostics.

### `environment.simuoriginal_replica`

This module is the plant-only API. It does not depend on an RL agent.

- `ParmsOriginal` stores pneumatic, mechanical, valve, and solver constants.
- `SimuOriginalProfile` stores the saved SimuOriginal input and environment
  switch settings.
- `SimuOriginalState` stores the twelve-state plant vector and converts to/from
  a NumPy array.
- `SimuOriginalResult` stores time-aligned plant, force, valve, and flow
  histories.
- `build_saved_simuoriginal_state(...)` creates the saved equilibrium state.
- `simuoriginal_derivatives(t, y, ...)` returns the twelve-state derivative.
- `simulate_simuoriginal_replica(...)` integrates with fixed-step RK4 and
  returns `SimuOriginalResult`.
- `write_simuoriginal_result(result, out_dir)` writes CSV and text summary
  files. It creates `out_dir` if needed.

All plant positions are in metres, forces in newtons, pressures in pascals,
time in seconds, and control input in volts unless a field name says
otherwise.

## Configuration and variants

`config/config.py` is the single source for physical constants, RL timing,
normalization scales, action levels, reward weights, and tabular bin edges.
Import it as `from matlab_env_python_replica.config import config as cfg`.

`dqn.state_variants` exposes `DQNStateVariant` objects. Each object provides a
name, feature names, description, optional metadata, an `obs_dim`, and an
`extractor(observation, info) -> np.ndarray` callable. Use:

- `build_dqn_state_variants()` and `get_dqn_state_variant(name)` for named
  variants.
- `available_custom_state_features()` to inspect notebook-selectable features.
- `build_custom_dqn_state_variant(...)` for a programmatic variant.
- `build_custom_dqn_state_variant_from_spec(spec)` or
  `load_custom_dqn_state_variant(path)` for JSON-defined and temporal states.

`ql.state_variants` exposes `QLStateVariant` objects. Each provides a
`discretizer(observation, info) -> tuple[int, ...]`, `state_dims`, and feature
metadata. Use `build_ql_state_variants()` or `get_ql_state_variant(name)`.

## Agents and training

### DQN

`dqn.agent.DQNAgent` consumes a continuous observation vector and selects one
of the configured voltage actions. Its public workflow is:

1. `select_action(observation)` during training;
2. `store_transition(...)` after each transition;
3. `train_step()` when the replay buffer is ready;
4. `save(path)`/`load(path)` for `*.pt` checkpoints;
5. `q_values(observation)` or `greedy_action(observation)` for evaluation.

`dqn.training.train_dqn_variant(...)` returns a `RunResult` and writes model,
history, summary, TensorBoard, and diagnostic plot artifacts under `out_dir`.
`evaluate_dqn(...)` returns `(aggregate_metrics, aggregate_history)` and runs
with exploration disabled.

### Tabular Q-learning

`ql.agent.QLearningAgent` maps a discrete state tuple to a sparse Q-table.
`select_action`, `update`, and `decay_epsilon` implement the training loop;
`q_values`, `common.study_utils.greedy_q_action`, `save`, and `load` support
evaluation and persistence. `ql.training.train_qlearning_variant(...)` and
`evaluate_qlearning(...)` use the same result contract as DQN.

### Policy gradient

`policy_gradient.training` supports `ppo_continuous`, `td3`, `sac`, and
`ppo_discrete`. The main interfaces are:

- `build_policy_gradient_env_factory(...)` creates fresh Gymnasium environments
  for SB3 workers.
- `train_policy_gradient_variant(...)` trains one state/reward/algo
  combination and returns a `RunResult`.
- `evaluate_policy_gradient(...)` evaluates a trained model and returns
  aggregate metrics plus histories.
- `save_policy_gradient_visuals(...)` writes the common diagnostic plots.
- `get_policy_gradient_state_variant(...)` and
  `get_policy_gradient_reward_variant(...)` resolve named or JSON variants.

Training functions may create directories and write checkpoints, histories,
CSV/JSON summaries, plots, and TensorBoard logs. Generated result trees are
local artifacts and are ignored by Git.

## Rewards and evaluation

`common.rewarding` defines the reward contract. A `RewardVariant` names the
variant and its terms; `ReplicaRewardEnv` wraps the base environment and
replaces its reward while preserving the Gymnasium reset/step interface.

- `baseline_reward_variant()`, `build_core_reward_variants()`, and
  `build_full_reward_variants()` return built-in variants.
- `reward_variant_from_spec(spec)` builds a notebook/JSON-style variant.
- `load_reward_variant_from_json(path)` loads one from disk.
- `reward_variant_from_name(name)` resolves a built-in name or an existing JSON
  path.
- `compute_reward_terms(...)` returns the individual scalar reward terms for
  one transition.

`common.focused_evaluation` defines reusable scenario batteries. An
`EvaluationScenario.reset_options()` result can be passed directly to
`env.reset(options=...)`. The main functions are:

- `build_focused_scenarios(summary)` for tracking/contact scenarios;
- `build_bode_scenarios(summary)` for frequency-response tests;
- `evaluate_policy_on_scenario(...)` for one rollout;
- `compute_non_bode_metrics(result)` and `compute_bode_metrics(result)` for
  metric extraction;
- `run_focused_evaluation(...)` for the complete battery and artifact bundle.

`common.saved_policy_eval` provides `evaluate_saved_policy(...)` for DQN and
Q-learning checkpoints and `save_evaluation_bundle(...)` for CSV, JSON, and
plot output.

## Shared result helpers

`common.study_utils` owns the common result contract used by all algorithms:

- `mk_run_dirs(out_dir)` creates `m`, `l`, `plots`, and `tb` directories.
- `rollout_metrics(history, env_switch_time=...)` calculates tracking,
  transparency, control, and termination metrics.
- `aggregate_episode_histories(...)` aligns episode histories for plotting.
- `save_history_npz`, `save_json`, and `write_run_summary` serialize results.
- `save_common_visuals(...)` and the plotting helpers create presentation
  figures from the same histories and metric rows.
- `stage_completed(...)` and `stage_summary_rows_to_csv(...)` support resumable
  staged studies.

Most plotting functions return `None` and write the requested image path;
metric functions return dictionaries or arrays and do not mutate the input
history.

## Notebook contract

Notebook code should follow this sequence:

1. import the public helper or construct a CLI command;
2. use relative repository paths or the package path helpers;
3. execute the reusable script when a study is needed;
4. load `summary.json`, `summary.csv`, `*.npz`, and generated plots;
5. explain the selected metric and unit in a Markdown cell.

Avoid hard-coded user-machine paths and avoid reimplementing training,
environment, reward, or evaluation functions inside notebooks.
