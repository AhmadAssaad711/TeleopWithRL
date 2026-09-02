# Policy-Gradient Formulation

This note translates the current teleoperation control task into a policy-gradient RL problem using the code and notebook conventions already present in the repo.

## 1. What the Current Repo Already Tells Us

From the active code paths and notebook studies:

- The physically meaningful setup is now `master_input_mode="force"`, not reference tracking.
- The native control input is continuous valve voltage `u_v in [-5, 5]`.
- The current value-based baselines (`Q-learning`, `DQN`) discretize that action into `cfg.V_LEVELS`, which is convenient but not natural for the plant.
- The notebook work in `20_baselines/`, `30_ablations/`, and `40_generalization/` is centered on `TeleopWithRL.matlab_env_python_replica`, especially `SimuOriginalReplicaEnv`.
- Existing DQN ablations suggest that adding direct force cues helps a lot. In the saved `dyn` summaries, `S6_full10_plus_forces` is the strongest non-curriculum baseline among the static comparisons we inspected.

Two quick local rollout checks also matter:

- On the package-root `TeleopEnv`, zero-voltage control already outperforms random control by a large margin.
- On the SimuOriginal replica, both zero and random control can hit stroke limits, but zero control is still much better than random.

Interpretation:

- this is not a pure stabilization problem where the policy must fight a chaotic open loop from scratch
- it is closer to a precision continuous-control problem where the policy should learn smooth corrections on top of a passive pneumatic coupling baseline

## 2. Recommended Training Problem

For the first policy-gradient implementation, use the same experiment stack as the notebooks:

- Base environment:
  `TeleopWithRL.matlab_env_python_replica.simuoriginal_env.SimuOriginalReplicaEnv`
- Reward wrapper:
  `TeleopWithRL.matlab_env_python_replica.studies.rewarding.ReplicaRewardEnv`
- Observation wrapper:
  `TeleopWithRL.matlab_env_python_replica.studies.dqn_state_variants.ReplicaStateVariantEnv`

Initial environment configuration:

- `env_mode = cfg.ENV_MODE_CHANGING`
- `master_input_mode = cfg.MASTER_INPUT_FORCE`
- `terminate_on_error = True`
- same reset options used by the good notebook baselines unless we intentionally change the scenario

Initial observation choice:

- Recommended: `S6_full10_plus_forces`

Why `S6_full10_plus_forces` first:

- it preserves the full plant observation already used successfully by DQN
- it adds `F_h` and `F_e`, which are highly relevant for transparency control
- it is closer to a Markov state than the older force-hidden observation variants

Initial reward choice:

- Recommended first pass: `baseline_cfg`

Why keep the baseline reward first:

- it is already used throughout the notebook comparisons
- it lets us compare algorithm class rather than changing algorithm and reward at the same time
- once the policy-gradient baseline works, we can run the same reward ablations already used for DQN

## 3. MDP Formulation

### Latent state

The true physical state is continuous and includes at least:

- piston positions and velocities for master and slave
- chamber pressures
- tube mass-flow rates
- valve spool position and velocity
- current environment regime (`skin` or `fat`)
- exogenous force-input phase / instantaneous `F_h`

In code terms, the plant state is essentially the 12-D internal state plus exogenous context.

### Observation

For the recommended first version, let the policy observe:

- the 10-D normalized replica observation
- plus normalized `F_h`
- plus normalized `F_e`

That corresponds to `S6_full10_plus_forces`.

### Action

The action is the valve command:

- `a_t = u_v(t) in [-5, 5]`

This should be treated as continuous in the policy-gradient version.

### Transition

The transition is given by the nonlinear pneumatic dynamics plus the switched environment profile:

- `x_{t+1} ~ p(x_{t+1} | x_t, u_t, F_h(t), env(t))`

Operationally this is deterministic inside the simulator once the reset seed and input profile are fixed.

### Reward

With `baseline_cfg`, the per-step reward is:

`r_t = -(tracking_term + transparency_term + effort_term)`

where

- `tracking_term = alpha * ((x_m - x_s) / s_x)^2`
- `transparency_term = beta * ((F_e * v_m - F_h * v_s) / s_p)^2`
- `effort_term = gamma * u_t^2`

and the wrapper can add terminal penalties for stroke-limit, invalid-state, or tracking-fail events.

### Episode end

An episode ends when one of these happens:

- time horizon reached
- tracking error fail threshold exceeded
- stroke limit or invalid state termination in the replica env

## 4. Why This Is a Better Fit for Policy Gradients Than DQN

The environment already exposes a continuous control variable, but the current value-based baselines force it into 11 bins. Policy methods let us:

- produce smooth voltage corrections instead of coarse action jumps
- learn a stochastic exploration policy directly in voltage space
- avoid learning a Q-value over an arbitrary discretization of a naturally continuous actuator
- model residual correction behavior around the passive baseline more naturally

This is especially important here because:

- the passive plant coupling already does part of the job
- excessive action tends to hurt performance and can drive the system into limits
- the control target is not "pick one of a few modes", it is "shape a continuous signal"

## 5. Policy-Gradient Objective

For a stochastic policy `pi_theta(a | o)`, define

`J(theta) = E_{tau ~ pi_theta}[sum_t gamma^t r_t]`

The policy-gradient identity gives

`grad J(theta) = E[sum_t grad log pi_theta(a_t | o_t) * G_t]`

For practical training, we should not start with raw REINFORCE returns. A better first implementation is actor-critic:

- actor learns `pi_theta(a | o)`
- critic learns `V_phi(o)`
- advantages `A_t` replace raw returns in the policy update

Recommended practical objective:

- clipped PPO-style objective or a simple A2C objective with generalized advantage estimation

## 6. Continuous Policy Parameterization

Recommended first policy:

- Gaussian actor with diagonal standard deviation
- mean produced by a neural net from the observation
- action squashed to `[-5, 5]`

A simple parameterization is:

- network outputs `mu_theta(o)` and `log_sigma_theta`
- sample `z_t ~ N(mu_theta(o_t), sigma_theta(o_t)^2)`
- map to voltage with `u_t = 5 * tanh(z_t)`

This is better than clipping a raw Gaussian because the support is naturally bounded.

Critic:

- separate or shared-backbone value network `V_phi(o_t)`

## 7. Important Markov / POMDP Note

The policy should not be asked to solve a hidden-context problem by accident.

Without force cues, the agent does not directly observe the exogenous master input. Without environment cues, the skin/fat switch is also partly hidden. That makes the task partially observed from the policy's point of view.

Using `S6_full10_plus_forces` helps because:

- `F_h` exposes the external forcing
- `F_e` exposes the current environment reaction
- positions, velocities, pressures, and flows expose the plant state

If we still see instability or poor generalization, the next observation upgrade should be one of:

- add normalized time-in-episode
- add `env_id`
- add previous action `u_{t-1}`

For a smooth-control policy, adding previous action is especially reasonable.

## 8. First Algorithm Choice

There are two sensible first implementations.

### Option A: Continuous PPO-style actor-critic

Recommended.

Pros:

- matches the true action space
- should produce smoother control
- strong default choice for bounded continuous control

Cons:

- more implementation work than a minimal REINFORCE baseline
- on-policy sample use is less efficient than off-policy methods

### Option B: Discrete categorical policy gradient over `cfg.V_LEVELS`

Useful as a sanity-check baseline.

Pros:

- easy apples-to-apples comparison against DQN
- can reuse the existing discrete action tables and policy evaluation tooling almost directly

Cons:

- keeps the same discretization bottleneck we are trying to move beyond
- less compelling as the long-term controller

Recommendation:

- build the continuous actor-critic first
- keep a discrete categorical policy-gradient baseline as a fallback or debugging checkpoint if needed

## 9. Initial Implementation Proposal

Phase 1:

- add a policy-gradient training path under `TeleopWithRL/matlab_env_python_replica/studies/`
- reuse `SimuOriginalReplicaEnv`, `ReplicaRewardEnv`, and `ReplicaStateVariantEnv`
- train with `S6_full10_plus_forces` and `baseline_cfg`
- save outputs in the same `m/`, `l/`, `p/`, `e/` structure as existing DQN/QL studies

Phase 2:

- compare against the saved DQN baseline on the same scenario
- run the same reward ablations
- test waveform generalization using the existing `40_generalization` workflow pattern

## 10. Minimal Spec for the First Policy Agent

Observation:

- `o_t in R^12` using `S6_full10_plus_forces`

Action:

- continuous scalar voltage `u_t in [-5, 5]`

Actor:

- MLP, for example `12 -> 128 -> 128 -> mean`
- trainable log-std or state-dependent std

Critic:

- MLP, for example `12 -> 128 -> 128 -> value`

Rollout data:

- `obs, action, reward, done, value, log_prob`

Training targets:

- returns and advantages

Primary evaluation metrics:

- episode return
- tracking RMSE
- transparency RMSE
- completed episode rate
- stroke-limit episode rate
- mean absolute voltage

## 11. Working Assumptions for the Next Step

Unless we intentionally change direction, the next implementation step should assume:

- environment: `SimuOriginalReplicaEnv`
- action space: continuous
- observation: `S6_full10_plus_forces`
- reward: `baseline_cfg`
- algorithm family: PPO-style actor-critic

That gives us a clean first policy-gradient baseline without changing the plant, the notebook workflow, and the evaluation metrics all at once.
