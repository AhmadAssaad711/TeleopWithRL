from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
import os
import time
from typing import Any

import numpy as np

from ... import config as cfg
from ...dqn_agent import DQNAgent
from ..simuoriginal_env import SimuOriginalReplicaEnv
from .common import (
    RunResult,
    aggregate_episode_histories,
    history_with_obs,
    mk_run_dirs,
    q_gap,
    require_tensorboard,
    rollout_metrics,
    save_common_visuals,
    save_history_npz,
    save_training_plot,
    write_run_summary,
    plot_dqn_policy_slices,
)
from .dqn_state_variants import (
    DQNStateVariant,
    ReplicaStateVariantEnv,
    get_dqn_state_variant,
)
from .rewarding import (
    RewardVariant,
    ReplicaRewardEnv,
    reward_variant_from_name,
)


@dataclass(frozen=True)
class _ParallelDQNEnvSpec:
    env_mode: str
    env_kwargs: dict[str, Any]
    reward_variant_name: str
    state_variant_name: str


def _worker_set_process_threads(num_threads: int) -> None:
    threads = max(1, int(num_threads))
    for env_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[env_var] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(threads, 2)))
    except Exception:
        pass


def build_dqn_env_factory(
    *,
    env_mode: str,
    env_kwargs: dict,
    reward_variant: RewardVariant,
    state_variant: DQNStateVariant,
):
    def _factory():
        base_env = SimuOriginalReplicaEnv(
            env_mode=env_mode,
            master_input_mode=cfg.MASTER_INPUT_FORCE,
            **dict(env_kwargs),
        )
        reward_env = ReplicaRewardEnv(base_env, reward_variant)
        return ReplicaStateVariantEnv(reward_env, state_variant)

    return _factory


def _build_env_from_spec(spec: _ParallelDQNEnvSpec):
    env_factory = build_dqn_env_factory(
        env_mode=str(spec.env_mode),
        env_kwargs=dict(spec.env_kwargs),
        reward_variant=reward_variant_from_name(str(spec.reward_variant_name)),
        state_variant=get_dqn_state_variant(str(spec.state_variant_name)),
    )
    return env_factory()


def _parallel_env_worker(conn, spec: _ParallelDQNEnvSpec) -> None:
    _worker_set_process_threads(1)
    env = _build_env_from_spec(spec)
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=int(payload["seed"]), options=dict(payload.get("options") or {}))
                conn.send((obs, info))
                continue
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(int(payload["action"]))
                history = env.render() if (terminated or truncated) else None
                conn.send((obs, reward, terminated, truncated, info, history))
                continue
            if cmd == "close":
                conn.close()
                return
            raise ValueError(f"Unknown worker command: {cmd}")
    except EOFError:
        return
    finally:
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception:
            pass


class _SubprocEnvBatch:
    def __init__(self, spec: _ParallelDQNEnvSpec, n_envs: int):
        self.n_envs = max(1, int(n_envs))
        self._ctx = mp.get_context("spawn")
        self._parents: list[Any] = []
        self._procs: list[Any] = []
        self._closed: set[int] = set()
        for _ in range(self.n_envs):
            parent_conn, child_conn = self._ctx.Pipe()
            proc = self._ctx.Process(target=_parallel_env_worker, args=(child_conn, spec), daemon=True)
            proc.start()
            child_conn.close()
            self._parents.append(parent_conn)
            self._procs.append(proc)

    def reset(self, slot: int, seed: int, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        parent = self._parents[int(slot)]
        parent.send(("reset", {"seed": int(seed), "options": dict(options or {})}))
        obs, info = parent.recv()
        return np.asarray(obs, dtype=np.float32), dict(info)

    def step_many(
        self,
        slots: list[int],
        actions: np.ndarray,
    ) -> list[tuple[np.ndarray, float, bool, bool, dict[str, Any], dict[str, Any] | None]]:
        for slot, action in zip(slots, np.asarray(actions, dtype=np.int64).tolist()):
            self._parents[int(slot)].send(("step", {"action": int(action)}))
        results = []
        for slot in slots:
            obs, reward, terminated, truncated, info, history = self._parents[int(slot)].recv()
            results.append(
                (
                    np.asarray(obs, dtype=np.float32),
                    float(reward),
                    bool(terminated),
                    bool(truncated),
                    dict(info),
                    history,
                )
            )
        return results

    def close_slot(self, slot: int) -> None:
        slot = int(slot)
        if slot in self._closed:
            return
        parent = self._parents[slot]
        proc = self._procs[slot]
        try:
            parent.send(("close", None))
        except Exception:
            pass
        try:
            parent.close()
        except Exception:
            pass
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
        self._closed.add(slot)

    def close(self) -> None:
        for slot in range(self.n_envs):
            self.close_slot(slot)


def _q_gap_batch(q_values_batch: np.ndarray) -> np.ndarray:
    q_values_batch = np.asarray(q_values_batch, dtype=np.float64)
    if q_values_batch.ndim == 1:
        q_values_batch = q_values_batch.reshape(1, -1)
    if q_values_batch.shape[1] <= 1:
        return q_values_batch[:, 0].copy()
    sorted_q = np.sort(q_values_batch, axis=1)
    return sorted_q[:, -1] - sorted_q[:, -2]


def _sample_reset_options(
    pool: list[dict[str, Any]] | None,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    if not pool:
        return None
    idx = int(rng.integers(0, len(pool)))
    return dict(pool[idx])


def _next_reset_options(
    *,
    episode_index: int,
    schedule: list[dict[str, Any]] | None,
    pool: list[dict[str, Any]] | None,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    if schedule and 0 <= int(episode_index) < len(schedule):
        return dict(schedule[int(episode_index)])
    return _sample_reset_options(pool, rng)


def evaluate_dqn(
    agent: DQNAgent,
    env_factory,
    n_episodes: int,
    seed_offset: int,
    parallel_envs: int = 1,
) -> tuple[dict[str, float], dict[str, Any]]:
    if int(parallel_envs) <= 1 or int(n_episodes) <= 1:
        episode_metrics: list[dict[str, float]] = []
        episode_histories: list[dict[str, Any]] = []

        old_eps = float(agent.epsilon)
        agent.epsilon = 0.0
        try:
            for ep in range(n_episodes):
                env = env_factory()
                obs, _ = env.reset(seed=seed_offset + ep)
                done = False
                obs_trace: list[np.ndarray] = []
                while not done:
                    obs_trace.append(np.asarray(obs, dtype=np.float32).copy())
                    action = agent.select_action(obs)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                history = history_with_obs(env.render() or {}, obs_trace)
                metrics = rollout_metrics(history, env_switch_time=float(getattr(env, "env_switch_time", cfg.ENV_SWITCH_TIME)))
                episode_metrics.append(metrics)
                episode_histories.append(history)
        finally:
            agent.epsilon = old_eps

        aggregate = {
            key: float(np.mean([row[key] for row in episode_metrics])) for key in episode_metrics[0].keys()
        }
        aggregate_history = aggregate_episode_histories(episode_histories)
        return aggregate, aggregate_history

    base_env = getattr(env_factory(), "base_env", None)
    reward_env = base_env if isinstance(base_env, ReplicaRewardEnv) else None
    if reward_env is None:
        raise RuntimeError("Parallel DQN evaluation requires a ReplicaRewardEnv-wrapped factory.")
    spec = _ParallelDQNEnvSpec(
        env_mode=str(getattr(reward_env.base_env, "env_mode")),
        env_kwargs=dict(
            episode_duration=float(getattr(reward_env.base_env, "episode_duration")),
            env_switch_time=float(getattr(reward_env.base_env, "env_switch_time")),
            terminate_on_error=bool(getattr(reward_env.base_env, "terminate_on_error")),
            reset_options=dict(getattr(reward_env.base_env, "default_reset_options", {})),
        ),
        reward_variant_name=str(getattr(reward_env, "variant").name),
        state_variant_name=str(getattr(env_factory(), "state_variant").name),
    )
    batch = _SubprocEnvBatch(spec, min(int(parallel_envs), int(n_episodes)))
    old_eps = float(agent.epsilon)
    agent.epsilon = 0.0
    try:
        active_slots: list[int] = []
        current_obs: dict[int, np.ndarray] = {}
        obs_traces: dict[int, list[np.ndarray]] = {}
        episode_metrics: list[dict[str, float]] = []
        episode_histories: list[dict[str, Any]] = []
        started = 0
        completed = 0

        for slot in range(batch.n_envs):
            if started >= int(n_episodes):
                break
            obs, _ = batch.reset(slot, seed_offset + started)
            active_slots.append(slot)
            current_obs[slot] = obs
            obs_traces[slot] = []
            started += 1

        while active_slots:
            obs_batch = np.stack([current_obs[slot] for slot in active_slots], axis=0)
            q_values_batch = agent.q_values_batch(obs_batch)
            actions = np.argmax(q_values_batch, axis=1)
            results = batch.step_many(active_slots, actions)

            for slot, result in zip(list(active_slots), results):
                next_obs, _, terminated, truncated, _, history = result
                obs_traces[slot].append(np.asarray(current_obs[slot], dtype=np.float32).copy())
                current_obs[slot] = next_obs
                if not (terminated or truncated):
                    continue

                hist = history_with_obs(history or {}, obs_traces[slot])
                metrics = rollout_metrics(
                    hist,
                    env_switch_time=float(spec.env_kwargs["env_switch_time"]),
                )
                episode_metrics.append(metrics)
                episode_histories.append(hist)
                completed += 1

                if started < int(n_episodes):
                    obs, _ = batch.reset(slot, seed_offset + started)
                    current_obs[slot] = obs
                    obs_traces[slot] = []
                    started += 1
                else:
                    batch.close_slot(slot)
                    active_slots.remove(slot)
                    current_obs.pop(slot, None)
                    obs_traces.pop(slot, None)

        aggregate = {
            key: float(np.mean([row[key] for row in episode_metrics])) for key in episode_metrics[0].keys()
        }
        aggregate_history = aggregate_episode_histories(episode_histories)
        return aggregate, aggregate_history
    finally:
        agent.epsilon = old_eps
        batch.close()


def _make_parallel_spec(
    *,
    env_mode: str,
    env_kwargs: dict[str, Any],
    reward_variant: RewardVariant,
    state_variant: DQNStateVariant,
) -> _ParallelDQNEnvSpec:
    return _ParallelDQNEnvSpec(
        env_mode=str(env_mode),
        env_kwargs=dict(env_kwargs),
        reward_variant_name=str(reward_variant.name),
        state_variant_name=str(state_variant.name),
    )


def train_dqn_variant(
    *,
    out_dir: str | Path,
    env_mode: str,
    env_kwargs: dict,
    state_variant: DQNStateVariant,
    reward_variant: RewardVariant,
    total_episodes: int,
    test_episodes: int,
    seed: int,
    label: str,
    parallel_envs: int = 1,
    train_reset_options_pool: list[dict[str, Any]] | None = None,
    train_reset_options_schedule: list[dict[str, Any]] | None = None,
    init_model_path: str | Path | None = None,
    replay_min_size_override: int | None = None,
    epsilon_after_load: float | None = None,
    decay_epsilon_on_learning_only: bool = False,
) -> RunResult:
    writer_cls = require_tensorboard()
    dirs = mk_run_dirs(out_dir)
    writer = writer_cls(log_dir=dirs["tensorboard"])
    env_factory = build_dqn_env_factory(
        env_mode=env_mode,
        env_kwargs=env_kwargs,
        reward_variant=reward_variant,
        state_variant=state_variant,
    )
    train_env = env_factory()
    curriculum_rng = np.random.default_rng(int(seed) + 7_777)
    epsilon_decay = DQNAgent.decay_for_horizon(
        cfg.DQN_EPSILON_START,
        cfg.DQN_EPSILON_END,
        total_episodes,
    )
    agent = DQNAgent(
        obs_dim=state_variant.obs_dim,
        n_actions=cfg.N_ACTIONS,
        seed=seed,
        epsilon_start=cfg.DQN_EPSILON_START,
        epsilon_end=cfg.DQN_EPSILON_END,
        epsilon_decay=epsilon_decay,
    )
    if replay_min_size_override is not None:
        agent.min_replay_size = max(1, int(replay_min_size_override))
    if init_model_path is not None:
        agent.load(str(init_model_path))
        if epsilon_after_load is not None:
            agent.epsilon = float(epsilon_after_load)

    parallel_envs = max(1, int(parallel_envs))
    parallel_envs = min(parallel_envs, int(total_episodes))

    ep_returns = np.zeros(total_episodes, dtype=np.float64)
    ep_track = np.zeros(total_episodes, dtype=np.float64)
    ep_transp = np.zeros(total_episodes, dtype=np.float64)
    ep_pre_track = np.zeros(total_episodes, dtype=np.float64)
    ep_post_track = np.zeros(total_episodes, dtype=np.float64)
    ep_pre_transp = np.zeros(total_episodes, dtype=np.float64)
    ep_post_transp = np.zeros(total_episodes, dtype=np.float64)
    ep_invalid = np.zeros(total_episodes, dtype=np.float64)
    ep_q_gap = np.zeros(total_episodes, dtype=np.float64)
    ep_loss = np.full(total_episodes, np.nan, dtype=np.float64)
    eval_steps: list[int] = []
    eval_returns: list[float] = []
    eval_track: list[float] = []
    eval_transp: list[float] = []
    log_every = max(1, total_episodes // 20)
    start_time = time.time()

    if parallel_envs <= 1:
        for ep in range(total_episodes):
            reset_options = _next_reset_options(
                episode_index=ep,
                schedule=train_reset_options_schedule,
                pool=train_reset_options_pool,
                rng=curriculum_rng,
            )
            obs, _ = train_env.reset(seed=seed + ep, options=reset_options)
            done = False
            ep_return = 0.0
            losses: list[float] = []
            q_gap_trace: list[float] = []
            while not done:
                q_gap_trace.append(q_gap(agent.q_values(obs)))
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = train_env.step(action)
                done = terminated or truncated
                agent.store_transition(obs, action, reward, next_obs, done)
                loss = agent.train_step()
                if loss is not None:
                    losses.append(float(loss))
                obs = next_obs
                ep_return += reward

            if (not decay_epsilon_on_learning_only) or losses:
                agent.decay_epsilon()
            history = train_env.render() or {}
            metrics = rollout_metrics(history, env_switch_time=float(getattr(train_env, "env_switch_time", cfg.ENV_SWITCH_TIME)))
            ep_returns[ep] = ep_return
            ep_track[ep] = metrics["tracking_rmse_m"]
            ep_transp[ep] = metrics["transparency_rmse_w"]
            ep_pre_track[ep] = metrics["pre_switch_tracking_rmse_m"]
            ep_post_track[ep] = metrics["post_switch_tracking_rmse_m"]
            ep_pre_transp[ep] = metrics["pre_switch_transparency_rmse_w"]
            ep_post_transp[ep] = metrics["post_switch_transparency_rmse_w"]
            ep_invalid[ep] = metrics["invalid_episode"]
            ep_q_gap[ep] = float(np.mean(q_gap_trace)) if q_gap_trace else 0.0
            if losses:
                ep_loss[ep] = float(np.mean(losses))

            step = ep + 1
            writer.add_scalar("train/episode_return", ep_return, step)
            writer.add_scalar("train/tracking_rmse_m", ep_track[ep], step)
            writer.add_scalar("train/transparency_rmse_w", ep_transp[ep], step)
            writer.add_scalar("train/pre_switch_tracking_rmse_m", ep_pre_track[ep], step)
            writer.add_scalar("train/post_switch_tracking_rmse_m", ep_post_track[ep], step)
            writer.add_scalar("train/pre_switch_transparency_rmse_w", ep_pre_transp[ep], step)
            writer.add_scalar("train/post_switch_transparency_rmse_w", ep_post_transp[ep], step)
            writer.add_scalar("train/invalid_episode", ep_invalid[ep], step)
            writer.add_scalar("train/epsilon", agent.epsilon, step)
            writer.add_scalar("train/replay_size", len(agent.replay_buffer), step)
            writer.add_scalar("train/grad_steps", agent.train_step_count, step)
            writer.add_scalar("train/q_gap_mean", ep_q_gap[ep], step)
            if losses:
                writer.add_scalar("train/mean_td_loss", ep_loss[ep], step)

            if step == 1 or step % cfg.DQN_EVAL_EVERY == 0 or step == total_episodes:
                eval_metrics, _ = evaluate_dqn(
                    agent,
                    env_factory,
                    n_episodes=max(1, min(cfg.DQN_EVAL_EPISODES, test_episodes)),
                    seed_offset=10_000 + step,
                    parallel_envs=1,
                )
                writer.add_scalar("eval/mean_reward", eval_metrics["mean_reward"], step)
                writer.add_scalar("eval/tracking_rmse_m", eval_metrics["tracking_rmse_m"], step)
                writer.add_scalar("eval/transparency_rmse_w", eval_metrics["transparency_rmse_w"], step)
                writer.add_scalar("eval/pre_switch_tracking_rmse_m", eval_metrics["pre_switch_tracking_rmse_m"], step)
                writer.add_scalar("eval/post_switch_tracking_rmse_m", eval_metrics["post_switch_tracking_rmse_m"], step)
                writer.add_scalar("eval/pre_switch_transparency_rmse_w", eval_metrics["pre_switch_transparency_rmse_w"], step)
                writer.add_scalar("eval/post_switch_transparency_rmse_w", eval_metrics["post_switch_transparency_rmse_w"], step)
                eval_steps.append(step)
                eval_returns.append(float(eval_metrics["mean_reward"]))
                eval_track.append(float(eval_metrics["tracking_rmse_m"]))
                eval_transp.append(float(eval_metrics["transparency_rmse_w"]))

            if step == 1 or step % log_every == 0 or step == total_episodes:
                elapsed_min = (time.time() - start_time) / 60.0
                loss_text = f"{ep_loss[ep]:.6f}" if np.isfinite(ep_loss[ep]) else "nan"
                print(
                    f"[dqn] {label} | ep {step}/{total_episodes} | "
                    f"return={ep_return:.2f} | track={ep_track[ep]:.4f} m | "
                    f"transp={ep_transp[ep]:.4f} | eps={agent.epsilon:.4f} | "
                    f"replay={len(agent.replay_buffer)} | loss={loss_text} | "
                    f"elapsed={elapsed_min:.1f} min",
                    flush=True,
                )
    else:
        spec = _make_parallel_spec(
            env_mode=env_mode,
            env_kwargs=env_kwargs,
            reward_variant=reward_variant,
            state_variant=state_variant,
        )
        batch = _SubprocEnvBatch(spec, parallel_envs)
        active_slots: list[int] = []
        current_obs: dict[int, np.ndarray] = {}
        current_returns: dict[int, float] = {}
        current_q_gaps: dict[int, list[float]] = {}
        current_losses: dict[int, list[float]] = {}
        started = 0
        completed = 0
        try:
            for slot in range(batch.n_envs):
                if started >= total_episodes:
                    break
                obs, _ = batch.reset(
                    slot,
                    seed + started,
                    options=_next_reset_options(
                        episode_index=started,
                        schedule=train_reset_options_schedule,
                        pool=train_reset_options_pool,
                        rng=curriculum_rng,
                    ),
                )
                active_slots.append(slot)
                current_obs[slot] = obs
                current_returns[slot] = 0.0
                current_q_gaps[slot] = []
                current_losses[slot] = []
                started += 1

            while active_slots:
                obs_batch = np.stack([current_obs[slot] for slot in active_slots], axis=0)
                q_values_batch = agent.q_values_batch(obs_batch)
                q_gap_values = _q_gap_batch(q_values_batch)
                actions = agent.select_actions_batch(obs_batch, q_values=q_values_batch)
                results = batch.step_many(active_slots, actions)

                for slot, action, slot_q_gap, result in zip(list(active_slots), actions, q_gap_values, results):
                    next_obs, reward, terminated, truncated, _, history = result
                    done = terminated or truncated
                    current_returns[slot] += reward
                    current_q_gaps[slot].append(float(slot_q_gap))
                    agent.store_transition(current_obs[slot], int(action), reward, next_obs, done)
                    loss = agent.train_step()
                    if loss is not None:
                        current_losses[slot].append(float(loss))
                    current_obs[slot] = next_obs

                    if not done:
                        continue

                    metrics = rollout_metrics(
                        history or {},
                        env_switch_time=float(env_kwargs["env_switch_time"]),
                    )
                    ep_returns[completed] = current_returns[slot]
                    ep_track[completed] = metrics["tracking_rmse_m"]
                    ep_transp[completed] = metrics["transparency_rmse_w"]
                    ep_pre_track[completed] = metrics["pre_switch_tracking_rmse_m"]
                    ep_post_track[completed] = metrics["post_switch_tracking_rmse_m"]
                    ep_pre_transp[completed] = metrics["pre_switch_transparency_rmse_w"]
                    ep_post_transp[completed] = metrics["post_switch_transparency_rmse_w"]
                    ep_invalid[completed] = metrics["invalid_episode"]
                    ep_q_gap[completed] = float(np.mean(current_q_gaps[slot])) if current_q_gaps[slot] else 0.0
                    if current_losses[slot]:
                        ep_loss[completed] = float(np.mean(current_losses[slot]))

                    if (not decay_epsilon_on_learning_only) or current_losses[slot]:
                        agent.decay_epsilon()
                    step = completed + 1
                    writer.add_scalar("train/episode_return", ep_returns[completed], step)
                    writer.add_scalar("train/tracking_rmse_m", ep_track[completed], step)
                    writer.add_scalar("train/transparency_rmse_w", ep_transp[completed], step)
                    writer.add_scalar("train/pre_switch_tracking_rmse_m", ep_pre_track[completed], step)
                    writer.add_scalar("train/post_switch_tracking_rmse_m", ep_post_track[completed], step)
                    writer.add_scalar("train/pre_switch_transparency_rmse_w", ep_pre_transp[completed], step)
                    writer.add_scalar("train/post_switch_transparency_rmse_w", ep_post_transp[completed], step)
                    writer.add_scalar("train/invalid_episode", ep_invalid[completed], step)
                    writer.add_scalar("train/epsilon", agent.epsilon, step)
                    writer.add_scalar("train/replay_size", len(agent.replay_buffer), step)
                    writer.add_scalar("train/grad_steps", agent.train_step_count, step)
                    writer.add_scalar("train/q_gap_mean", ep_q_gap[completed], step)
                    if current_losses[slot]:
                        writer.add_scalar("train/mean_td_loss", ep_loss[completed], step)

                    if step == 1 or step % cfg.DQN_EVAL_EVERY == 0 or step == total_episodes:
                        eval_metrics, _ = evaluate_dqn(
                            agent,
                            env_factory,
                            n_episodes=max(1, min(cfg.DQN_EVAL_EPISODES, test_episodes)),
                            seed_offset=10_000 + step,
                            parallel_envs=min(parallel_envs, max(1, min(cfg.DQN_EVAL_EPISODES, test_episodes))),
                        )
                        writer.add_scalar("eval/mean_reward", eval_metrics["mean_reward"], step)
                        writer.add_scalar("eval/tracking_rmse_m", eval_metrics["tracking_rmse_m"], step)
                        writer.add_scalar("eval/transparency_rmse_w", eval_metrics["transparency_rmse_w"], step)
                        writer.add_scalar("eval/pre_switch_tracking_rmse_m", eval_metrics["pre_switch_tracking_rmse_m"], step)
                        writer.add_scalar("eval/post_switch_tracking_rmse_m", eval_metrics["post_switch_tracking_rmse_m"], step)
                        writer.add_scalar("eval/pre_switch_transparency_rmse_w", eval_metrics["pre_switch_transparency_rmse_w"], step)
                        writer.add_scalar("eval/post_switch_transparency_rmse_w", eval_metrics["post_switch_transparency_rmse_w"], step)
                        eval_steps.append(step)
                        eval_returns.append(float(eval_metrics["mean_reward"]))
                        eval_track.append(float(eval_metrics["tracking_rmse_m"]))
                        eval_transp.append(float(eval_metrics["transparency_rmse_w"]))

                    if step == 1 or step % log_every == 0 or step == total_episodes:
                        elapsed_min = (time.time() - start_time) / 60.0
                        loss_text = f"{ep_loss[completed]:.6f}" if np.isfinite(ep_loss[completed]) else "nan"
                        print(
                            f"[dqn] {label} | ep {step}/{total_episodes} | "
                            f"return={ep_returns[completed]:.2f} | track={ep_track[completed]:.4f} m | "
                            f"transp={ep_transp[completed]:.4f} | eps={agent.epsilon:.4f} | "
                            f"replay={len(agent.replay_buffer)} | loss={loss_text} | "
                            f"parallel_envs={parallel_envs} | elapsed={elapsed_min:.1f} min",
                            flush=True,
                        )

                    completed += 1
                    if started < total_episodes:
                        obs, _ = batch.reset(
                            slot,
                            seed + started,
                            options=_next_reset_options(
                                episode_index=started,
                                schedule=train_reset_options_schedule,
                                pool=train_reset_options_pool,
                                rng=curriculum_rng,
                            ),
                        )
                        current_obs[slot] = obs
                        current_returns[slot] = 0.0
                        current_q_gaps[slot] = []
                        current_losses[slot] = []
                        started += 1
                    else:
                        batch.close_slot(slot)
                        active_slots.remove(slot)
                        current_obs.pop(slot, None)
                        current_returns.pop(slot, None)
                        current_q_gaps.pop(slot, None)
                        current_losses.pop(slot, None)
        finally:
            batch.close()

    writer.flush()
    writer.close()

    agent_path = Path(dirs["models"]) / "dqn_model.pt"
    Path(dirs["models"]).mkdir(parents=True, exist_ok=True)
    Path(dirs["logs"]).mkdir(parents=True, exist_ok=True)
    Path(dirs["plots"]).mkdir(parents=True, exist_ok=True)
    Path(dirs["episodes"]).mkdir(parents=True, exist_ok=True)
    agent.save(str(agent_path))
    np.savez(
        Path(dirs["logs"]) / "train.npz",
        episode_returns=ep_returns,
        episode_tracking_rmse=ep_track,
        episode_transparency_rmse=ep_transp,
        episode_pre_switch_tracking_rmse=ep_pre_track,
        episode_post_switch_tracking_rmse=ep_post_track,
        episode_pre_switch_transparency_rmse=ep_pre_transp,
        episode_post_switch_transparency_rmse=ep_post_transp,
        episode_invalid=ep_invalid,
        episode_q_gap=ep_q_gap,
        episode_td_loss=ep_loss,
        eval_steps=np.asarray(eval_steps, dtype=np.int64),
        eval_mean_reward=np.asarray(eval_returns, dtype=np.float64),
        eval_tracking_rmse=np.asarray(eval_track, dtype=np.float64),
        eval_transparency_rmse=np.asarray(eval_transp, dtype=np.float64),
    )
    save_training_plot(
        ep_returns,
        ep_track,
        ep_transp,
        Path(dirs["plots"]) / "train.png",
        label,
        losses=ep_loss,
        eval_payload={
            "steps": np.asarray(eval_steps, dtype=np.int64),
            "mean_reward": np.asarray(eval_returns, dtype=np.float64),
            "tracking_rmse_m": np.asarray(eval_track, dtype=np.float64),
            "transparency_rmse_w": np.asarray(eval_transp, dtype=np.float64),
        },
    )

    eval_metrics, history = evaluate_dqn(
        agent,
        env_factory,
        n_episodes=test_episodes,
        seed_offset=20_000,
        parallel_envs=min(parallel_envs, max(1, test_episodes)),
    )
    save_history_npz(history, Path(dirs["episodes"]) / "test.npz")
    save_common_visuals(history, dirs["plots"], label, env_switch_time=float(env_kwargs["env_switch_time"]))
    plot_dqn_policy_slices(agent, history, state_variant, Path(dirs["plots"]) / "slices.png")

    result = RunResult(
        label=label,
        family="dqn",
        mean_reward=eval_metrics["mean_reward"],
        tracking_rmse_m=eval_metrics["tracking_rmse_m"],
        transparency_rmse_w=eval_metrics["transparency_rmse_w"],
        pre_switch_tracking_rmse_m=eval_metrics["pre_switch_tracking_rmse_m"],
        post_switch_tracking_rmse_m=eval_metrics["post_switch_tracking_rmse_m"],
        pre_switch_transparency_rmse_w=eval_metrics["pre_switch_transparency_rmse_w"],
        post_switch_transparency_rmse_w=eval_metrics["post_switch_transparency_rmse_w"],
        invalid_episode_rate=eval_metrics["invalid_episode"],
        history=history,
        out_dir=dirs["base"],
        tensorboard_dir=dirs["tensorboard"],
        model_path=str(agent_path),
        reward_variant=reward_variant.name,
        state_variant=state_variant.name,
    )
    write_run_summary(
        dirs,
        result,
        extra={
            "env_mode": env_mode,
            "master_input_mode": cfg.MASTER_INPUT_FORCE,
            "total_episodes": total_episodes,
            "test_episodes": test_episodes,
            "evaluation_history_mode": "mean_over_test_episodes",
            "obs_dim": state_variant.obs_dim,
            "state_features": list(state_variant.feature_names),
            "grad_steps": agent.train_step_count,
            "epsilon_start": float(cfg.DQN_EPSILON_START),
            "epsilon_end": float(cfg.DQN_EPSILON_END),
            "epsilon_decay": float(epsilon_decay),
            "final_epsilon": agent.epsilon,
            "episode_duration": float(env_kwargs["episode_duration"]),
            "env_switch_time": float(env_kwargs["env_switch_time"]),
            "terminate_on_error": bool(env_kwargs["terminate_on_error"]),
            "enforce_stroke_limit": bool(env_kwargs.get("enforce_stroke_limit", True)),
            "stroke_limit_mode": str(env_kwargs.get("stroke_limit_mode", "terminate")),
            "reset_options": dict(env_kwargs.get("reset_options", {})),
            "parallel_envs": int(parallel_envs),
            "curriculum_training": bool(train_reset_options_pool or train_reset_options_schedule),
            "train_reset_options_pool": list(train_reset_options_pool or []),
            "train_reset_options_schedule": list(train_reset_options_schedule or []),
            "init_model_path": str(init_model_path) if init_model_path is not None else None,
            "replay_min_size": int(agent.min_replay_size),
            "epsilon_after_load": None if epsilon_after_load is None else float(epsilon_after_load),
            "decay_epsilon_on_learning_only": bool(decay_epsilon_on_learning_only),
        },
    )
    return result
