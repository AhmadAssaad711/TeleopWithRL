"""Q-learning training functions that consume the shared replica environment."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from ..config import config as cfg
from .agent import QLearningAgent
from ..environment.simuoriginal_env import SimuOriginalReplicaEnv
from ..common.study_utils import (
    RunResult,
    aggregate_episode_histories,
    greedy_q_action,
    history_with_obs,
    mk_run_dirs,
    q_gap,
    require_tensorboard,
    rollout_metrics,
    save_common_visuals,
    save_history_npz,
    save_training_plot,
    write_run_summary,
    plot_qlearning_policy_maps,
    plot_qlearning_state_visit_heatmap,
    resolve_action_levels,
)
from .state_variants import QLStateVariant
from ..common.rewarding import RewardVariant, ReplicaRewardEnv


def build_qlearning_env_factory(
    *,
    env_mode: str,
    env_kwargs: dict,
    reward_variant: RewardVariant,
):
    """Return a factory producing fresh reward-wrapped replica environments."""
    def _factory():
        base_env = SimuOriginalReplicaEnv(
            env_mode=env_mode,
            master_input_mode=cfg.MASTER_INPUT_FORCE,
            **dict(env_kwargs),
        )
        return ReplicaRewardEnv(base_env, reward_variant)

    return _factory


def evaluate_qlearning(
    agent: QLearningAgent,
    env_factory,
    state_variant: QLStateVariant,
    n_episodes: int,
    seed_offset: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate a tabular policy greedily and return metrics plus history."""
    episode_metrics: list[dict[str, float]] = []
    episode_histories: list[dict[str, Any]] = []
    completed_episodes = 0
    terminated_episodes = 0
    truncated_episodes = 0
    stroke_limit_episodes = 0
    tracking_error_fail_episodes = 0
    volume_singularity_episodes = 0
    stroke_stop_hit_episodes = 0
    episode_steps: list[int] = []
    for ep in range(n_episodes):
        env = env_factory()
        obs, info = env.reset(seed=seed_offset + ep)
        action_levels = resolve_action_levels(getattr(env, "action_levels", None), expected_n_actions=agent.n_actions)
        state = state_variant.discretizer(obs, info)
        done = False
        obs_trace: list[np.ndarray] = []
        final_info: dict[str, Any] = dict(info)
        final_terminated = False
        final_truncated = False
        while not done:
            obs_trace.append(np.asarray(obs, dtype=np.float32).copy())
            action = greedy_q_action(agent.q_values(state), action_levels=action_levels)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            final_info = dict(info)
            final_terminated = bool(terminated)
            final_truncated = bool(truncated)
            state = state_variant.discretizer(obs, info)
        history = history_with_obs(env.render() or {}, obs_trace)
        metrics = rollout_metrics(history, env_switch_time=float(getattr(env, "env_switch_time", cfg.ENV_SWITCH_TIME)))
        episode_metrics.append(metrics)
        episode_histories.append(history)
        episode_steps.append(int(len(history.get("time", []))))
        if np.any(np.asarray(history.get("hit_stroke_stop", []), dtype=np.float64) > 0.0):
            stroke_stop_hit_episodes += 1
        if str(final_info.get("termination_reason")) == "max_steps" or final_truncated:
            completed_episodes += 1
        if str(final_info.get("termination_reason")) == "stroke_limit":
            stroke_limit_episodes += 1
        if str(final_info.get("termination_reason")) == "tracking_error_fail":
            tracking_error_fail_episodes += 1
        if str(final_info.get("termination_reason")) == "volume_singularity":
            volume_singularity_episodes += 1
        if final_terminated:
            terminated_episodes += 1
        if final_truncated:
            truncated_episodes += 1

    aggregate = {
        key: float(np.mean([row[key] for row in episode_metrics])) for key in episode_metrics[0].keys()
    }
    aggregate_history = aggregate_episode_histories(episode_histories)
    aggregate.update(
        {
            "evaluation_episodes": float(n_episodes),
            "completed_episodes": float(completed_episodes),
            "completed_episode_rate": float(completed_episodes / max(1, n_episodes)),
            "terminated_episodes": float(terminated_episodes),
            "truncated_episodes": float(truncated_episodes),
            "stroke_limit_episodes": float(stroke_limit_episodes),
            "tracking_error_fail_episodes": float(tracking_error_fail_episodes),
            "volume_singularity_episodes": float(volume_singularity_episodes),
            "stroke_stop_hit_episodes": float(stroke_stop_hit_episodes),
            "mean_episode_steps": float(np.mean(episode_steps)) if episode_steps else 0.0,
            "mean_episode_seconds": float(np.mean(episode_steps) * cfg.RL_DT) if episode_steps else 0.0,
        }
    )
    aggregate_history["evaluation_episodes"] = int(n_episodes)
    aggregate_history["completed_episodes"] = int(completed_episodes)
    aggregate_history["completed_episode_rate"] = float(completed_episodes / max(1, n_episodes))
    aggregate_history["terminated_episodes"] = int(terminated_episodes)
    aggregate_history["truncated_episodes"] = int(truncated_episodes)
    aggregate_history["stroke_limit_episodes"] = int(stroke_limit_episodes)
    aggregate_history["tracking_error_fail_episodes"] = int(tracking_error_fail_episodes)
    aggregate_history["volume_singularity_episodes"] = int(volume_singularity_episodes)
    aggregate_history["stroke_stop_hit_episodes"] = int(stroke_stop_hit_episodes)
    aggregate_history["mean_episode_steps"] = float(np.mean(episode_steps)) if episode_steps else 0.0
    aggregate_history["mean_episode_seconds"] = float(np.mean(episode_steps) * cfg.RL_DT) if episode_steps else 0.0
    return aggregate, aggregate_history


def train_qlearning_variant(
    *,
    out_dir: str | Path,
    env_mode: str,
    env_kwargs: dict,
    state_variant: QLStateVariant,
    reward_variant: RewardVariant,
    total_episodes: int,
    test_episodes: int,
    seed: int,
    label: str,
) -> RunResult:
    """Train one Q-learning state/reward variant and write its artifacts.

    The state encoder supplies the integer tuple used by the sparse table. The
    returned ``RunResult`` and the files under ``out_dir`` follow the shared
    DQN/policy-gradient result contract.
    """
    writer_cls = require_tensorboard()
    dirs = mk_run_dirs(out_dir)
    writer = writer_cls(log_dir=dirs["tensorboard"])
    env_factory = build_qlearning_env_factory(env_mode=env_mode, env_kwargs=env_kwargs, reward_variant=reward_variant)
    train_env = env_factory()
    action_levels = resolve_action_levels(getattr(train_env, "action_levels", None))
    epsilon_decay = QLearningAgent.decay_for_horizon(
        cfg.EPSILON_START,
        cfg.EPSILON_END,
        total_episodes,
    )
    agent = QLearningAgent(
        state_dims=state_variant.state_dims,
        n_actions=int(action_levels.size),
        seed=seed,
        epsilon_start=cfg.EPSILON_START,
        epsilon_end=cfg.EPSILON_END,
        epsilon_decay=epsilon_decay,
    )

    ep_returns = np.zeros(total_episodes, dtype=np.float64)
    ep_track = np.zeros(total_episodes, dtype=np.float64)
    ep_transp = np.zeros(total_episodes, dtype=np.float64)
    ep_pre_track = np.zeros(total_episodes, dtype=np.float64)
    ep_post_track = np.zeros(total_episodes, dtype=np.float64)
    ep_pre_transp = np.zeros(total_episodes, dtype=np.float64)
    ep_post_transp = np.zeros(total_episodes, dtype=np.float64)
    ep_invalid = np.zeros(total_episodes, dtype=np.float64)
    ep_q_gap = np.zeros(total_episodes, dtype=np.float64)
    eval_steps: list[int] = []
    eval_returns: list[float] = []
    eval_track: list[float] = []
    eval_transp: list[float] = []
    log_every = max(1, total_episodes // 20)
    start_time = time.time()

    for ep in range(total_episodes):
        obs, info = train_env.reset(seed=seed + ep)
        state = state_variant.discretizer(obs, info)
        done = False
        ep_return = 0.0
        q_gap_trace: list[float] = []

        while not done:
            q_gap_trace.append(q_gap(agent.q_values(state)))
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, next_info = train_env.step(action)
            next_state = state_variant.discretizer(next_obs, next_info)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            obs = next_obs
            info = next_info
            ep_return += reward

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
        writer.add_scalar("train/discovered_states", agent.discovered_states(), step)
        writer.add_scalar("train/action_coverage", agent.coverage(), step)
        writer.add_scalar("train/q_gap_mean", ep_q_gap[ep], step)

        if step == 1 or step % cfg.EVAL_EVERY == 0 or step == total_episodes:
            eval_metrics, _ = evaluate_qlearning(
                agent,
                env_factory,
                state_variant,
                n_episodes=max(1, min(cfg.EVAL_EPISODES, test_episodes)),
                seed_offset=10_000 + step,
            )
            writer.add_scalar("eval/mean_reward", eval_metrics["mean_reward"], step)
            writer.add_scalar("eval/tracking_rmse_m", eval_metrics["tracking_rmse_m"], step)
            writer.add_scalar("eval/transparency_rmse_w", eval_metrics["transparency_rmse_w"], step)
            writer.add_scalar("eval/pre_switch_tracking_rmse_m", eval_metrics["pre_switch_tracking_rmse_m"], step)
            writer.add_scalar("eval/post_switch_tracking_rmse_m", eval_metrics["post_switch_tracking_rmse_m"], step)
            writer.add_scalar("eval/pre_switch_transparency_rmse_w", eval_metrics["pre_switch_transparency_rmse_w"], step)
            writer.add_scalar("eval/post_switch_transparency_rmse_w", eval_metrics["post_switch_transparency_rmse_w"], step)
            writer.add_scalar("eval/completed_episode_rate", eval_metrics["completed_episode_rate"], step)
            writer.add_scalar("eval/stroke_limit_episode_rate", eval_metrics["stroke_limit_episodes"] / max(1, eval_metrics["evaluation_episodes"]), step)
            writer.add_scalar("eval/tracking_error_fail_rate", eval_metrics["tracking_error_fail_episodes"] / max(1, eval_metrics["evaluation_episodes"]), step)
            eval_steps.append(step)
            eval_returns.append(float(eval_metrics["mean_reward"]))
            eval_track.append(float(eval_metrics["tracking_rmse_m"]))
            eval_transp.append(float(eval_metrics["transparency_rmse_w"]))

        if step == 1 or step % log_every == 0 or step == total_episodes:
            elapsed_min = (time.time() - start_time) / 60.0
            print(
                f"[q_learning] {label} | ep {step}/{total_episodes} | "
                f"return={ep_return:.2f} | track={ep_track[ep]:.4f} m | "
                f"transp={ep_transp[ep]:.4f} | eps={agent.epsilon:.4f} | "
                f"states={agent.discovered_states()} | elapsed={elapsed_min:.1f} min",
                flush=True,
            )

    writer.flush()
    writer.close()

    agent_path = Path(dirs["models"]) / "q_table.npy"
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
        eval_payload={
            "steps": np.asarray(eval_steps, dtype=np.int64),
            "mean_reward": np.asarray(eval_returns, dtype=np.float64),
            "tracking_rmse_m": np.asarray(eval_track, dtype=np.float64),
            "transparency_rmse_w": np.asarray(eval_transp, dtype=np.float64),
        },
    )

    eval_metrics, history = evaluate_qlearning(
        agent,
        env_factory,
        state_variant,
        n_episodes=test_episodes,
        seed_offset=20_000,
    )
    save_history_npz(history, Path(dirs["episodes"]) / "test.npz")
    save_common_visuals(
        history,
        dirs["plots"],
        label,
        env_switch_time=float(env_kwargs["env_switch_time"]),
        action_levels=action_levels,
    )
    plot_qlearning_policy_maps(
        agent,
        state_variant.feature_names,
        Path(dirs["plots"]) / "polmap.png",
        action_levels=action_levels,
    )
    plot_qlearning_state_visit_heatmap(agent, state_variant.feature_names, Path(dirs["plots"]) / "visit.png")

    result = RunResult(
        label=label,
        family="q_learning",
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
            "epsilon_start": float(cfg.EPSILON_START),
            "epsilon_end": float(cfg.EPSILON_END),
            "epsilon_decay": float(epsilon_decay),
            "final_epsilon": float(agent.epsilon),
            "state_dims": state_variant.state_dims,
            "state_features": list(state_variant.feature_names),
            "action_levels": action_levels.tolist(),
            "episode_duration": float(env_kwargs["episode_duration"]),
            "env_switch_time": float(env_kwargs["env_switch_time"]),
            "terminate_on_error": bool(env_kwargs["terminate_on_error"]),
            "enforce_stroke_limit": bool(env_kwargs.get("enforce_stroke_limit", True)),
            "stroke_limit_mode": str(env_kwargs.get("stroke_limit_mode", "terminate")),
            "evaluation_episodes": int(eval_metrics["evaluation_episodes"]),
            "completed_episodes": int(eval_metrics["completed_episodes"]),
            "completed_episode_rate": float(eval_metrics["completed_episode_rate"]),
            "terminated_episodes": int(eval_metrics["terminated_episodes"]),
            "truncated_episodes": int(eval_metrics["truncated_episodes"]),
            "stroke_limit_episodes": int(eval_metrics["stroke_limit_episodes"]),
            "tracking_error_fail_episodes": int(eval_metrics["tracking_error_fail_episodes"]),
            "volume_singularity_episodes": int(eval_metrics["volume_singularity_episodes"]),
            "stroke_stop_hit_episodes": int(eval_metrics["stroke_stop_hit_episodes"]),
            "mean_episode_steps": float(eval_metrics["mean_episode_steps"]),
            "mean_episode_seconds": float(eval_metrics["mean_episode_seconds"]),
            "reset_options": dict(env_kwargs.get("reset_options", {})),
        },
    )
    return result
