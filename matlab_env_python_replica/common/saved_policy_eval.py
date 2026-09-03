"""Evaluation and serialization helpers for already-trained policies."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..config import config as cfg
from ..dqn.agent import DQNAgent
from ..ql.agent import QLearningAgent
from .study_utils import (
    action_fraction,
    greedy_q_action,
    history_array,
    plot_episode_metrics,
    plot_input_signal_dashboard,
    plot_policy_dashboard,
    plot_scenario_dashboard,
    plot_summary_bars,
    policy_summary,
    q_gap,
    resolve_action_levels,
    save_json,
    scenario_plan,
    transparency_power_error_array,
    transparency_ratio_array,
)
from ..dqn.training import build_dqn_env_factory
from ..dqn.state_variants import get_dqn_state_variant
from ..ql.training import build_qlearning_env_factory
from ..ql.state_variants import get_ql_state_variant
from .rewarding import reward_variant_from_name


def resolve_model_path(path: str | Path) -> Path:
    """Resolve a DQN checkpoint or Q-table from a file or run directory."""
    path = Path(path)
    if path.is_file():
        return path
    for candidate in (path / "m" / "dqn_model.pt", path / "m" / "q_table.npy", path / "dqn_model.pt", path / "q_table.npy"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find a saved model under: {path}")


def _summary_for_model(model_path: Path) -> dict[str, Any]:
    summary_path = model_path.parent.parent / "l" / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as fh:
        return __import__("json").load(fh)


def _build_eval_context(summary: dict[str, Any]):
    reward_variant = reward_variant_from_name(str(summary["reward_variant"]))
    action_levels = resolve_action_levels(summary.get("action_levels", None))
    env_kwargs = {
        "episode_duration": float(summary["episode_duration"]),
        "env_switch_time": float(summary["env_switch_time"]),
        "terminate_on_error": bool(summary["terminate_on_error"]),
        "action_levels": action_levels.tolist(),
        "reset_options": dict(summary.get("reset_options", {})),
    }
    env_mode = str(summary["env_mode"])
    family = str(summary["family"])
    state_name = str(summary["state_variant"])
    if family == "q_learning":
        state_variant = get_ql_state_variant(state_name)
        env_factory = build_qlearning_env_factory(env_mode=env_mode, env_kwargs=env_kwargs, reward_variant=reward_variant)
    elif family == "dqn":
        state_variant = get_dqn_state_variant(state_name)
        env_factory = build_dqn_env_factory(env_mode=env_mode, env_kwargs=env_kwargs, reward_variant=reward_variant, state_variant=state_variant)
    else:
        raise ValueError(f"Unsupported saved-policy family: {family}")
    return family, state_variant, reward_variant, env_factory, env_kwargs


def _episode_metrics(history: dict[str, Any], episode_policy_rows: list[dict[str, Any]]) -> dict[str, float]:
    reward = history_array(history, "reward", dtype=np.float64)
    pos_error = history_array(history, "pos_error", dtype=np.float64)
    transparency_power_error = transparency_power_error_array(history)
    transparency_ratio = transparency_ratio_array(history)
    u_v = history_array(history, "u_v", dtype=np.float64)
    q_gap_arr = np.asarray([row["q_gap"] for row in episode_policy_rows], dtype=np.float64)
    max_q = np.asarray([row["max_q"] for row in episode_policy_rows], dtype=np.float64)
    chosen_q = np.asarray([row["chosen_q"] for row in episode_policy_rows], dtype=np.float64)
    return {
        "episode_return": float(reward.sum()) if reward.size else 0.0,
        "tracking_rmse_mm": float(np.sqrt(np.mean(pos_error ** 2)) * 1000.0) if pos_error.size else 0.0,
        "transparency_rmse_w": (
            float(np.sqrt(np.mean(transparency_power_error ** 2))) if transparency_power_error.size else 0.0
        ),
        "transparency_ratio_mean": float(np.mean(transparency_ratio)) if transparency_ratio.size else 0.0,
        "transparency_ratio_error_rmse": (
            float(np.sqrt(np.mean((transparency_ratio - 1.0) ** 2))) if transparency_ratio.size else 0.0
        ),
        "mean_abs_u_v": float(np.mean(np.abs(u_v))) if u_v.size else 0.0,
        "mean_q_gap": float(np.mean(q_gap_arr)) if q_gap_arr.size else 0.0,
        "mean_max_q": float(np.mean(max_q)) if max_q.size else 0.0,
        "mean_chosen_q": float(np.mean(chosen_q)) if chosen_q.size else 0.0,
    }


def evaluate_saved_policy(
    *,
    model_path: str | Path,
    episodes: int,
    seed: int,
    scenario_set: str | None,
    noise_std: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], float]:
    """Evaluate a saved DQN or Q-learning policy on a scenario set.

    Returns episode rows, aggregate metrics, per-step policy rows, a policy
    summary, and the environment switch time used by the run.
    """
    model_path = resolve_model_path(model_path)
    summary = _summary_for_model(model_path)
    family, state_variant, reward_variant, env_factory, env_kwargs = _build_eval_context(summary)
    action_levels = resolve_action_levels(summary.get("action_levels", None))
    scenarios = scenario_plan(scenario_set, noise_std)
    if scenarios is not None:
        episodes = len(scenarios)

    if family == "q_learning":
        agent = QLearningAgent(state_dims=state_variant.state_dims, n_actions=int(action_levels.size), seed=seed)
        agent.load(str(model_path))
    else:
        agent = DQNAgent(obs_dim=state_variant.obs_dim, n_actions=int(action_levels.size), seed=seed)
        agent.load(str(model_path))

    episode_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    env_switch_time = float(env_kwargs["env_switch_time"])

    for ep in range(episodes):
        scenario = scenarios[ep] if scenarios is not None else {"name": f"episode_{ep + 1:02d}", "reset_options": {}}
        env = env_factory()
        obs, info = env.reset(seed=seed + ep, options=scenario["reset_options"])
        base_env = env.base_env if hasattr(env, "base_env") else env
        env_switch_time = float(getattr(base_env, "env_switch_time", env_switch_time))
        done = False
        ep_rows: list[dict[str, Any]] = []

        while not done:
            if family == "q_learning":
                state = state_variant.discretizer(obs, info)
                q_values = agent.q_values(state)
                action = greedy_q_action(q_values, action_levels=action_levels)
            else:
                q_values = agent.q_values(obs)
                action = int(np.argmax(q_values))

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            row = {
                "episode": ep + 1,
                "scenario_name": str(scenario["name"]),
                "time": float(next_info["time"]),
                "env_label": str(next_info["env_label"]),
                "action_idx": int(action),
                "u_v": float(action_levels[int(action)]),
                "reward": float(reward),
                "pos_error": float(next_info["x_m"] - next_info["x_s"]),
                "transparency_ratio": float(next_info.get("transparency_ratio", 0.0)),
                "transparency_error": float(next_info.get("transparency_error", 0.0)),
                "chosen_q": float(q_values[int(action)]),
                "max_q": float(np.max(q_values)),
                "q_gap": q_gap(q_values),
                "F_h": float(next_info["F_h"]),
                "F_h_nominal": float(next_info["F_h_nominal"]),
                "F_h_noise": float(next_info["F_h_noise"]),
            }
            ep_rows.append(row)
            obs, info = next_obs, next_info
            done = terminated or truncated

        history = env.render() or {}
        row = {
            "episode": ep + 1,
            "scenario_name": str(scenario["name"]),
            "terminated": int(bool(terminated)),
            "truncated": int(bool(truncated)),
            "reward_variant": reward_variant.name,
            "state_variant": state_variant.name,
        }
        row.update(_episode_metrics(history, ep_rows))
        episode_rows.append(row)
        policy_rows.extend(ep_rows)

    returns = np.asarray([row["episode_return"] for row in episode_rows], dtype=np.float64)
    tracking = np.asarray([row["tracking_rmse_mm"] for row in episode_rows], dtype=np.float64)
    transparency = np.asarray([row["transparency_rmse_w"] for row in episode_rows], dtype=np.float64)
    mean_abs_u = np.asarray([row["mean_abs_u_v"] for row in episode_rows], dtype=np.float64)
    mean_q_gap = np.asarray([row["mean_q_gap"] for row in episode_rows], dtype=np.float64)
    term = np.asarray([row["terminated"] for row in episode_rows], dtype=np.float64)
    aggregate = {
        "model_path": str(model_path),
        "episodes": int(episodes),
        "scenario_set": scenario_set or "baseline_repeat",
        "noise_std_n": float(noise_std) if scenario_set == "force_noise_10" else 0.0,
        "reward_name": reward_variant.name,
        "state_variant_name": state_variant.name,
        "family": family,
        "action_levels": action_levels.tolist(),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_tracking_rmse_mm": float(np.mean(tracking)),
        "std_tracking_rmse_mm": float(np.std(tracking)),
        "mean_transparency_rmse_w": float(np.mean(transparency)),
        "std_transparency_rmse_w": float(np.std(transparency)),
        "mean_abs_u_v": float(np.mean(mean_abs_u)),
        "mean_q_gap": float(np.mean(mean_q_gap)),
        "terminated_fraction": float(np.mean(term)),
    }
    return episode_rows, aggregate, policy_rows, policy_summary(policy_rows, episode_rows, action_levels=action_levels), env_switch_time


def save_evaluation_bundle(
    *,
    out_dir: str | Path,
    prefix: str,
    episode_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    policy_rows: list[dict[str, Any]],
    policy_summary_payload: dict[str, Any],
    env_switch_time: float,
) -> None:
    """Write evaluation CSV/JSON summaries and diagnostic plots to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}_metrics.csv"
    json_path = out_dir / f"{prefix}_summary.json"
    policy_csv_path = out_dir / f"{prefix}_policy_steps.csv"
    policy_json_path = out_dir / f"{prefix}_policy.json"

    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    with open(policy_csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(policy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(policy_rows)

    save_json(json_path, summary)
    save_json(policy_json_path, policy_summary_payload)
    plot_episode_metrics(episode_rows, out_dir / f"{prefix}_metrics.png")
    plot_summary_bars(summary, out_dir / f"{prefix}_bars.png")
    plot_policy_dashboard(
        policy_rows,
        out_dir / f"{prefix}_policy.png",
        env_switch_time,
        action_levels=summary.get("action_levels"),
    )
    plot_scenario_dashboard(episode_rows, out_dir / f"{prefix}_scenario.png")
    plot_input_signal_dashboard(policy_rows, episode_rows, out_dir / f"{prefix}_input.png", env_switch_time)
