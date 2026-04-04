"""Temporary evaluator for saved DQN checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .. import config as cfg
from ..dqn_agent import DQNAgent
from ..state_variants import StateVariantEnv, get_state_variant
from ..teleop_env import TeleopEnv
from .long_study import RewardAblationEnv, RewardVariant


def _fs_path(path: str | Path) -> str:
    resolved = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def _load_summary_for_model(model_path: Path) -> dict:
    summary_path = model_path.parent.parent / "l" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find sibling summary.json for model: {model_path}")
    with open(_fs_path(summary_path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_model_path(model_path_arg: str) -> Path:
    path = Path(model_path_arg).resolve()
    if path.is_dir():
        candidate = path / "m" / "dqn_model.pt"
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find dqn_model.pt under folder: {path}")
        return candidate
    return path


def _build_env_factory(model_summary: dict, disable_jerk: bool = False):
    env_mode = model_summary.get("env_mode", cfg.ENV_MODE_CHANGING)
    master_input_mode = model_summary.get("master_input_mode", cfg.DEFAULT_MASTER_INPUT_MODE)
    reward_variant_cfg = model_summary.get("reward_variant")
    state_variant_name = model_summary.get("state_variant_name")
    state_variant = get_state_variant(str(state_variant_name)) if state_variant_name else None
    obs_dim = int(model_summary.get("obs_dim", state_variant.obs_dim if state_variant is not None else 10))

    def maybe_wrap(env):
        if state_variant is None:
            return env
        return StateVariantEnv(env, state_variant)

    if reward_variant_cfg:
        reward_variant = RewardVariant(
            name=str(reward_variant_cfg["name"]),
            tracking_weight=float(reward_variant_cfg["tracking_weight"]),
            transparency_weight=float(reward_variant_cfg["transparency_weight"]),
            jerk_weight=float(reward_variant_cfg["jerk_weight"]),
            use_jerk=bool(reward_variant_cfg["use_jerk"]),
        )
        if disable_jerk:
            reward_variant = RewardVariant(
                name=f"{reward_variant.name}_nojerk_eval",
                tracking_weight=reward_variant.tracking_weight,
                transparency_weight=reward_variant.transparency_weight,
                jerk_weight=0.0,
                use_jerk=False,
            )

        def factory():
            env = RewardAblationEnv(
                TeleopEnv(env_mode=env_mode, master_input_mode=master_input_mode),
                reward_variant,
            )
            return maybe_wrap(env)

        return factory, reward_variant.name, env_mode, master_input_mode, state_variant_name, obs_dim

    def factory():
        env = TeleopEnv(env_mode=env_mode, master_input_mode=master_input_mode)
        return maybe_wrap(env)

    return factory, "base_env_reward", env_mode, master_input_mode, state_variant_name, obs_dim


def _history_array(history: dict[str, Any], key: str, dtype=np.float64) -> np.ndarray:
    values = history.get(key, [])
    return np.asarray(values, dtype=dtype)


def _q_gap(q_values: np.ndarray) -> float:
    q_values = np.asarray(q_values, dtype=np.float64)
    if q_values.size <= 1:
        return float(q_values[0]) if q_values.size else 0.0
    sorted_q = np.sort(q_values)
    return float(sorted_q[-1] - sorted_q[-2])


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _scenario_plan(
    scenario_set: str | None,
    model_summary: dict,
    noise_std: float,
) -> list[dict[str, Any]] | None:
    if not scenario_set:
        return None

    master_input_mode = model_summary.get("master_input_mode", cfg.DEFAULT_MASTER_INPUT_MODE)
    if master_input_mode != cfg.MASTER_INPUT_FORCE:
        raise ValueError("The built-in waveform sweep currently supports only force-mode evaluation.")

    amp = float(cfg.FORCE_INPUT_AMP)
    phase = float(cfg.FORCE_INPUT_PHASE)
    if scenario_set == "force_generalization_10":
        return [
            {"name": "sine_0p25hz", "reset_options": {"force_amp": amp, "force_freq": 0.25, "force_phase": phase, "force_waveform": "sine"}},
            {"name": "sine_0p35hz", "reset_options": {"force_amp": amp, "force_freq": 0.35, "force_phase": phase, "force_waveform": "sine"}},
            {"name": "sine_0p50hz", "reset_options": {"force_amp": amp, "force_freq": 0.50, "force_phase": phase, "force_waveform": "sine"}},
            {"name": "sine_0p75hz", "reset_options": {"force_amp": amp, "force_freq": 0.75, "force_phase": phase, "force_waveform": "sine"}},
            {"name": "sine_1p00hz", "reset_options": {"force_amp": amp, "force_freq": 1.00, "force_phase": phase, "force_waveform": "sine"}},
            {"name": "cosine_0p50hz", "reset_options": {"force_amp": amp, "force_freq": 0.50, "force_phase": phase, "force_waveform": "cosine"}},
            {"name": "cosine_0p75hz", "reset_options": {"force_amp": amp, "force_freq": 0.75, "force_phase": phase, "force_waveform": "cosine"}},
            {"name": "multisine_0p50hz", "reset_options": {"force_amp": amp, "force_freq": 0.50, "force_phase": phase, "force_waveform": "multisine"}},
            {"name": "multisine_0p75hz", "reset_options": {"force_amp": amp, "force_freq": 0.75, "force_phase": phase, "force_waveform": "multisine"}},
            {"name": "multisine_1p00hz", "reset_options": {"force_amp": amp, "force_freq": 1.00, "force_phase": phase, "force_waveform": "multisine"}},
        ]

    if scenario_set == "force_square_10":
        square_freqs = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10]
        return [
            {
                "name": f"square_{freq:0.2f}hz".replace(".", "p"),
                "reset_options": {
                    "force_amp": amp,
                    "force_freq": float(freq),
                    "force_phase": phase,
                    "force_waveform": "square",
                },
            }
            for freq in square_freqs
        ]

    if scenario_set == "force_noise_10":
        return [
            {
                "name": f"noisy_sine_{idx + 1:02d}",
                "reset_options": {
                    "force_amp": amp,
                    "force_freq": float(cfg.FORCE_INPUT_FREQ),
                    "force_phase": phase,
                    "force_waveform": "sine",
                    "force_noise_std": float(noise_std),
                    "force_noise_seed": 17_100 + idx,
                },
            }
            for idx in range(10)
        ]

    raise ValueError(f"Unknown scenario set: {scenario_set}")


def _episode_metrics(history: dict, episode_policy_rows: list[dict[str, Any]]) -> dict[str, float]:
    reward = _history_array(history, "reward", dtype=np.float64)
    pos_error = _history_array(history, "pos_error", dtype=np.float64)
    transparency_error = _history_array(history, "transparency_error", dtype=np.float64)
    u_v = _history_array(history, "u_v", dtype=np.float64)

    delta_u = np.diff(u_v) if u_v.size >= 2 else np.zeros(0, dtype=np.float64)
    q_gap = np.asarray([row["q_gap"] for row in episode_policy_rows], dtype=np.float64)
    max_q = np.asarray([row["max_q"] for row in episode_policy_rows], dtype=np.float64)
    chosen_q = np.asarray([row["chosen_q"] for row in episode_policy_rows], dtype=np.float64)

    return {
        "episode_return": float(reward.sum()) if reward.size else 0.0,
        "tracking_rmse_m": float(np.sqrt(np.mean(pos_error ** 2))) if pos_error.size else 0.0,
        "tracking_rmse_mm": float(np.sqrt(np.mean(pos_error ** 2)) * 1000.0) if pos_error.size else 0.0,
        "transparency_rmse_w": float(np.sqrt(np.mean(transparency_error ** 2))) if transparency_error.size else 0.0,
        "mean_abs_u_v": float(np.mean(np.abs(u_v))) if u_v.size else 0.0,
        "max_abs_u_v": float(np.max(np.abs(u_v))) if u_v.size else 0.0,
        "mean_abs_delta_u_v": float(np.mean(np.abs(delta_u))) if delta_u.size else 0.0,
        "mean_q_gap": _safe_mean(q_gap),
        "mean_max_q": _safe_mean(max_q),
        "mean_chosen_q": _safe_mean(chosen_q),
    }


def _policy_rows_for_episode(
    episode_idx: int,
    history: dict[str, Any],
    action_indices: list[int],
    q_trace: list[np.ndarray],
    reward_name: str,
    env_mode: str,
    master_input_mode: str,
    scenario_name: str,
    force_freq_hz: float,
    force_waveform: str,
    force_noise_std: float,
    force_noise_seed: int,
) -> list[dict[str, Any]]:
    time_s = _history_array(history, "time", dtype=np.float64)
    reward = _history_array(history, "reward", dtype=np.float64)
    pos_error_mm = _history_array(history, "pos_error", dtype=np.float64) * 1000.0
    transparency_error = _history_array(history, "transparency_error", dtype=np.float64)
    env_label = _history_array(history, "env_label", dtype=object)
    force_input = _history_array(history, "F_h", dtype=np.float64)
    force_nominal = _history_array(history, "F_h_nominal", dtype=np.float64)
    force_noise = _history_array(history, "F_h_noise", dtype=np.float64)

    step_count = min(len(action_indices), len(q_trace), time_s.size)
    rows: list[dict[str, Any]] = []
    for step_idx in range(step_count):
        q_values = np.asarray(q_trace[step_idx], dtype=np.float64)
        action_idx = int(action_indices[step_idx])
        action_voltage = float(cfg.V_LEVELS[action_idx])
        rows.append(
            {
                "episode": episode_idx + 1,
                "scenario_name": scenario_name,
                "force_freq_hz": float(force_freq_hz),
                "force_waveform": force_waveform,
                "force_noise_std": float(force_noise_std),
                "force_noise_seed": int(force_noise_seed),
                "step": step_idx + 1,
                "time_s": float(time_s[step_idx]),
                "env_label": str(env_label[step_idx]) if env_label.size else "",
                "reward_name": reward_name,
                "env_mode": env_mode,
                "master_input_mode": master_input_mode,
                "action_index": action_idx,
                "action_voltage_v": action_voltage,
                "reward": float(reward[step_idx]) if reward.size > step_idx else 0.0,
                "pos_error_mm": float(pos_error_mm[step_idx]) if pos_error_mm.size > step_idx else 0.0,
                "transparency_error_w": float(transparency_error[step_idx]) if transparency_error.size > step_idx else 0.0,
                "force_input_n": float(force_input[step_idx]) if force_input.size > step_idx else 0.0,
                "force_nominal_n": float(force_nominal[step_idx]) if force_nominal.size > step_idx else 0.0,
                "force_noise_n": float(force_noise[step_idx]) if force_noise.size > step_idx else 0.0,
                "chosen_q": float(q_values[action_idx]),
                "max_q": float(np.max(q_values)),
                "q_gap": _q_gap(q_values),
            }
        )
    return rows


def _action_fraction(rows: list[dict[str, Any]], env_name: str | None = None) -> dict[str, float]:
    filtered = rows if env_name is None else [row for row in rows if row["env_label"] == env_name]
    if not filtered:
        return {f"{voltage:.1f}": 0.0 for voltage in cfg.V_LEVELS}

    counts = {f"{voltage:.1f}": 0.0 for voltage in cfg.V_LEVELS}
    for row in filtered:
        key = f"{row['action_voltage_v']:.1f}"
        counts[key] += 1.0
    total = float(len(filtered))
    return {key: value / total for key, value in counts.items()}


def _policy_summary(policy_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    q_gap = np.asarray([row["q_gap"] for row in policy_rows], dtype=np.float64)
    max_q = np.asarray([row["max_q"] for row in policy_rows], dtype=np.float64)
    chosen_q = np.asarray([row["chosen_q"] for row in policy_rows], dtype=np.float64)
    action_voltage = np.asarray([row["action_voltage_v"] for row in policy_rows], dtype=np.float64)

    env_rows = {
        env_name: [row for row in policy_rows if row["env_label"] == env_name]
        for env_name in cfg.ENV_LABELS
    }
    scenario_names = sorted({row["scenario_name"] for row in policy_rows})

    return {
        "total_steps": int(len(policy_rows)),
        "mean_q_gap": _safe_mean(q_gap),
        "std_q_gap": float(np.std(q_gap)) if q_gap.size else 0.0,
        "mean_max_q": _safe_mean(max_q),
        "std_max_q": float(np.std(max_q)) if max_q.size else 0.0,
        "mean_chosen_q": _safe_mean(chosen_q),
        "std_chosen_q": float(np.std(chosen_q)) if chosen_q.size else 0.0,
        "mean_action_voltage_v": _safe_mean(action_voltage),
        "mean_abs_action_voltage_v": float(np.mean(np.abs(action_voltage))) if action_voltage.size else 0.0,
        "action_usage_fraction": _action_fraction(policy_rows),
        "action_usage_by_env_fraction": {
            env_name: _action_fraction(policy_rows, env_name=env_name) for env_name in cfg.ENV_LABELS
        },
        "env_policy_metrics": {
            env_name: {
                "steps": int(len(rows)),
                "mean_q_gap": _safe_mean(np.asarray([row["q_gap"] for row in rows], dtype=np.float64)),
                "mean_max_q": _safe_mean(np.asarray([row["max_q"] for row in rows], dtype=np.float64)),
                "mean_abs_action_voltage_v": float(
                    np.mean(np.abs(np.asarray([row["action_voltage_v"] for row in rows], dtype=np.float64)))
                )
                if rows
                else 0.0,
            }
            for env_name, rows in env_rows.items()
        },
        "scenario_metrics": {
            name: {
                "force_freq_hz": float(next(row["force_freq_hz"] for row in policy_rows if row["scenario_name"] == name)),
                "force_waveform": str(next(row["force_waveform"] for row in policy_rows if row["scenario_name"] == name)),
                "force_noise_std": float(next(row["force_noise_std"] for row in policy_rows if row["scenario_name"] == name)),
                "force_noise_seed": int(next(row["force_noise_seed"] for row in policy_rows if row["scenario_name"] == name)),
                "tracking_rmse_mm": float(next(row["tracking_rmse_mm"] for row in episode_rows if row["scenario_name"] == name)),
                "transparency_rmse_w": float(next(row["transparency_rmse_w"] for row in episode_rows if row["scenario_name"] == name)),
                "episode_return": float(next(row["episode_return"] for row in episode_rows if row["scenario_name"] == name)),
                "mean_q_gap": _safe_mean(np.asarray([row["q_gap"] for row in policy_rows if row["scenario_name"] == name], dtype=np.float64)),
                "mean_abs_action_voltage_v": float(
                    np.mean(np.abs(np.asarray([row["action_voltage_v"] for row in policy_rows if row["scenario_name"] == name], dtype=np.float64)))
                ),
            }
            for name in scenario_names
        },
    }


def _plot_episode_metrics(rows: list[dict[str, Any]], out_path: Path) -> None:
    episodes = np.asarray([row["episode"] for row in rows], dtype=np.int64)
    returns = np.asarray([row["episode_return"] for row in rows], dtype=np.float64)
    tracking = np.asarray([row["tracking_rmse_mm"] for row in rows], dtype=np.float64)
    transparency = np.asarray([row["transparency_rmse_w"] for row in rows], dtype=np.float64)
    q_gap = np.asarray([row["mean_q_gap"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(episodes, returns, marker="o", lw=1.5, color="tab:blue")
    axes[0, 0].set_title("Episode return")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Return")

    axes[0, 1].plot(episodes, tracking, marker="o", lw=1.5, color="tab:orange")
    axes[0, 1].set_title("Tracking RMSE")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("RMSE [mm]")

    axes[1, 0].plot(episodes, transparency, marker="o", lw=1.5, color="tab:green")
    axes[1, 0].set_title("Transparency RMSE")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("RMSE [W]")

    axes[1, 1].plot(episodes, q_gap, marker="o", lw=1.5, color="tab:red")
    axes[1, 1].set_title("Mean policy Q-gap")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Q-gap")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(_fs_path(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_bars(summary: dict[str, Any], out_path: Path) -> None:
    labels = [
        "Return",
        "Track RMSE [mm]",
        "Transp RMSE [W]",
        "Mean |u_v| [V]",
        "Mean |du| [V]",
        "Mean Q-gap",
    ]
    values = [
        summary["mean_return"],
        summary["mean_tracking_rmse_mm"],
        summary["mean_transparency_rmse_w"],
        summary["mean_abs_u_v"],
        summary["mean_abs_delta_u_v"],
        summary["mean_q_gap"],
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"])
    ax.set_title("Greedy evaluation summary")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(_fs_path(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_policy_dashboard(
    policy_rows: list[dict[str, Any]],
    out_path: Path,
    env_switch_time: float,
) -> None:
    if not policy_rows:
        return

    action_levels = np.asarray(cfg.V_LEVELS, dtype=np.float64)
    overall = _action_fraction(policy_rows)
    skin = _action_fraction(policy_rows, env_name="skin")
    fat = _action_fraction(policy_rows, env_name="fat")
    first_episode = [row for row in policy_rows if row["episode"] == 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].bar([f"{v:.0f}" for v in action_levels], [overall[f"{v:.1f}"] for v in action_levels], color="tab:blue", alpha=0.85)
    axes[0, 0].set_title("Action usage across all test steps")
    axes[0, 0].set_xlabel("Voltage action [V]")
    axes[0, 0].set_ylabel("Fraction")

    x = np.arange(action_levels.size)
    width = 0.38
    axes[0, 1].bar(x - width / 2.0, [skin[f"{v:.1f}"] for v in action_levels], width=width, color="tab:blue", alpha=0.85, label="skin")
    axes[0, 1].bar(x + width / 2.0, [fat[f"{v:.1f}"] for v in action_levels], width=width, color="tab:orange", alpha=0.85, label="fat")
    axes[0, 1].set_xticks(x, [f"{v:.0f}" for v in action_levels])
    axes[0, 1].set_title("Action usage by environment")
    axes[0, 1].set_xlabel("Voltage action [V]")
    axes[0, 1].set_ylabel("Fraction")
    axes[0, 1].legend()

    t = np.asarray([row["time_s"] for row in first_episode], dtype=np.float64)
    u_v = np.asarray([row["action_voltage_v"] for row in first_episode], dtype=np.float64)
    q_gap = np.asarray([row["q_gap"] for row in first_episode], dtype=np.float64)
    max_q = np.asarray([row["max_q"] for row in first_episode], dtype=np.float64)

    axes[1, 0].plot(t, u_v, lw=1.6, color="tab:red")
    axes[1, 0].axvline(env_switch_time, color="0.35", ls="--", lw=1.1)
    axes[1, 0].set_title("Greedy action over time (episode 1)")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Voltage [V]")

    axes[1, 1].plot(t, q_gap, lw=1.6, color="tab:purple", label="Q-gap")
    axes[1, 1].plot(t, max_q, lw=1.2, color="tab:green", alpha=0.8, label="max Q")
    axes[1, 1].axvline(env_switch_time, color="0.35", ls="--", lw=1.1)
    axes[1, 1].set_title("Policy confidence over time (episode 1)")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(_fs_path(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_input_signal_dashboard(
    policy_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    out_path: Path,
    env_switch_time: float,
) -> None:
    scenario_names = [row["scenario_name"] for row in episode_rows]
    if not scenario_names:
        return

    n_items = len(scenario_names)
    n_cols = 2
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.1 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for idx, scenario_name in enumerate(scenario_names):
        ax = axes[idx // n_cols, idx % n_cols]
        episode_row = episode_rows[idx]
        scenario_rows = [row for row in policy_rows if row["scenario_name"] == scenario_name]
        t = np.asarray([row["time_s"] for row in scenario_rows], dtype=np.float64)
        force_input = np.asarray([row["force_input_n"] for row in scenario_rows], dtype=np.float64)
        force_nominal = np.asarray([row["force_nominal_n"] for row in scenario_rows], dtype=np.float64)
        force_noise = np.asarray([row["force_noise_n"] for row in scenario_rows], dtype=np.float64)

        ax.plot(t, force_input, lw=1.4, color="tab:blue", label="Noisy input")
        if np.any(np.abs(force_nominal) > 1e-12):
            ax.plot(t, force_nominal, lw=1.0, ls="--", color="0.35", alpha=0.9, label="Nominal")
        if np.any(np.abs(force_noise) > 1e-12):
            ax.plot(t, force_noise, lw=0.9, color="tab:orange", alpha=0.85, label="Noise")
        ax.axvline(env_switch_time, color="0.35", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(
            f"{scenario_name} | track {episode_row['tracking_rmse_mm']:.2f} mm | "
            f"transp {episode_row['transparency_rmse_w']:.3f} W"
        )
        ax.set_ylabel("Force [N]")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    for idx in range(n_items, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig(_fs_path(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scenario_dashboard(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows or len({row["scenario_name"] for row in rows}) <= 1:
        return

    labels = [row["scenario_name"] for row in rows]
    tracking = np.asarray([row["tracking_rmse_mm"] for row in rows], dtype=np.float64)
    transparency = np.asarray([row["transparency_rmse_w"] for row in rows], dtype=np.float64)
    q_gap = np.asarray([row["mean_q_gap"] for row in rows], dtype=np.float64)
    returns = np.asarray([row["episode_return"] for row in rows], dtype=np.float64)
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    axes[0, 0].bar(x, tracking, color="tab:orange")
    axes[0, 0].set_title("Tracking RMSE by scenario")
    axes[0, 0].set_ylabel("RMSE [mm]")

    axes[0, 1].bar(x, transparency, color="tab:green")
    axes[0, 1].set_title("Transparency RMSE by scenario")
    axes[0, 1].set_ylabel("RMSE [W]")

    axes[1, 0].bar(x, q_gap, color="tab:purple")
    axes[1, 0].set_title("Mean Q-gap by scenario")
    axes[1, 0].set_ylabel("Q-gap")

    axes[1, 1].bar(x, returns, color="tab:blue")
    axes[1, 1].set_title("Return by scenario")
    axes[1, 1].set_ylabel("Return")

    for ax in axes.ravel():
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(_fs_path(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _evaluate_saved_dqn(
    model_path: Path,
    n_episodes: int,
    seed: int,
    scenario_set: str | None,
    noise_std: float,
    disable_jerk: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], float]:
    model_summary = _load_summary_for_model(model_path)
    env_factory, reward_name, env_mode, master_input_mode, state_variant_name, obs_dim = _build_env_factory(
        model_summary,
        disable_jerk=disable_jerk,
    )
    scenarios = _scenario_plan(scenario_set, model_summary, noise_std)
    if scenarios is not None:
        n_episodes = len(scenarios)

    agent = DQNAgent(obs_dim=obs_dim, n_actions=cfg.N_ACTIONS, seed=seed)
    agent.load(str(model_path))

    episode_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    env_switch_time = float(cfg.ENV_SWITCH_TIME)
    old_eps = float(agent.epsilon)
    agent.epsilon = 0.0
    try:
        for ep in range(n_episodes):
            scenario = scenarios[ep] if scenarios is not None else {"name": f"episode_{ep + 1:02d}", "reset_options": {}}
            env = env_factory()
            obs, _ = env.reset(seed=seed + ep, options=scenario["reset_options"])
            base_env = env.base_env if hasattr(env, "base_env") else env
            env_switch_time = float(getattr(base_env, "env_switch_time", env_switch_time))
            force_freq_hz = float(getattr(base_env, "force_freq", cfg.FORCE_INPUT_FREQ))
            force_waveform = str(getattr(base_env, "force_waveform", "sine"))
            force_noise_std = float(getattr(base_env, "force_noise_std", 0.0))
            force_noise_seed = int(getattr(base_env, "force_noise_seed", 0))

            done = False
            terminated = False
            truncated = False
            ep_q_trace: list[np.ndarray] = []
            ep_actions: list[int] = []

            while not done:
                q_values = agent.q_values(obs)
                action = int(np.argmax(q_values))
                ep_q_trace.append(q_values)
                ep_actions.append(action)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            history = env.render() or {}
            ep_policy_rows = _policy_rows_for_episode(
                episode_idx=ep,
                history=history,
                action_indices=ep_actions,
                q_trace=ep_q_trace,
                reward_name=reward_name,
                env_mode=env_mode,
                master_input_mode=master_input_mode,
                scenario_name=str(scenario["name"]),
                force_freq_hz=force_freq_hz,
                force_waveform=force_waveform,
                force_noise_std=force_noise_std,
                force_noise_seed=force_noise_seed,
            )
            policy_rows.extend(ep_policy_rows)

            row = {
                "episode": ep + 1,
                "scenario_name": str(scenario["name"]),
                "force_freq_hz": force_freq_hz,
                "force_waveform": force_waveform,
                "force_noise_std": force_noise_std,
                "force_noise_seed": force_noise_seed,
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
                "reward_name": reward_name,
                "env_mode": env_mode,
                "master_input_mode": master_input_mode,
            }
            row.update(_episode_metrics(history, ep_policy_rows))
            episode_rows.append(row)
    finally:
        agent.epsilon = old_eps

    returns = np.array([row["episode_return"] for row in episode_rows], dtype=np.float64)
    track = np.array([row["tracking_rmse_mm"] for row in episode_rows], dtype=np.float64)
    transp = np.array([row["transparency_rmse_w"] for row in episode_rows], dtype=np.float64)
    mean_abs_u = np.array([row["mean_abs_u_v"] for row in episode_rows], dtype=np.float64)
    mean_abs_du = np.array([row["mean_abs_delta_u_v"] for row in episode_rows], dtype=np.float64)
    mean_q_gap = np.array([row["mean_q_gap"] for row in episode_rows], dtype=np.float64)
    mean_max_q = np.array([row["mean_max_q"] for row in episode_rows], dtype=np.float64)
    term = np.array([row["terminated"] for row in episode_rows], dtype=np.float64)

    aggregate = {
        "model_path": str(model_path),
        "episodes": int(n_episodes),
        "scenario_set": scenario_set or "baseline_repeat",
        "noise_std_n": float(noise_std) if scenario_set == "force_noise_10" else 0.0,
        "reward_name": reward_name,
        "env_mode": env_mode,
        "master_input_mode": master_input_mode,
        "state_variant_name": state_variant_name or "S0_baseline_full10",
        "obs_dim": int(obs_dim),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_tracking_rmse_mm": float(np.mean(track)),
        "std_tracking_rmse_mm": float(np.std(track)),
        "mean_transparency_rmse_w": float(np.mean(transp)),
        "std_transparency_rmse_w": float(np.std(transp)),
        "mean_abs_u_v": float(np.mean(mean_abs_u)),
        "mean_abs_delta_u_v": float(np.mean(mean_abs_du)),
        "mean_q_gap": float(np.mean(mean_q_gap)),
        "mean_max_q": float(np.mean(mean_max_q)),
        "terminated_fraction": float(np.mean(term)),
    }
    policy_summary = _policy_summary(policy_rows, episode_rows)
    return episode_rows, aggregate, policy_rows, policy_summary, env_switch_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved DQN checkpoint for greedy test runs.")
    parser.add_argument("--model-path", required=True, help="Path to the saved dqn_model.pt checkpoint.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of greedy test episodes to run.")
    parser.add_argument("--seed", type=int, default=70_000, help="Base random seed for test episodes.")
    parser.add_argument(
        "--scenario-set",
        choices=["force_generalization_10", "force_square_10", "force_noise_10"],
        default=None,
        help="Optional built-in signal transfer sweep. Overrides --episodes to the scenario count.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.5,
        help="Noise standard deviation in Newtons for the force_noise_10 sweep.",
    )
    parser.add_argument(
        "--disable-jerk",
        action="store_true",
        help=(
            "Evaluate with the same tracking/transparency reward weights but zero jerk penalty. "
            "This rescales evaluation reward only; it does not retrain the saved policy."
        ),
    )
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model_path)
    rows, summary, policy_rows, policy_summary, env_switch_time = _evaluate_saved_dqn(
        model_path=model_path,
        n_episodes=args.episodes,
        seed=args.seed,
        scenario_set=args.scenario_set,
        noise_std=args.noise_std,
        disable_jerk=args.disable_jerk,
    )

    out_dir = model_path.parent.parent / "temp_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"test_{len(rows)}ep" if not args.scenario_set else args.scenario_set
    if args.scenario_set == "force_noise_10":
        noise_label = f"{args.noise_std:.2f}".replace(".", "p")
        prefix = f"{prefix}_n{noise_label}"
    if args.disable_jerk:
        prefix = f"{prefix}_nojerk"
    csv_path = out_dir / f"{prefix}_metrics.csv"
    json_path = out_dir / f"{prefix}_summary.json"
    policy_csv_path = out_dir / f"{prefix}_policy_step_metrics.csv"
    policy_json_path = out_dir / f"{prefix}_policy_summary.json"

    with open(_fs_path(csv_path), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(_fs_path(json_path), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    with open(_fs_path(policy_csv_path), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(policy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(policy_rows)

    with open(_fs_path(policy_json_path), "w", encoding="utf-8") as fh:
        json.dump(policy_summary, fh, indent=2)

    _plot_episode_metrics(rows, out_dir / f"{prefix}_metrics_dashboard.png")
    _plot_summary_bars(summary, out_dir / f"{prefix}_summary_bars.png")
    _plot_policy_dashboard(policy_rows, out_dir / f"{prefix}_policy_dashboard.png", env_switch_time)
    _plot_scenario_dashboard(rows, out_dir / f"{prefix}_scenario_dashboard.png")
    _plot_input_signal_dashboard(policy_rows, rows, out_dir / f"{prefix}_input_signal_dashboard.png", env_switch_time)

    print(f"Saved per-episode metrics to: {csv_path}")
    print(f"Saved summary to: {json_path}")
    print(f"Saved per-step policy metrics to: {policy_csv_path}")
    print(f"Saved policy summary to: {policy_json_path}")
    print("")
    print("Aggregate metrics:")
    print(f"  scenario set:           {summary['scenario_set']}")
    if summary.get("noise_std_n", 0.0) > 0.0:
        print(f"  noise std:              {summary['noise_std_n']:.3f} N")
    print(f"  mean return:            {summary['mean_return']:+.6f} +/- {summary['std_return']:.6f}")
    print(f"  tracking RMSE:          {summary['mean_tracking_rmse_mm']:.3f} +/- {summary['std_tracking_rmse_mm']:.3f} mm")
    print(f"  transparency RMSE:      {summary['mean_transparency_rmse_w']:.6f} +/- {summary['std_transparency_rmse_w']:.6f} W")
    print(f"  mean |u_v|:             {summary['mean_abs_u_v']:.6f} V")
    print(f"  mean |delta u_v|:       {summary['mean_abs_delta_u_v']:.6f} V")
    print(f"  mean Q-gap:             {summary['mean_q_gap']:.6f}")
    print(f"  mean max Q:             {summary['mean_max_q']:.6f}")
    print(f"  terminated fraction:    {summary['terminated_fraction']:.3f}")


if __name__ == "__main__":
    main()
