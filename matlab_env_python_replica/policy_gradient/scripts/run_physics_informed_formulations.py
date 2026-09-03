"""CLI study comparing physics-informed PPO formulations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL.matlab_env_python_replica.config import config as cfg
    from TeleopWithRL.matlab_env_python_replica.policy_gradient.paths import suite_root as policy_gradient_suite_root
    from TeleopWithRL.matlab_env_python_replica.common.cli import replica_env_kwargs_from_args
    from TeleopWithRL.matlab_env_python_replica.environment.simuoriginal_replica import FE_MODE_DYNAMICS
    from TeleopWithRL.matlab_env_python_replica.common.study_utils import (
        history_array,
        save_history_npz,
        save_json,
        transparency_ratio_array,
    )
    from TeleopWithRL.matlab_env_python_replica.dqn.state_variants import build_custom_dqn_state_variant_from_spec
    from TeleopWithRL.matlab_env_python_replica.policy_gradient.training import (
        PG_ALGO_PPO_CONTINUOUS,
        build_policy_gradient_env_factory,
        evaluate_policy_gradient,
        load_reset_options_json,
        require_sb3,
        save_policy_gradient_visuals,
        train_policy_gradient_variant,
    )
    from TeleopWithRL.matlab_env_python_replica.common.focused_evaluation import run_focused_evaluation
    from TeleopWithRL.matlab_env_python_replica.common.rewarding import (
        DEFAULT_ACTION_DELTA_SCALE_V,
        DEFAULT_ACTION_SCALE_V,
        DEFAULT_TRACKING_SCALE_M,
        DEFAULT_VELOCITY_ERROR_SCALE_MPS,
        reward_variant_from_spec,
    )
else:
    from ...config import config as cfg
    from ..paths import suite_root as policy_gradient_suite_root
    from ...common.cli import replica_env_kwargs_from_args
    from ...environment.simuoriginal_replica import FE_MODE_DYNAMICS
    from ...common.study_utils import history_array, save_history_npz, save_json, transparency_ratio_array
    from ...dqn.state_variants import build_custom_dqn_state_variant_from_spec
    from ..training import (
        PG_ALGO_PPO_CONTINUOUS,
        build_policy_gradient_env_factory,
        evaluate_policy_gradient,
        load_reset_options_json,
        require_sb3,
        save_policy_gradient_visuals,
        train_policy_gradient_variant,
    )
    from ...common.focused_evaluation import run_focused_evaluation
    from ...common.rewarding import (
        DEFAULT_ACTION_DELTA_SCALE_V,
        DEFAULT_ACTION_SCALE_V,
        DEFAULT_TRACKING_SCALE_M,
        DEFAULT_VELOCITY_ERROR_SCALE_MPS,
        reward_variant_from_spec,
    )


@dataclass(frozen=True)
class Formulation:
    """Definition of one physics-informed observation/reward formulation."""

    key: str
    label: str
    state_features: tuple[str, ...]
    extra_terms: tuple[dict[str, Any], ...]
    note: str


BASELINE_FEATURES = ("x_m", "x_s", "v_m", "v_s", "u_v")


def _acceleration_scale() -> float:
    return max(float(cfg.OBS_SCALE_VEL) / max(float(cfg.RL_DT), 1e-9), 1e-6)


def _term(name: str, source: str, weight: float, scale_name: str) -> dict[str, Any]:
    return {
        "name": name,
        "source": source,
        "shape": "square",
        "sign": "penalty",
        "weight": float(weight),
        "scale_name": scale_name,
    }


BASE_REWARD_TERMS = (
    _term("tracking_base", "pos_error", 40.0, "tracking_error_m"),
    _term("control_effort", "u_v", 0.01, "action_voltage_v"),
)


FORMULATIONS = (
    Formulation(
        "F0_baseline",
        "Baseline",
        BASELINE_FEATURES,
        (),
        "x_m, x_s, xdot_m, xdot_s, u_{t-1}; baseline tracking plus effort reward.",
    ),
    Formulation(
        "F1_error_state_reward",
        "Add Error",
        (*BASELINE_FEATURES, "tracking_error"),
        (_term("physics_error", "pos_error", 10.0, "tracking_error_m"),),
        "Adds e = x_m - x_s to observation and an extra normalized e^2 reward penalty.",
    ),
    Formulation(
        "F2_error_dot_state_reward",
        "Add Error Dot",
        (*BASELINE_FEATURES, "tracking_error", "velocity_error"),
        (
            _term("physics_error", "pos_error", 10.0, "tracking_error_m"),
            _term("physics_error_dot", "velocity_error", 2.0, "velocity_error_mps"),
        ),
        "Adds e and edot to observation and reward.",
    ),
    Formulation(
        "F3_error_ddot_state_reward",
        "Add Error DDot",
        (*BASELINE_FEATURES, "tracking_error", "velocity_error", "acceleration_error"),
        (
            _term("physics_error", "pos_error", 10.0, "tracking_error_m"),
            _term("physics_error_dot", "velocity_error", 2.0, "velocity_error_mps"),
            _term("physics_error_ddot", "acceleration_error", 0.5, "acceleration_error_mps2"),
        ),
        "Adds e, edot, and eddot to observation and reward.",
    ),
    Formulation(
        "F4_accel_state",
        "Accel State",
        (*BASELINE_FEATURES, "x_m_ddot", "x_s_ddot"),
        (),
        "Adds xddot_m and xddot_s to observation only.",
    ),
    Formulation(
        "F5_accel_state_reward",
        "Accel State + Reward",
        (*BASELINE_FEATURES, "x_m_ddot", "x_s_ddot"),
        (_term("acceleration_error", "acceleration_error", 0.5, "acceleration_error_mps2"),),
        "Adds xddot_m and xddot_s to observation plus an acceleration-error reward term.",
    ),
    Formulation(
        "F6_effort_plus_delta_u",
        "Effort + Delta U",
        BASELINE_FEATURES,
        (_term("smooth_delta_u", "action_delta", 0.05, "action_delta_voltage_v"),),
        "Baseline state and reward plus smoothness penalty delta_u = u_t - u_{t-1}.",
    ),
)


SUMMARY_FIELDS = (
    "key",
    "label",
    "obs_dim",
    "state_features",
    "reward_terms",
    "total_timesteps",
    "model_path",
    "mean_reward",
    "tracking_rmse_m",
    "tracking_mae_m",
    "tracking_max_abs_m",
    "velocity_error_rmse_mps",
    "acceleration_error_rmse_mps2",
    "transparency_rmse_w",
    "transparency_ratio_mean",
    "transparency_ratio_rmse",
    "transparency_ratio_error_rmse",
    "mean_abs_u_v",
    "rms_u_v",
    "control_energy_v2_s",
    "mean_abs_delta_u_v",
    "rms_delta_u_v",
    "saturation_fraction",
    "completed_episode_rate",
    "invalid_episode_rate",
    "tensorboard_dir",
    "out_dir",
    "focused_scenario_count",
    "focused_tracking_rmse_mm",
    "focused_post_contact_rmse_mm",
    "focused_transparency_rmse_w",
    "focused_transparency_ratio_median",
    "focused_transparency_ratio_error_rmse",
    "focused_transparency_ratio_valid_fraction",
    "focused_transparency_ratio_within_20pct",
    "focused_rms_u_v",
    "focused_mean_abs_delta_u_v",
    "focused_mean_abs_delta2_u_v",
    "focused_failure_rate",
    "note",
    "train_requested_episodes",
    "actual_train_timesteps",
    "parallel_envs",
    "vec_env_type",
    "resolved_vec_env_type",
    "ppo_n_steps",
    "ppo_batch_size",
    "ppo_n_epochs",
    "ppo_gamma",
    "ppo_ent_coef",
    "eval_every_episodes",
    "test_episodes",
    "train_signal_count",
    "eval_signal_count",
)


def build_reward_spec(formulation: Formulation) -> dict[str, Any]:
    """Build the JSON-serializable reward specification for a formulation."""
    terms = [dict(term) for term in BASE_REWARD_TERMS]
    terms.extend(dict(term) for term in formulation.extra_terms)
    return {
        "name": f"{formulation.key}_reward",
        "description": f"Physics-informed reward for {formulation.key}.",
        "scale_catalog": {
            "tracking_error_m": {"value": DEFAULT_TRACKING_SCALE_M, "unit": "m"},
            "tracking_failure_threshold_m": {"value": float(cfg.POS_ERROR_FAIL_THRESHOLD), "unit": "m"},
            "velocity_error_mps": {"value": DEFAULT_VELOCITY_ERROR_SCALE_MPS, "unit": "m/s"},
            "acceleration_error_mps2": {"value": _acceleration_scale(), "unit": "m/s^2"},
            "action_voltage_v": {"value": DEFAULT_ACTION_SCALE_V, "unit": "V"},
            "action_delta_voltage_v": {"value": DEFAULT_ACTION_DELTA_SCALE_V, "unit": "V"},
        },
        "terms": terms,
        "weights": {
            "tracking": 0.0,
            "transparency": 0.0,
            "velocity": 0.0,
            "force_difference": 0.0,
            "effort": 0.0,
            "jerk": 0.0,
        },
        "penalties": {
            "stroke_limit": 250.0,
            "invalid_state": 100.0,
            "tracking_error_fail": 1000.0,
            "edge_buffer_m": 0.0,
            "low_force_threshold_n": 0.0,
        },
    }


def build_state_spec(formulation: Formulation) -> dict[str, Any]:
    """Build the JSON-serializable state specification for a formulation."""
    return {
        "name": f"{formulation.key}_state",
        "description": formulation.note,
        "selected_features": list(formulation.state_features),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the fixed-column formulation comparison table to CSV."""
    path = Path(path)
    os.makedirs(_long_path(path.parent), exist_ok=True)
    with open(_long_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON object used by the formulation study."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_training_npz(run_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(run_dir) / "l" / "train.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _load_episode_npz(run_dir: str | Path, filename: str = "test.npz") -> dict[str, Any]:
    path = Path(run_dir) / "e" / filename
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _use_symlog_if_needed(ax, values: list[float] | np.ndarray, *, linthresh: float = 1.0) -> None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size and float(np.nanmax(np.abs(arr))) > 100.0:
        ax.set_yscale("symlog", linthresh=linthresh)


def plot_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    """Write summary bars, learning curves, and ratio rollout plots."""
    if not rows:
        return
    labels = [str(row["key"]).replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    metrics = [
        ("tracking_rmse_m", "Tracking RMSE [mm]", 1000.0),
        ("transparency_rmse_w", "Transparency RMSE [W]", 1.0),
        ("transparency_ratio_mean", "Actual transparency ratio mean: (F_h/v_m)/(F_e/v_s)", 1.0),
        ("rms_u_v", "RMS u_v [V]", 1.0),
        ("mean_abs_delta_u_v", "Mean |delta u| [V]", 1.0),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 17), constrained_layout=True)
    for ax, (key, ylabel, scale) in zip(axes, metrics):
        values = [float(row.get(key, 0.0)) * scale for row in rows]
        ax.bar(x, values, color="tab:blue", alpha=0.78)
        if key == "transparency_ratio_mean":
            ax.axhline(1.0, color="tab:red", lw=1.2, ls="--", alpha=0.85, label="ideal ratio = 1")
            _use_symlog_if_needed(ax, values)
            ax.legend(loc="best", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Physics-informed formulation comparison")
    fig.savefig(root / "summary_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    for row in rows:
        train = _load_training_npz(row["out_dir"])
        if not train:
            continue
        eval_steps = train.get("eval_steps", np.asarray([], dtype=np.float64))
        if eval_steps.size == 0:
            continue
        axes[0].plot(eval_steps, train.get("eval_tracking_rmse", np.asarray([])) * 1000.0, marker="o", label=row["key"])
        axes[1].plot(eval_steps, train.get("eval_transparency_rmse", np.asarray([])), marker="o", label=row["key"])
        axes[2].plot(eval_steps, train.get("eval_mean_reward", np.asarray([])), marker="o", label=row["key"])
    axes[0].set_ylabel("Eval tracking RMSE [mm]")
    axes[1].set_ylabel("Eval transparency RMSE [W]")
    all_eval_transparency = [
        value
        for row in rows
        for value in np.asarray(_load_training_npz(row["out_dir"]).get("eval_transparency_rmse", []), dtype=np.float64).tolist()
    ]
    _use_symlog_if_needed(axes[1], all_eval_transparency)
    axes[2].set_ylabel("Eval return")
    axes[2].set_xlabel("Completed training episodes")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.savefig(root / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_transparency_ratio_rollouts(root, rows)


def plot_transparency_ratio_rollouts(root: Path, rows: list[dict[str, Any]]) -> None:
    """Write the actual force/velocity transparency ratio for each formulation."""
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    plotted = False
    all_values: list[float] = []
    for row in rows:
        history = _load_episode_npz(row["out_dir"])
        if not history:
            continue
        time_s = history_array(history, "time", dtype=np.float64)
        ratio = transparency_ratio_array(history)
        n = min(time_s.size, ratio.size)
        if n == 0:
            continue
        time_s = time_s[:n]
        ratio = ratio[:n]
        finite = np.isfinite(time_s) & np.isfinite(ratio)
        if not np.any(finite):
            continue
        time_s = time_s[finite]
        ratio = ratio[finite]
        ax.plot(time_s, ratio, lw=1.35, alpha=0.85, label=str(row["key"]))
        all_values.extend(ratio.tolist())
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.8, label="ideal ratio = 1")
    _use_symlog_if_needed(ax, all_values)
    ax.set_title("Actual transparency ratio over evaluation rollouts")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("(F_h/v_m) / (F_e/v_s)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(root / "transparency_ratio_rollouts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(root: Path, rows: list[dict[str, Any]], tensorboard_root: Path) -> None:
    """Write a human-readable formulation summary with units and metrics."""
    lines = [
        "# Physics-Informed Formulation Study",
        "",
        f"TensorBoard root: `{tensorboard_root}`",
        "",
        "The learning transparency term is the old impedance/power error `F_e*v_m - F_h*v_s` in W. The actual force/velocity ratio `(F_h/v_m)/(F_e/v_s)` is also reported; its ideal value is `1.0`.",
        "",
        "| Formulation | Track RMSE mm | Transp RMSE W | Actual transp ratio mean | Ratio error RMSE | RMS u V | Mean abs(delta u) V | Completed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {key} | {track:.3f} | {transp:.4f} | {ratio:.4g} | {trans_err:.4g} | {rms_u:.3f} | {du:.3f} | {done:.2f} |".format(
                key=row["key"],
                track=1000.0 * float(row.get("tracking_rmse_m", 0.0)),
                transp=float(row.get("transparency_rmse_w", 0.0)),
                ratio=float(row.get("transparency_ratio_mean", 0.0)),
                trans_err=float(row.get("transparency_ratio_error_rmse", 0.0)),
                rms_u=float(row.get("rms_u_v", 0.0)),
                du=float(row.get("mean_abs_delta_u_v", 0.0)),
                done=float(row.get("completed_episode_rate", 0.0)),
            )
        )
    lines.extend([
        "",
        "Generated artifacts:",
        "",
        "- `summary.csv`: flat metric table",
        "- `summary_bars.png`: final metric comparison with actual transparency ratio",
        "- `learning_curves.png`: evaluation checkpoints across training",
        "- `transparency_ratio_rollouts.png`: actual `(F_h/v_m)/(F_e/v_s)` over reevaluated rollouts",
    ])
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def aggregate_focused_metrics(path: Path) -> dict[str, float]:
    """Aggregate focused-evaluation scenario rows into one formulation row."""
    try:
        with open(_long_path(path), "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError:
        return {}
    if not rows:
        return {}

    def mean(key: str) -> float:
        return float(np.mean([_float(row, key) for row in rows]))

    return {
        "focused_scenario_count": float(len(rows)),
        "focused_tracking_rmse_mm": 1000.0 * mean("rms_error_m"),
        "focused_post_contact_rmse_mm": 1000.0 * mean("post_contact_rms_error_m"),
        "focused_transparency_rmse_w": mean("transparency_rmse_w"),
        "focused_transparency_ratio_median": mean("transparency_ratio_median"),
        "focused_transparency_ratio_error_rmse": mean("transparency_ratio_error_rmse"),
        "focused_transparency_ratio_valid_fraction": mean("transparency_ratio_valid_fraction"),
        "focused_transparency_ratio_within_20pct": mean("transparency_ratio_within_20pct"),
        "focused_rms_u_v": mean("rms_u_v"),
        "focused_mean_abs_delta_u_v": mean("control_smoothness_mean_abs_delta_v"),
        "focused_mean_abs_delta2_u_v": mean("control_smoothness_mean_abs_delta2_v"),
        "focused_failure_rate": mean("failure_flag"),
    }


def collect_available_rows(
    root: Path,
    formulations: list[Formulation] | tuple[Formulation, ...],
    current_rows: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load completed formulation summaries, optionally reusing current rows."""
    current_rows = dict(current_rows or {})
    rows: list[dict[str, Any]] = []
    for formulation in formulations:
        if formulation.key in current_rows:
            rows.append(current_rows[formulation.key])
            continue
        summary_path = _summary_path_for(root, formulation)
        if not summary_path.exists():
            continue
        focused_csv = root / formulation.key / "focused_eval" / "focused_eval_metrics.csv"
        rows.append(row_from_summary(formulation, load_json(summary_path), aggregate_focused_metrics(focused_csv)))
    return rows


def _long_path(path: str | Path) -> str | Path:
    path = Path(path)
    if os.name != "nt":
        return path
    resolved = path.resolve()
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text.lstrip("\\")
    return "\\\\?\\" + text


def _file_exists(path: str | Path) -> bool:
    try:
        with open(_long_path(path), "rb"):
            return True
    except FileNotFoundError:
        return False


def row_from_summary(
    formulation: Formulation,
    summary: dict[str, Any],
    focused: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Normalize one saved formulation summary into the study-table schema."""
    eval_metrics = dict(summary.get("eval_metrics") or {})
    row = {
        "key": formulation.key,
        "label": formulation.label,
        "obs_dim": int(summary.get("obs_dim", len(formulation.state_features))),
        "state_features": " ".join(formulation.state_features),
        "reward_terms": " ".join(term["name"] for term in (BASE_REWARD_TERMS + formulation.extra_terms)),
        "total_timesteps": int(summary.get("total_timesteps", 0) or 0),
        "actual_train_timesteps": int(summary.get("actual_train_timesteps", summary.get("total_timesteps", 0)) or 0),
        "train_requested_episodes": int(summary.get("total_episodes", 0) or 0),
        "parallel_envs": int(summary.get("parallel_envs", 0) or 0),
        "vec_env_type": str(summary.get("vec_env_type", "")),
        "resolved_vec_env_type": str(summary.get("resolved_vec_env_type", "")),
        "ppo_n_steps": int(summary.get("ppo_n_steps", 0) or 0),
        "ppo_batch_size": int(summary.get("ppo_batch_size", 0) or 0),
        "ppo_n_epochs": int(summary.get("ppo_n_epochs", 0) or 0),
        "ppo_gamma": float(summary.get("ppo_gamma", 0.0) or 0.0),
        "ppo_ent_coef": float(summary.get("ppo_ent_coef", 0.0) or 0.0),
        "eval_every_episodes": int(summary.get("eval_every_episodes", 0) or 0),
        "test_episodes": int(summary.get("test_episodes", 0) or 0),
        "train_signal_count": int(summary.get("train_signal_count", 0) or 0),
        "eval_signal_count": int(summary.get("eval_signal_count", 0) or 0),
        "model_path": str(summary.get("model_path", "")),
        "out_dir": str(summary.get("out_dir", "")),
        "note": formulation.note,
    }
    for key in SUMMARY_FIELDS:
        if key in row:
            continue
        if key in summary:
            row[key] = summary[key]
        elif key in eval_metrics:
            row[key] = eval_metrics[key]
    row["completed_episode_rate"] = eval_metrics.get("completed_episode_rate", summary.get("completed_episode_rate", 0.0))
    if focused:
        row.update(focused)
    return row


def _summary_path_for(root: Path, formulation: Formulation) -> Path:
    return root / formulation.key / "ppo" / "l" / "summary.json"


def _out_dir_for(root: Path, formulation: Formulation) -> Path:
    return root / formulation.key / "ppo"


def _env_kwargs_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    reset_options = dict(summary.get("reset_options") or {})
    env_kwargs: dict[str, Any] = {
        "episode_duration": float(summary.get("episode_duration", cfg.EPISODE_DURATION)),
        "env_switch_time": float(summary.get("env_switch_time", cfg.PAPER_ENV_SWITCH_TIME)),
        "terminate_on_error": bool(summary.get("terminate_on_error", True)),
        "legacy_baseline_env": bool(reset_options.get("legacy_baseline_env", False)),
        "enforce_stroke_limit": bool(summary.get("enforce_stroke_limit", True)),
        "stroke_limit_mode": str(summary.get("stroke_limit_mode", reset_options.get("stroke_limit_mode", "clamp"))),
        "reset_position_mode": str(reset_options.get("reset_position_mode", "midpoint")),
        "reset_options": reset_options,
    }
    if summary.get("action_levels") is not None:
        env_kwargs["action_levels"] = [float(level) for level in summary["action_levels"]]
    return env_kwargs


def _load_ppo_model(model_path: str | Path, summary: dict[str, Any] | None = None):
    require_sb3()
    from stable_baselines3 import PPO

    _install_numpy_pickle_compat()
    custom_objects = _sb3_space_custom_objects(summary or {})
    return PPO.load(str(model_path), custom_objects=custom_objects)


def _sb3_space_custom_objects(summary: dict[str, Any]) -> dict[str, Any]:
    try:
        from gymnasium import spaces
    except Exception:
        return {}
    obs_dim = int(summary.get("obs_dim", 0) or 0)
    if obs_dim <= 0:
        return {}
    action_levels = np.asarray(summary.get("action_levels", cfg.V_LEVELS), dtype=np.float32).reshape(-1)
    action_limit = float(np.max(np.abs(action_levels))) if action_levels.size else 5.0
    return {
        "observation_space": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        "action_space": spaces.Box(
            low=np.asarray([-action_limit], dtype=np.float32),
            high=np.asarray([action_limit], dtype=np.float32),
            dtype=np.float32,
        ),
    }


def _install_numpy_pickle_compat() -> None:
    try:
        import numpy.core as np_core
        import numpy.core.multiarray as np_multiarray
        import numpy.core.numeric as np_numeric
        import numpy.random._pickle as np_random_pickle
    except Exception:
        return
    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)
    original_ctor = getattr(np_random_pickle, "__bit_generator_ctor", None)
    bit_generators = getattr(np_random_pickle, "BitGenerators", {})
    if original_ctor is None or getattr(original_ctor, "_teleop_compat", False):
        return

    def _bit_generator_ctor_compat(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        if bit_generator_name in bit_generators:
            return bit_generators[bit_generator_name]()
        return original_ctor(bit_generator_name)

    _bit_generator_ctor_compat._teleop_compat = True  # type: ignore[attr-defined]
    np_random_pickle.__bit_generator_ctor = _bit_generator_ctor_compat


def _update_summary_with_eval(summary: dict[str, Any], eval_metrics: dict[str, float], *, test_episodes: int) -> dict[str, Any]:
    summary = dict(summary)
    summary["test_episodes"] = int(test_episodes)
    summary["evaluation_history_mode"] = "mean_over_test_episodes_eval_only"
    summary["eval_metrics"] = dict(eval_metrics)
    summary["mean_reward"] = float(eval_metrics.get("mean_reward", summary.get("mean_reward", 0.0)))
    summary["tracking_rmse_m"] = float(eval_metrics.get("tracking_rmse_m", summary.get("tracking_rmse_m", 0.0)))
    summary["transparency_rmse_w"] = float(eval_metrics.get("transparency_rmse_w", summary.get("transparency_rmse_w", 0.0)))
    summary["pre_switch_tracking_rmse_m"] = float(
        eval_metrics.get("pre_switch_tracking_rmse_m", summary.get("pre_switch_tracking_rmse_m", 0.0))
    )
    summary["post_switch_tracking_rmse_m"] = float(
        eval_metrics.get("post_switch_tracking_rmse_m", summary.get("post_switch_tracking_rmse_m", 0.0))
    )
    summary["pre_switch_transparency_rmse_w"] = float(
        eval_metrics.get("pre_switch_transparency_rmse_w", summary.get("pre_switch_transparency_rmse_w", 0.0))
    )
    summary["post_switch_transparency_rmse_w"] = float(
        eval_metrics.get("post_switch_transparency_rmse_w", summary.get("post_switch_transparency_rmse_w", 0.0))
    )
    summary["invalid_episode_rate"] = float(eval_metrics.get("invalid_episode", summary.get("invalid_episode_rate", 0.0)))
    for key in SUMMARY_FIELDS:
        if key in eval_metrics:
            summary[key] = float(eval_metrics[key])
    for key in (
        "tracking_mae_m",
        "tracking_max_abs_m",
        "velocity_error_rmse_mps",
        "acceleration_error_rmse_mps2",
        "transparency_ratio_mean",
        "transparency_ratio_rmse",
        "transparency_ratio_error_rmse",
        "mean_abs_u_v",
        "rms_u_v",
        "control_energy_v2_s",
        "max_abs_u_v",
        "saturation_fraction",
        "mean_abs_delta_u_v",
        "rms_delta_u_v",
        "max_abs_delta_u_v",
        "completed_episode_rate",
    ):
        if key in eval_metrics:
            summary[key] = float(eval_metrics[key])
    return summary


def reevaluate_existing(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Re-evaluate existing formulation models and refresh study artifacts."""
    root = policy_gradient_suite_root(args.fe_mode, args.study_name)
    specs_root = root / "specs"
    selected_formulations = [
        formulation
        for formulation in FORMULATIONS
        if not args.only or formulation.key in set(str(key) for key in args.only)
    ]
    if not selected_formulations:
        raise ValueError(f"No formulations selected by --only {args.only!r}")

    current_rows: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = collect_available_rows(root, FORMULATIONS, current_rows)
    for index, formulation in enumerate(selected_formulations, start=1):
        summary_path = _summary_path_for(root, formulation)
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing trained summary for {formulation.key}: {summary_path}")

        summary = load_json(summary_path)
        model_path = Path(str(summary.get("model_path") or (_out_dir_for(root, formulation) / "m" / "ppo_model.zip")))
        if not model_path.exists():
            raise FileNotFoundError(f"Missing PPO model for {formulation.key}: {model_path}")

        state_spec_path = specs_root / f"{formulation.key}_state.json"
        reward_spec_path = specs_root / f"{formulation.key}_reward.json"
        state_spec = load_json(state_spec_path) if state_spec_path.exists() else build_state_spec(formulation)
        reward_spec = load_json(reward_spec_path) if reward_spec_path.exists() else build_reward_spec(formulation)

        env_kwargs = _env_kwargs_from_summary(summary)
        state_variant = build_custom_dqn_state_variant_from_spec(state_spec)
        reward_variant = reward_variant_from_spec(reward_spec)
        env_factory = build_policy_gradient_env_factory(
            algo=str(summary.get("algo", PG_ALGO_PPO_CONTINUOUS)),
            env_mode=str(summary.get("env_mode", args.env_mode)),
            env_kwargs=env_kwargs,
            state_variant=state_variant,
            reward_variant=reward_variant,
        )
        print(f"[{index}/{len(selected_formulations)}] evaluate {formulation.key}: {model_path}", flush=True)
        model = _load_ppo_model(model_path, summary)
        eval_metrics, history = evaluate_policy_gradient(
            model,
            env_factory,
            n_episodes=args.test_episodes,
            seed_offset=args.eval_seed_offset,
            reset_options_schedule=summary.get("eval_reset_options_schedule") or None,
        )

        out_dir = _out_dir_for(root, formulation)
        save_history_npz(history, out_dir / "e" / "test.npz")
        save_history_npz(history, out_dir / "e" / "test_reeval.npz")
        save_policy_gradient_visuals(
            history,
            out_dir / "p",
            f"{formulation.key}_{formulation.label}_reeval",
            env_switch_time=float(env_kwargs["env_switch_time"]),
            action_mode="continuous",
            action_levels=summary.get("action_levels", cfg.V_LEVELS),
        )

        updated_summary = _update_summary_with_eval(summary, eval_metrics, test_episodes=args.test_episodes)
        save_json(summary_path, updated_summary)
        current_rows[formulation.key] = row_from_summary(formulation, updated_summary)
        rows = collect_available_rows(root, FORMULATIONS, current_rows)
        write_summary_csv(root / "summary.csv", rows)
        plot_summary(root, rows)

    tensorboard_root = Path.home() / "AppData" / "Local" / "TeleopWithRL_tb" / root.relative_to(Path(__file__).resolve().parents[2])
    rows = collect_available_rows(root, FORMULATIONS, current_rows)
    write_summary_csv(root / "summary.csv", rows)
    plot_summary(root, rows)
    write_summary_markdown(root, rows, tensorboard_root)
    save_json(root / "study_manifest.json", {"rows": rows, "tensorboard_root": str(tensorboard_root)})
    print(f"summary_csv={root / 'summary.csv'}", flush=True)
    print(f"summary_md={root / 'summary.md'}", flush=True)
    print(f"transparency_ratio_rollouts={root / 'transparency_ratio_rollouts.png'}", flush=True)
    print(f"tensorboard_root={tensorboard_root}", flush=True)
    return rows


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run training or re-evaluation according to parsed CLI arguments."""
    if args.eval_only:
        return reevaluate_existing(args)

    root = policy_gradient_suite_root(args.fe_mode, args.study_name)
    root.mkdir(parents=True, exist_ok=True)
    specs_root = root / "specs"
    specs_root.mkdir(parents=True, exist_ok=True)
    train_reset_options_pool = load_reset_options_json(args.train_reset_options_json)
    eval_reset_options_schedule = load_reset_options_json(args.eval_reset_options_json)

    env_args = SimpleNamespace(
        episode_duration=args.episode_duration,
        env_switch_time=args.env_switch_time,
        disable_terminate_on_error=args.disable_terminate_on_error,
        legacy_baseline_env=False,
        disable_stroke_limit=False,
        stroke_limit_mode=args.stroke_limit_mode,
        reset_position_mode=args.reset_position_mode,
        action_levels=None,
        force_amp=args.force_amp,
        force_bias=args.force_bias,
        force_freq_rad=args.force_freq_rad,
        force_phase=args.force_phase,
        force_waveform=args.force_waveform,
        fe_mode=args.fe_mode,
    )
    env_kwargs = replica_env_kwargs_from_args(env_args)

    selected_formulations = tuple(
        formulation
        for formulation in FORMULATIONS
        if not args.only or formulation.key in set(str(key) for key in args.only)
    )
    if not selected_formulations:
        known = ", ".join(formulation.key for formulation in FORMULATIONS)
        raise ValueError(f"No formulations selected by --only {args.only!r}. Known formulations: {known}")

    current_rows: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = collect_available_rows(root, FORMULATIONS, current_rows)
    for index, formulation in enumerate(selected_formulations, start=1):
        state_spec = build_state_spec(formulation)
        reward_spec = build_reward_spec(formulation)
        state_spec_path = specs_root / f"{formulation.key}_state.json"
        reward_spec_path = specs_root / f"{formulation.key}_reward.json"
        save_json(state_spec_path, state_spec)
        save_json(reward_spec_path, reward_spec)

        out_dir = root / formulation.key / "ppo"
        summary_path = out_dir / "l" / "summary.json"
        if args.skip_existing and summary_path.exists():
            summary = load_json(summary_path)
            print(f"[{index}/{len(selected_formulations)}] skip existing {formulation.key}: {summary_path}", flush=True)
        else:
            print(f"[{index}/{len(selected_formulations)}] train {formulation.key}: {formulation.note}", flush=True)
            result = train_policy_gradient_variant(
                algo=PG_ALGO_PPO_CONTINUOUS,
                out_dir=out_dir,
                env_mode=args.env_mode,
                env_kwargs=env_kwargs,
                state_variant=build_custom_dqn_state_variant_from_spec(state_spec),
                reward_variant=reward_variant_from_spec(reward_spec),
                total_episodes=args.train_episodes,
                test_episodes=args.test_episodes,
                seed=args.seed,
                label=f"{formulation.key}_{formulation.label}",
                total_timesteps=args.total_timesteps,
                parallel_envs=args.parallel_envs,
                eval_every_episodes=args.eval_every_episodes,
                vec_env_type=args.vec_env,
                ppo_n_steps=args.ppo_n_steps,
                ppo_batch_size=args.ppo_batch_size,
                ppo_n_epochs=args.ppo_n_epochs,
                ppo_device=args.ppo_device,
                train_reset_options_pool=train_reset_options_pool,
                eval_reset_options_schedule=eval_reset_options_schedule,
            )
            summary = load_json(Path(result.out_dir) / "l" / "summary.json")
        focused_dir = root / formulation.key / "focused_eval"
        focused_csv = focused_dir / "focused_eval_metrics.csv"
        if args.no_focused_eval:
            print(f"[{index}/{len(selected_formulations)}] focused eval disabled {formulation.key}", flush=True)
            focused = {}
        elif args.skip_existing and _file_exists(focused_csv) and not args.refresh_focused_eval:
            print(f"[{index}/{len(selected_formulations)}] skip existing focused eval {formulation.key}", flush=True)
            focused = aggregate_focused_metrics(focused_csv)
        else:
            print(f"[{index}/{len(selected_formulations)}] focused eval {formulation.key}", flush=True)
            run_focused_evaluation(
                model_path=out_dir,
                out_dir=focused_dir,
                seed=int(args.focused_seed),
                deterministic=True,
                include_bode=not bool(args.skip_bode),
                save_plots=not bool(args.no_plots),
            )
            focused = aggregate_focused_metrics(focused_csv)
        current_rows[formulation.key] = row_from_summary(formulation, summary, focused)
        rows = collect_available_rows(root, FORMULATIONS, current_rows)
        write_summary_csv(root / "summary.csv", rows)
        plot_summary(root, rows)

    tensorboard_root = Path.home() / "AppData" / "Local" / "TeleopWithRL_tb" / root.relative_to(Path(__file__).resolve().parents[2])
    rows = collect_available_rows(root, FORMULATIONS, current_rows)
    write_summary_csv(root / "summary.csv", rows)
    plot_summary(root, rows)
    write_summary_markdown(root, rows, tensorboard_root)
    save_json(
        root / "study_manifest.json",
        {
            "study_name": str(args.study_name),
            "objective": "state/reward physics-informed PPO formulations",
            "training_protocol": {
                "train_episodes": int(args.train_episodes),
                "total_timesteps": int(args.total_timesteps),
                "test_episodes": int(args.test_episodes),
                "parallel_envs": int(args.parallel_envs),
                "vec_env": str(args.vec_env),
                "ppo_n_steps": int(args.ppo_n_steps),
                "ppo_batch_size": int(args.ppo_batch_size),
                "ppo_n_epochs": int(args.ppo_n_epochs),
                "ppo_device": str(args.ppo_device),
                "eval_every_episodes": int(args.eval_every_episodes),
                "train_signal_count": int(len(train_reset_options_pool)),
                "eval_signal_count": int(len(eval_reset_options_schedule)),
                "train_reset_options_json": None if args.train_reset_options_json is None else str(args.train_reset_options_json),
                "eval_reset_options_json": None if args.eval_reset_options_json is None else str(args.eval_reset_options_json),
            },
            "rows": rows,
            "tensorboard_root": str(tensorboard_root),
        },
    )
    print(f"summary_csv={root / 'summary.csv'}", flush=True)
    print(f"summary_md={root / 'summary.md'}", flush=True)
    print(f"tensorboard_root={tensorboard_root}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    """Parse physics-informed formulation study options."""
    parser = argparse.ArgumentParser(description="Run physics-informed PPO formulation comparisons.")
    parser.add_argument("--study-name", default="physics_informed_formulations_02_fair_500k")
    parser.add_argument("--env-mode", default=cfg.ENV_MODE_CHANGING, choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING])
    parser.add_argument("--fe-mode", default=FE_MODE_DYNAMICS)
    parser.add_argument("--episode-duration", type=float, default=30.0)
    parser.add_argument("--env-switch-time", type=float, default=10.0)
    parser.add_argument("--reset-position-mode", default="midpoint", choices=["midpoint", "zero"])
    parser.add_argument("--stroke-limit-mode", default="clamp", choices=["terminate", "clamp"])
    parser.add_argument("--force-amp", type=float, default=10.0)
    parser.add_argument("--force-bias", type=float, default=0.0)
    parser.add_argument("--force-freq-rad", type=float, default=1.0)
    parser.add_argument("--force-phase", type=float, default=0.0)
    parser.add_argument("--force-waveform", default="sine", choices=["sine", "cosine", "square", "ramp", "multisine"])
    parser.add_argument("--train-episodes", type=int, default=334)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--test-episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=["auto", "dummy", "subproc"], default="subproc")
    parser.add_argument("--ppo-n-steps", type=int, default=256)
    parser.add_argument("--ppo-batch-size", type=int, default=512)
    parser.add_argument("--ppo-n-epochs", type=int, default=4)
    parser.add_argument("--ppo-device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--eval-every-episodes", type=int, default=150)
    parser.add_argument("--train-reset-options-json", default=None)
    parser.add_argument("--eval-reset-options-json", default=None)
    parser.add_argument("--only", nargs="*", default=None, help="Optional formulation key filter, e.g. F2_error_dot_state_reward.")
    parser.add_argument("--eval-only", action="store_true", help="Load saved PPO models and rerun deterministic test evaluations.")
    parser.add_argument("--eval-seed-offset", type=int, default=30_000)
    parser.add_argument("--focused-seed", type=int, default=42)
    parser.add_argument("--skip-bode", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-focused-eval", action="store_true")
    parser.add_argument("--refresh-focused-eval", action="store_true")
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Parse arguments and launch the physics-informed formulation study."""
    run(parse_args())


if __name__ == "__main__":
    main()
