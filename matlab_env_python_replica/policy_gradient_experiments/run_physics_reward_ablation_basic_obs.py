from __future__ import annotations

import argparse
import csv
import math
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
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL import config as cfg
    from TeleopWithRL.matlab_env_python_replica.policy_gradient_experiments.paths import suite_root as policy_gradient_suite_root
    from TeleopWithRL.matlab_env_python_replica.scripts._common import replica_env_kwargs_from_args
    from TeleopWithRL.matlab_env_python_replica.simuoriginal_replica import FE_MODE_DYNAMICS
    from TeleopWithRL.matlab_env_python_replica.studies.common import history_array, save_json
    from TeleopWithRL.matlab_env_python_replica.studies.dqn_state_variants import build_custom_dqn_state_variant_from_spec
    from TeleopWithRL.matlab_env_python_replica.studies.focused_evaluation import run_focused_evaluation
    from TeleopWithRL.matlab_env_python_replica.studies.policy_gradient import (
        PG_ALGO_PPO_CONTINUOUS,
        load_reset_options_json,
        train_policy_gradient_variant,
    )
    from TeleopWithRL.matlab_env_python_replica.studies.rewarding import (
        DEFAULT_ACTION_DELTA_SCALE_V,
        DEFAULT_ACTION_SCALE_V,
        DEFAULT_HIGH_PASS_TAU_S,
        DEFAULT_SECOND_ORDER_OMEGA_N,
        DEFAULT_SECOND_ORDER_ZETA,
        DEFAULT_SLIDING_LAMBDA,
        DEFAULT_TRACKING_SCALE_M,
        DEFAULT_VELOCITY_ERROR_SCALE_MPS,
        reward_variant_from_spec,
    )
else:
    try:
        from ... import config as cfg
    except ImportError:
        import config as cfg
    from .paths import suite_root as policy_gradient_suite_root
    from ..scripts._common import replica_env_kwargs_from_args
    from ..simuoriginal_replica import FE_MODE_DYNAMICS
    from ..studies.common import history_array, save_json
    from ..studies.dqn_state_variants import build_custom_dqn_state_variant_from_spec
    from ..studies.focused_evaluation import run_focused_evaluation
    from ..studies.policy_gradient import PG_ALGO_PPO_CONTINUOUS, load_reset_options_json, train_policy_gradient_variant
    from ..studies.rewarding import (
        DEFAULT_ACTION_DELTA_SCALE_V,
        DEFAULT_ACTION_SCALE_V,
        DEFAULT_HIGH_PASS_TAU_S,
        DEFAULT_SECOND_ORDER_OMEGA_N,
        DEFAULT_SECOND_ORDER_ZETA,
        DEFAULT_SLIDING_LAMBDA,
        DEFAULT_TRACKING_SCALE_M,
        DEFAULT_VELOCITY_ERROR_SCALE_MPS,
        reward_variant_from_spec,
    )


BASIC_OBS_FEATURES = ("x_m", "x_s", "v_m", "v_s", "u_v")


@dataclass(frozen=True)
class RewardAblation:
    key: str
    label: str
    terms: tuple[dict[str, Any], ...]
    note: str


def _term(
    name: str,
    source: str,
    weight: float,
    scale_name: str,
    *,
    shape: str = "square",
    deadband_name: str | None = None,
) -> dict[str, Any]:
    term = {
        "name": name,
        "source": source,
        "shape": shape,
        "sign": "penalty",
        "weight": float(weight),
        "scale_name": scale_name,
    }
    if deadband_name:
        term["deadband_name"] = deadband_name
    return term


def _high_pass(values: np.ndarray, *, tau_s: float = DEFAULT_HIGH_PASS_TAU_S) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    dt = float(cfg.RL_DT)
    alpha = dt / max(float(tau_s) + dt, 1e-9)
    low = float(values[0])
    out = np.zeros(values.size, dtype=np.float64)
    for idx, value in enumerate(values):
        if idx == 0:
            low = float(value)
        else:
            low += alpha * (float(value) - low)
        out[idx] = float(value) - low
    return out


def _term_series(history: dict[str, Any]) -> dict[str, np.ndarray]:
    e = history_array(history, "pos_error", dtype=np.float64)
    v_m = history_array(history, "v_m", dtype=np.float64)
    v_s = history_array(history, "v_s", dtype=np.float64)
    a_m = history_array(history, "a_m_signal", dtype=np.float64)
    a_s = history_array(history, "a_s_signal", dtype=np.float64)
    u_v = history_array(history, "u_v", dtype=np.float64)
    n = min(e.size, v_m.size, v_s.size)
    if n == 0:
        return {}
    e = e[:n]
    v_m = v_m[:n]
    v_s = v_s[:n]
    edot = v_m - v_s
    if a_m.size >= n and a_s.size >= n:
        eddot = a_m[:n] - a_s[:n]
    else:
        eddot = np.zeros(n, dtype=np.float64)
    u = u_v[:n] if u_v.size >= n else np.zeros(n, dtype=np.float64)
    du = np.concatenate([[0.0], np.diff(u)])
    ddu = np.concatenate([[0.0], np.diff(du)])
    sliding = edot + (DEFAULT_SLIDING_LAMBDA * e)
    second_order = (
        eddot
        + (2.0 * DEFAULT_SECOND_ORDER_ZETA * DEFAULT_SECOND_ORDER_OMEGA_N * edot)
        + ((DEFAULT_SECOND_ORDER_OMEGA_N ** 2) * e)
    )
    energy = 0.5 * (e ** 2) + 0.5 * (edot ** 2)
    denergy = np.concatenate([[0.0], np.diff(energy)])
    return {
        "tracking_error": e,
        "velocity_error": edot,
        "sliding_error": sliding,
        "second_order_error": second_order,
        "action_delta": du,
        "action_delta2": ddu,
        "lyapunov_increase": np.maximum(denergy, 0.0),
        "phase_lag_proxy": e * v_m,
        "direction_disagreement": np.maximum(-(v_m * v_s), 0.0),
        "tracking_error_hf": _high_pass(e),
        "u_v_hf": _high_pass(u),
    }


def _load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _robust_scale(values: list[np.ndarray], fallback: float) -> float:
    merged = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values if np.asarray(value).size]
    if not merged:
        return float(fallback)
    arr = np.concatenate(merged)
    arr = np.abs(arr[np.isfinite(arr)])
    if arr.size == 0:
        return float(fallback)
    scale = float(np.nanpercentile(arr, 95.0))
    return max(scale, abs(float(fallback)), 1e-9)


def calibrate_scale_catalog(results_root: Path, calibration_study: str) -> dict[str, dict[str, Any]]:
    calibration_root = results_root.parent / str(calibration_study)
    paths = list(calibration_root.glob("F*/ppo/e/test.npz"))
    paths.extend(calibration_root.glob("force_bias_15_test/F*/bias15_test.npz"))
    buckets: dict[str, list[np.ndarray]] = {}
    for path in paths:
        try:
            history = _load_npz(path)
        except OSError:
            continue
        for key, values in _term_series(history).items():
            buckets.setdefault(key, []).append(values)

    fallback = {
        "tracking_error": DEFAULT_TRACKING_SCALE_M,
        "velocity_error": DEFAULT_VELOCITY_ERROR_SCALE_MPS,
        "sliding_error": DEFAULT_VELOCITY_ERROR_SCALE_MPS,
        "second_order_error": max(DEFAULT_VELOCITY_ERROR_SCALE_MPS / max(float(cfg.RL_DT), 1e-9), 1.0),
        "action_delta": DEFAULT_ACTION_DELTA_SCALE_V,
        "action_delta2": DEFAULT_ACTION_DELTA_SCALE_V,
        "lyapunov_increase": DEFAULT_TRACKING_SCALE_M ** 2,
        "phase_lag_proxy": DEFAULT_TRACKING_SCALE_M * DEFAULT_VELOCITY_ERROR_SCALE_MPS,
        "direction_disagreement": DEFAULT_VELOCITY_ERROR_SCALE_MPS ** 2,
        "tracking_error_hf": DEFAULT_TRACKING_SCALE_M,
        "u_v_hf": DEFAULT_ACTION_SCALE_V,
    }
    catalog = {
        f"{key}_scale": {"value": _robust_scale(buckets.get(key, []), fallback_value), "unit": "calibrated"}
        for key, fallback_value in fallback.items()
    }
    catalog.update(
        {
            "tracking_deadband_m": {"value": 0.002, "unit": "m"},
            "velocity_deadband_mps": {"value": 0.005, "unit": "m/s"},
            "action_voltage_v": {"value": DEFAULT_ACTION_SCALE_V, "unit": "V"},
        }
    )
    return catalog


def build_ablations() -> tuple[RewardAblation, ...]:
    e = _term("tracking_error", "tracking_error", 1.0, "tracking_error_scale")
    edot = _term("velocity_error", "velocity_error", 1.0, "velocity_error_scale")
    sliding = _term("sliding_error", "sliding_error", 1.0, "sliding_error_scale")
    du = _term("action_delta", "action_delta", 0.10, "action_delta_scale")
    ddu = _term("action_delta2", "action_delta2", 0.05, "action_delta2_scale")
    dyn = _term("second_order_error", "second_order_error", 0.50, "second_order_error_scale")
    lyap = _term("lyapunov_increase", "lyapunov_increase", 0.50, "lyapunov_increase_scale")
    phase = _term("phase_lag_proxy", "phase_lag_proxy", 0.25, "phase_lag_proxy_scale")
    direction = _term("direction_disagreement", "direction_disagreement", 0.25, "direction_disagreement_scale")
    hf_e = _term("tracking_error_hf", "tracking_error_hf", 0.25, "tracking_error_hf_scale")
    hf_u = _term("u_v_hf", "u_v_hf", 0.25, "u_v_hf_scale")
    e_dead = _term(
        "tracking_error_deadzone",
        "tracking_error",
        1.0,
        "tracking_error_scale",
        shape="deadband_square",
        deadband_name="tracking_deadband_m",
    )
    edot_dead = _term(
        "velocity_error_deadzone",
        "velocity_error",
        0.50,
        "velocity_error_scale",
        shape="deadband_square",
        deadband_name="velocity_deadband_mps",
    )
    return (
        RewardAblation("R0_e_only", "e only", (e,), "Tracking manifold only: e^2."),
        RewardAblation("R1_e_edot", "e + edot", (e, edot), "Tracking manifold plus velocity error."),
        RewardAblation("R2_sliding", "Sliding", (e, sliding), "First-order desired error dynamics."),
        RewardAblation("R3_sliding_du", "Sliding + du", (e, sliding, du), "Sliding dynamics plus valve smoothness."),
        RewardAblation("R4_sliding_du_ddu", "Sliding + du + ddu", (e, sliding, du, ddu), "Adds valve jerk penalty."),
        RewardAblation("R5_second_order", "Second order", (e, sliding, du, ddu, dyn), "Adds damped second-order residual."),
        RewardAblation("R6_lyapunov", "Lyapunov", (e, sliding, du, ddu, dyn, lyap), "Adds tracking-energy increase penalty."),
        RewardAblation(
            "R7_phase_direction",
            "Phase + direction",
            (e, sliding, du, ddu, dyn, lyap, phase, direction),
            "Adds phase-lag proxy and velocity direction agreement.",
        ),
        RewardAblation(
            "R8_hf_deadzone",
            "HF + deadzone",
            (e_dead, edot_dead, sliding, du, ddu, dyn, lyap, phase, direction, hf_e, hf_u),
            "Dead-zone tracking/velocity errors plus high-frequency tracking/control penalties.",
        ),
    )


def build_state_spec() -> dict[str, Any]:
    return {
        "name": "basic_obs_xm_xs_vm_vs_u",
        "description": "Basic observation space shared by all reward-ablation runs.",
        "selected_features": list(BASIC_OBS_FEATURES),
    }


def build_reward_spec(ablation: RewardAblation, scale_catalog: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": f"{ablation.key}_reward",
        "description": ablation.note,
        "scale_catalog": scale_catalog,
        "terms": [dict(term) for term in ablation.terms],
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(_plot_save_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        import json

        return json.load(handle)


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def aggregate_focused_metrics(path: Path) -> dict[str, float]:
    rows = _read_csv_rows(path)
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


def row_from_summary(ablation: RewardAblation, summary: dict[str, Any], focused: dict[str, float]) -> dict[str, Any]:
    eval_metrics = dict(summary.get("eval_metrics") or {})
    train_path = Path(str(summary.get("out_dir", ""))) / "l" / "train.npz"
    train_completed_episodes = 0
    if train_path.exists():
        try:
            train_data = np.load(train_path, allow_pickle=False)
            train_completed_episodes = int(np.asarray(train_data.get("episode_returns", [])).size)
        except Exception:
            train_completed_episodes = 0
    train_requested_episodes = int(summary.get("total_episodes", 0) or 0)
    train_episode_coverage = (
        float(train_completed_episodes / max(train_requested_episodes, 1))
        if train_requested_episodes
        else 0.0
    )
    row: dict[str, Any] = {
        "key": ablation.key,
        "label": ablation.label,
        "obs_dim": int(summary.get("obs_dim", len(BASIC_OBS_FEATURES))),
        "state_features": " ".join(BASIC_OBS_FEATURES),
        "reward_terms": " ".join(term["name"] for term in ablation.terms),
        "note": ablation.note,
        "out_dir": str(summary.get("out_dir", "")),
        "model_path": str(summary.get("model_path", "")),
        "train_requested_episodes": train_requested_episodes,
        "train_completed_episodes": train_completed_episodes,
        "train_episode_coverage": train_episode_coverage,
        "total_timesteps": int(summary.get("total_timesteps", 0) or 0),
        "actual_train_timesteps": int(summary.get("actual_train_timesteps", summary.get("total_timesteps", 0)) or 0),
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
    }
    for key in (
        "mean_reward",
        "tracking_rmse_m",
        "tracking_mae_m",
        "tracking_max_abs_m",
        "velocity_error_rmse_mps",
        "acceleration_error_rmse_mps2",
        "transparency_rmse_w",
        "transparency_ratio_median",
        "transparency_ratio_error_rmse",
        "transparency_ratio_valid_fraction",
        "transparency_ratio_within_20pct",
        "rms_u_v",
        "mean_abs_delta_u_v",
        "mean_abs_delta2_u_v",
        "completed_episode_rate",
        "invalid_episode_rate",
    ):
        row[key] = summary.get(key, eval_metrics.get(key, ""))
    row.update(focused)
    return row


def collect_available_rows(
    root: Path,
    ablations: list[RewardAblation],
    current_rows: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    current_rows = dict(current_rows or {})
    rows: list[dict[str, Any]] = []
    for ablation in ablations:
        if ablation.key in current_rows:
            rows.append(current_rows[ablation.key])
            continue
        summary_path = root / ablation.key / "ppo" / "l" / "summary.json"
        if not summary_path.exists():
            continue
        focused_csv = root / ablation.key / "focused_eval" / "focused_eval_metrics.csv"
        rows.append(row_from_summary(ablation, load_json(summary_path), aggregate_focused_metrics(focused_csv)))
    return rows


def plot_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [str(row["key"]).replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    panels = [
        ("focused_tracking_rmse_mm", "Focused tracking RMSE [mm]", False, "linear"),
        ("focused_transparency_rmse_w", "Focused transparency RMSE [W]", False, "linear"),
        ("focused_transparency_ratio_median", "Focused ratio median", True, "linear"),
        ("focused_mean_abs_delta2_u_v", "Mean |delta2 u| [V]", False, "log"),
        ("focused_failure_rate", "Failure rate", False, "linear"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 18), constrained_layout=True)
    for ax, (key, ylabel, ratio_panel, yscale) in zip(axes, panels):
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        ax.bar(x, values, color="tab:blue", alpha=0.78)
        if yscale == "log" and any(value > 0.0 for value in values):
            min_positive = min(value for value in values if value > 0.0)
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(min_positive * 0.5, 1e-9))
        if ratio_panel:
            ax.axhline(1.0, color="tab:red", lw=1.2, ls="--", label="ideal = 1")
            ax.legend(loc="best", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Physics-informed reward ablation with basic observation space")
    fig.savefig(_plot_save_path(root / "reward_ablation_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_save_path(path: str | Path) -> str | Path:
    path = Path(path)
    if os.name != "nt":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    resolved = path.resolve()
    parent = str(path.parent.resolve())
    if parent.startswith("\\\\?\\"):
        parent_text = parent
    elif parent.startswith("\\\\"):
        parent_text = "\\\\?\\UNC\\" + parent.lstrip("\\")
    else:
        parent_text = "\\\\?\\" + parent
    os.makedirs(parent_text, exist_ok=True)
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text.lstrip("\\")
    return "\\\\?\\" + text


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


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    try:
        with open(_long_path(path), "r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except FileNotFoundError:
        return []


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def plot_training_curves(root: Path, ablations: list[RewardAblation]) -> None:
    series: list[tuple[RewardAblation, dict[str, np.ndarray]]] = []
    for ablation in ablations:
        train_path = root / ablation.key / "ppo" / "l" / "train.npz"
        if not train_path.exists():
            continue
        try:
            data = np.load(train_path, allow_pickle=False)
            series.append((ablation, {key: data[key] for key in data.files}))
        except Exception:
            continue
    if not series:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    for ablation, data in series:
        label = f"{ablation.key.split('_')[0]} {ablation.label}"
        eval_steps = np.asarray(data.get("eval_steps", []), dtype=np.float64)
        if eval_steps.size and "eval_tracking_rmse" in data:
            axes[0].plot(eval_steps, np.asarray(data["eval_tracking_rmse"], dtype=np.float64) * 1000.0, marker="o", label=label)
        if eval_steps.size and "eval_mean_reward" in data:
            axes[1].plot(eval_steps, np.asarray(data["eval_mean_reward"], dtype=np.float64), marker="o", label=label)
        returns = np.asarray(data.get("episode_returns", []), dtype=np.float64)
        if returns.size:
            episodes = np.arange(1, returns.size + 1)
            window = max(1, min(5, returns.size))
            kernel = np.ones(window, dtype=np.float64) / float(window)
            smooth = np.convolve(returns, kernel, mode="same")
            axes[2].plot(episodes, smooth, label=label)

    axes[0].set_ylabel("eval tracking RMSE [mm]")
    axes[1].set_ylabel("eval mean reward")
    axes[2].set_ylabel("episode return, smoothed")
    axes[2].set_xlabel("episode / eval step")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Physics-informed reward ablation training curves")
    axes[0].legend(loc="best", fontsize=7, ncol=3)
    fig.savefig(_plot_save_path(root / "reward_ablation_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_group_heatmap(root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = [str(row["key"]) for row in rows]
    labels = [f"{key.split('_')[0]}\n{str(row['label']).replace(' + ', '+')}" for key, row in zip(keys, rows)]
    per_variant: dict[str, list[dict[str, str]]] = {
        key: _read_csv_rows(root / key / "focused_eval" / "focused_eval_metrics.csv")
        for key in keys
    }
    groups = sorted({str(row.get("group", "")) for variant_rows in per_variant.values() for row in variant_rows if row.get("group")})
    if not groups:
        return

    panels = [
        ("rms_error_m", "tracking RMSE [mm]", 1000.0, "{:.1f}"),
        ("transparency_ratio_within_20pct", "ratio within +/-20% [%]", 100.0, "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(max(12, 1.2 * len(groups)), 7), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (metric, title, scale, fmt) in zip(axes, panels):
        matrix = np.full((len(keys), len(groups)), np.nan, dtype=np.float64)
        for i, key in enumerate(keys):
            rows_by_group: dict[str, list[float]] = {group: [] for group in groups}
            for metric_row in per_variant.get(key, []):
                group = str(metric_row.get("group", ""))
                if group in rows_by_group:
                    rows_by_group[group].append(_as_float(metric_row.get(metric)))
            for j, group in enumerate(groups):
                values = np.asarray(rows_by_group[group], dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size:
                    matrix[i, j] = float(np.mean(values) * scale)
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=6, color="white")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Focused validation group heatmap")
    fig.savefig(_plot_save_path(root / "reward_ablation_group_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = policy_gradient_suite_root(args.fe_mode, args.study_name)
    root.mkdir(parents=True, exist_ok=True)
    specs_root = root / "specs"
    specs_root.mkdir(parents=True, exist_ok=True)
    train_reset_options_pool = load_reset_options_json(args.train_reset_options_json)
    eval_reset_options_schedule = load_reset_options_json(args.eval_reset_options_json)
    scale_catalog = calibrate_scale_catalog(root, args.calibration_study)
    save_json(specs_root / "reward_scale_catalog.json", scale_catalog)
    state_spec = build_state_spec()
    save_json(specs_root / "basic_obs_state.json", state_spec)

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

    all_ablations = build_ablations()
    current_rows: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = collect_available_rows(root, all_ablations, current_rows)
    ablations = [
        ablation
        for ablation in all_ablations
        if not args.only or ablation.key in set(str(key) for key in args.only)
    ]
    if not ablations:
        raise ValueError(f"No reward ablations selected by --only {args.only!r}")
    for index, ablation in enumerate(ablations, start=1):
        reward_spec = build_reward_spec(ablation, scale_catalog)
        save_json(specs_root / f"{ablation.key}_reward.json", reward_spec)
        out_dir = root / ablation.key / "ppo"
        summary_path = out_dir / "l" / "summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[{index}/{len(ablations)}] skip existing train {ablation.key}", flush=True)
        else:
            print(f"[{index}/{len(ablations)}] train {ablation.key}: {ablation.note}", flush=True)
            train_policy_gradient_variant(
                algo=PG_ALGO_PPO_CONTINUOUS,
                out_dir=out_dir,
                env_mode=args.env_mode,
                env_kwargs=env_kwargs,
                state_variant=build_custom_dqn_state_variant_from_spec(state_spec),
                reward_variant=reward_variant_from_spec(reward_spec),
                total_episodes=args.train_episodes,
                test_episodes=args.test_episodes,
                seed=args.seed,
                label=f"{ablation.key}_{ablation.label}",
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
        summary = load_json(summary_path)

        focused_dir = root / ablation.key / "focused_eval"
        focused_csv = focused_dir / "focused_eval_metrics.csv"
        if args.no_focused_eval:
            print(f"[{index}/{len(ablations)}] focused eval disabled {ablation.key}", flush=True)
        elif args.skip_existing and _file_exists(focused_csv) and not args.refresh_focused_eval:
            print(f"[{index}/{len(ablations)}] skip existing focused eval {ablation.key}", flush=True)
        else:
            print(f"[{index}/{len(ablations)}] focused eval {ablation.key}", flush=True)
            run_focused_evaluation(
                model_path=out_dir,
                out_dir=focused_dir,
                seed=args.focused_seed,
                deterministic=True,
                include_bode=not args.skip_bode,
                save_plots=not args.no_plots,
            )
        focused = aggregate_focused_metrics(focused_csv)
        current_rows[ablation.key] = row_from_summary(ablation, summary, focused)
        rows = collect_available_rows(root, all_ablations, current_rows)
        write_csv(root / "summary.csv", rows)
        plot_summary(root, rows)

    rows = collect_available_rows(root, all_ablations, current_rows)
    plot_summary(root, rows)
    plot_training_curves(root, all_ablations)
    plot_group_heatmap(root, rows)
    save_json(
        root / "study_manifest.json",
        {
            "study_name": args.study_name,
            "objective": "physics-informed reward ablation with fixed basic observation space",
            "basic_observation_features": list(BASIC_OBS_FEATURES),
            "focused_eval_seed": int(args.focused_seed),
            "focused_eval_include_bode": not bool(args.skip_bode),
            "training_protocol": {
                "train_episodes": int(args.train_episodes),
                "total_timesteps": None if args.total_timesteps is None else int(args.total_timesteps),
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
            "scale_catalog": scale_catalog,
            "rows": rows,
        },
    )
    write_csv(root / "summary.csv", rows)
    plot_summary(root, rows)
    print(f"summary_csv={root / 'summary.csv'}", flush=True)
    print(f"summary_plot={root / 'reward_ablation_summary.png'}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run physics-informed reward ablations with fixed basic observations.")
    parser.add_argument("--study-name", default="physics_reward_ablation_basic_obs_03_fair_500k")
    parser.add_argument("--calibration-study", default="physics_informed_formulations_02_fair_500k")
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
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help=(
            "Override PPO timesteps. The default matches the full PPO baseline notebook budget."
        ),
    )
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
    parser.add_argument("--focused-seed", type=int, default=42)
    parser.add_argument("--only", nargs="*", default=None, help="Optional ablation key filter, e.g. R5_second_order.")
    parser.add_argument("--skip-bode", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-focused-eval", action="store_true")
    parser.add_argument("--refresh-focused-eval", action="store_true")
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
