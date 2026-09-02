"""CLI study comparing temporal observation-stack formulations."""

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
    from TeleopWithRL.matlab_env_python_replica.common.study_utils import save_json
    from TeleopWithRL.matlab_env_python_replica.dqn.state_variants import build_custom_dqn_state_variant_from_spec
    from TeleopWithRL.matlab_env_python_replica.common.focused_evaluation import run_focused_evaluation
    from TeleopWithRL.matlab_env_python_replica.policy_gradient.training import (
        PG_ALGO_PPO_CONTINUOUS,
        load_reset_options_json,
        train_policy_gradient_variant,
    )
    from TeleopWithRL.matlab_env_python_replica.common.rewarding import (
        DEFAULT_ACTION_SCALE_V,
        DEFAULT_TRACKING_SCALE_M,
        reward_variant_from_spec,
    )
else:
    from ...config import config as cfg
    from ..paths import suite_root as policy_gradient_suite_root
    from ...common.cli import replica_env_kwargs_from_args
    from ...environment.simuoriginal_replica import FE_MODE_DYNAMICS
    from ...common.study_utils import save_json
    from ...dqn.state_variants import build_custom_dqn_state_variant_from_spec
    from ...common.focused_evaluation import run_focused_evaluation
    from ..training import PG_ALGO_PPO_CONTINUOUS, load_reset_options_json, train_policy_gradient_variant
    from ...common.rewarding import DEFAULT_ACTION_SCALE_V, DEFAULT_TRACKING_SCALE_M, reward_variant_from_spec


@dataclass(frozen=True)
class TemporalFormulation:
    key: str
    label: str
    state_features: tuple[str, ...]
    lags: tuple[int, ...]
    note: str


BASE_POS_ACTION = ("x_m", "x_s", "u_v")
BASE_POSVEL_ACTION = ("x_m", "x_s", "v_m", "v_s", "u_v")


FORMULATIONS = (
    TemporalFormulation(
        "T0_pos_current",
        "Position Current",
        BASE_POS_ACTION,
        (0,),
        "Current x_m, x_s, and u only; no explicit velocity.",
    ),
    TemporalFormulation(
        "T1_pos_stack3",
        "Position Stack 3",
        BASE_POS_ACTION,
        (0, 1, 2),
        "x_m, x_s, and u at t, t-1, and t-2 so the policy can infer first differences.",
    ),
    TemporalFormulation(
        "T2_pos_stack5",
        "Position Stack 5",
        BASE_POS_ACTION,
        (0, 1, 2, 3, 4),
        "x_m, x_s, and u over a 5-step window so the policy can infer richer local dynamics.",
    ),
    TemporalFormulation(
        "T3_posvel_current",
        "Position Velocity Current",
        BASE_POSVEL_ACTION,
        (0,),
        "Reference current-state baseline with explicit velocities.",
    ),
    TemporalFormulation(
        "T4_posvel_stack3",
        "Position Velocity Stack 3",
        BASE_POSVEL_ACTION,
        (0, 1, 2),
        "Baseline state x_m, x_s, v_m, v_s, and u at t, t-1, and t-2.",
    ),
)


SUMMARY_FIELDS = (
    "key",
    "label",
    "obs_dim",
    "base_features",
    "lags",
    "total_timesteps",
    "actual_train_timesteps",
    "train_requested_episodes",
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
    "model_path",
    "mean_reward",
    "tracking_rmse_m",
    "tracking_mae_m",
    "tracking_max_abs_m",
    "velocity_error_rmse_mps",
    "acceleration_error_rmse_mps2",
    "transparency_rmse_w",
    "transparency_ratio_mean",
    "transparency_ratio_error_rmse",
    "mean_abs_u_v",
    "rms_u_v",
    "mean_abs_delta_u_v",
    "rms_delta_u_v",
    "completed_episode_rate",
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
)


def _term(name: str, source: str, weight: float, scale_name: str) -> dict[str, Any]:
    return {
        "name": name,
        "source": source,
        "shape": "square",
        "sign": "penalty",
        "weight": float(weight),
        "scale_name": scale_name,
    }


def build_state_spec(formulation: TemporalFormulation) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": f"{formulation.key}_state",
        "description": formulation.note,
        "selected_features": list(formulation.state_features),
    }
    if formulation.lags != (0,):
        spec["temporal_stack"] = {
            "enabled": True,
            "lags": list(formulation.lags),
            "reset_fill": "repeat_current",
        }
    return spec


def build_reward_spec(formulation: TemporalFormulation) -> dict[str, Any]:
    return {
        "name": f"{formulation.key}_reward",
        "description": f"Tracking/effort reward for temporal observation study: {formulation.key}.",
        "scale_catalog": {
            "tracking_error_m": {"value": DEFAULT_TRACKING_SCALE_M, "unit": "m"},
            "action_voltage_v": {"value": DEFAULT_ACTION_SCALE_V, "unit": "V"},
        },
        "terms": [
            _term("tracking_base", "pos_error", 40.0, "tracking_error_m"),
            _term("control_effort", "u_v", 0.01, "action_voltage_v"),
        ],
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


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    os.makedirs(_long_path(path.parent), exist_ok=True)
    with open(_long_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def _load_training_npz(run_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(run_dir) / "l" / "train.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def aggregate_focused_metrics(path: Path) -> dict[str, float]:
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
    formulations: tuple[TemporalFormulation, ...],
    current_rows: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    current_rows = dict(current_rows or {})
    rows: list[dict[str, Any]] = []
    for formulation in formulations:
        if formulation.key in current_rows:
            rows.append(current_rows[formulation.key])
            continue
        summary_path = root / formulation.key / "ppo" / "l" / "summary.json"
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
    formulation: TemporalFormulation,
    summary: dict[str, Any],
    focused: dict[str, float] | None = None,
) -> dict[str, Any]:
    eval_metrics = dict(summary.get("eval_metrics") or {})
    row: dict[str, Any] = {
        "key": formulation.key,
        "label": formulation.label,
        "obs_dim": int(summary.get("obs_dim", len(formulation.state_features) * len(formulation.lags))),
        "base_features": " ".join(formulation.state_features),
        "lags": " ".join(str(lag) for lag in formulation.lags),
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


def _maybe_symlog(ax, values: list[float]) -> None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size and float(np.max(np.abs(finite))) > 100.0:
        ax.set_yscale("symlog", linthresh=1.0)


def plot_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [str(row["key"]).replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    metrics = [
        ("tracking_rmse_m", "Tracking RMSE [mm]", 1000.0),
        ("velocity_error_rmse_mps", "Velocity Error RMSE [m/s]", 1.0),
        ("transparency_rmse_w", "Transparency RMSE [W]", 1.0),
        ("rms_u_v", "RMS u_v [V]", 1.0),
        ("mean_abs_delta_u_v", "Mean abs(delta u) [V]", 1.0),
        ("transparency_ratio_mean", "Actual transparency ratio (F_h/v_m)/(F_e/v_s)", 1.0),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 17), constrained_layout=True)
    for ax, (key, ylabel, scale) in zip(axes, metrics):
        values = [float(row.get(key, 0.0)) * scale for row in rows]
        ax.bar(x, values, color="tab:blue", alpha=0.78)
        if key == "transparency_ratio_mean":
            ax.axhline(1.0, color="tab:red", lw=1.1, ls="--", label="ideal ratio = 1")
            _maybe_symlog(ax, values)
            ax.legend(loc="best", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Temporal observation stack comparison")
    fig.savefig(root / "summary_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    all_ratios: list[float] = []
    for row in rows:
        train = _load_training_npz(row["out_dir"])
        if not train:
            continue
        steps = train.get("eval_steps", np.asarray([], dtype=np.float64))
        if steps.size == 0:
            continue
        axes[0].plot(steps, train.get("eval_tracking_rmse", np.asarray([])) * 1000.0, marker="o", label=row["key"])
        transparency_rmse = np.asarray(train.get("eval_transparency_rmse", np.asarray([])), dtype=np.float64)
        axes[1].plot(steps, transparency_rmse, marker="o", label=row["key"])
        all_ratios.extend(transparency_rmse.tolist())
        axes[2].plot(steps, train.get("eval_mean_reward", np.asarray([])), marker="o", label=row["key"])
    axes[0].set_ylabel("Eval tracking RMSE [mm]")
    axes[1].set_ylabel("Eval transparency RMSE [W]")
    _maybe_symlog(axes[1], all_ratios)
    axes[2].set_ylabel("Eval return")
    axes[2].set_xlabel("Completed training episodes")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.savefig(root / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(root: Path, rows: list[dict[str, Any]], tensorboard_root: Path) -> None:
    lines = [
        "# Temporal Observation Stack Study",
        "",
        f"TensorBoard root: `{tensorboard_root}`",
        "",
        "| Formulation | Obs dim | Lags | Track RMSE mm | Vel err RMSE | Transp RMSE W | RMS u V | Mean abs(delta u) V | Actual transp ratio mean | Completed |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {key} | {obs_dim} | {lags} | {track:.3f} | {vel:.4f} | {transp:.4f} | {rms_u:.3f} | {du:.4f} | {ratio:.4g} | {done:.2f} |".format(
                key=row["key"],
                obs_dim=int(row.get("obs_dim", 0)),
                lags=str(row.get("lags", "")),
                track=1000.0 * float(row.get("tracking_rmse_m", 0.0)),
                vel=float(row.get("velocity_error_rmse_mps", 0.0)),
                transp=float(row.get("transparency_rmse_w", 0.0)),
                rms_u=float(row.get("rms_u_v", 0.0)),
                du=float(row.get("mean_abs_delta_u_v", 0.0)),
                ratio=float(row.get("transparency_ratio_mean", 0.0)),
                done=float(row.get("completed_episode_rate", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "Generated artifacts:",
            "",
            "- `summary.csv`: flat metric table",
            "- `summary_bars.png`: final metric comparison",
            "- `learning_curves.png`: evaluation checkpoints across training",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
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
        if not args.only or formulation.key in set(args.only)
    )
    if not selected_formulations:
        known = ", ".join(formulation.key for formulation in FORMULATIONS)
        raise KeyError(f"No formulations matched --only {args.only!r}. Known formulations: {known}")

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
            "objective": "temporal observation stack PPO comparison",
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
    parser = argparse.ArgumentParser(description="Run PPO temporal-observation-stack comparisons.")
    parser.add_argument("--study-name", default="temporal_observation_stack_02_fair_500k")
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
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--parallel-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=["auto", "dummy", "subproc"], default="subproc")
    parser.add_argument("--ppo-n-steps", type=int, default=256)
    parser.add_argument("--ppo-batch-size", type=int, default=512)
    parser.add_argument("--ppo-n-epochs", type=int, default=4)
    parser.add_argument("--ppo-device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--eval-every-episodes", type=int, default=150)
    parser.add_argument("--train-reset-options-json", default=None)
    parser.add_argument("--eval-reset-options-json", default=None)
    parser.add_argument("--only", nargs="*", default=None, help="Optional formulation key filter, e.g. T2_pos_stack5.")
    parser.add_argument("--focused-seed", type=int, default=42)
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
