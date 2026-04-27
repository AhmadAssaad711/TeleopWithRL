from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL import config as cfg
    from TeleopWithRL.matlab_literal_env.simuoriginal_replica import FE_MODE_GUI
    from TeleopWithRL.matlab_literal_env.studies.common import save_json, stage_summary_rows_to_csv
    from TeleopWithRL.matlab_literal_env.studies.ql_state_variants import get_ql_state_variant
    from TeleopWithRL.matlab_literal_env.studies.qlearning import train_qlearning_variant
    from TeleopWithRL.matlab_literal_env.studies.rewarding import baseline_reward_variant
else:
    from ... import config as cfg
    from ..simuoriginal_replica import FE_MODE_GUI
    from ..studies.common import save_json, stage_summary_rows_to_csv
    from ..studies.ql_state_variants import get_ql_state_variant
    from ..studies.qlearning import train_qlearning_variant
    from ..studies.rewarding import baseline_reward_variant


def _results_root(study_name: str, fe_mode: str) -> Path:
    fe_dir = "gui" if str(fe_mode) == FE_MODE_GUI else "dyn"
    return Path(__file__).resolve().parent / "results" / fe_dir / study_name


def _load_summary(run_dir: Path) -> dict[str, Any]:
    with open(run_dir / "l" / "summary.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _variant_row(variant_name: str, summary: dict[str, Any], baseline_summary: dict[str, Any] | None) -> dict[str, Any]:
    completed_rate = float(summary.get("completed_episode_rate", 0.0))
    mean_seconds = float(summary.get("mean_episode_seconds", 0.0))
    stroke_limit = int(summary.get("stroke_limit_episodes", 0))
    tracking_fail = int(summary.get("tracking_error_fail_episodes", 0))
    row = {
        "variant_name": variant_name,
        "reward_variant": str(summary.get("reward_variant")),
        "completed_episode_rate": completed_rate,
        "mean_episode_seconds": mean_seconds,
        "stroke_limit_episodes": stroke_limit,
        "tracking_error_fail_episodes": tracking_fail,
        "mean_reward": float(summary.get("mean_reward", 0.0)),
        "tracking_rmse_m": float(summary.get("tracking_rmse_m", 0.0)),
        "transparency_rmse_w": float(summary.get("transparency_rmse_w", 0.0)),
        "invalid_episode_rate": float(summary.get("invalid_episode_rate", 0.0)),
        "model_path": str(summary.get("model_path", "")),
        "out_dir": str(summary.get("out_dir", "")),
    }
    if baseline_summary is not None:
        row["delta_completed_episode_rate"] = completed_rate - float(baseline_summary.get("completed_episode_rate", 0.0))
        row["delta_mean_episode_seconds"] = mean_seconds - float(baseline_summary.get("mean_episode_seconds", 0.0))
        row["delta_tracking_rmse_m"] = row["tracking_rmse_m"] - float(baseline_summary.get("tracking_rmse_m", 0.0))
        row["delta_transparency_rmse_w"] = row["transparency_rmse_w"] - float(baseline_summary.get("transparency_rmse_w", 0.0))
    else:
        row["delta_completed_episode_rate"] = 0.0
        row["delta_mean_episode_seconds"] = 0.0
        row["delta_tracking_rmse_m"] = 0.0
        row["delta_transparency_rmse_w"] = 0.0
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-change-at-a-time Q-learning edge-fix screening.")
    parser.add_argument("--study-name", default="edgefix_qscreen_01")
    parser.add_argument("--env-mode", default=cfg.ENV_MODE_CHANGING)
    parser.add_argument("--episode-duration", type=float, default=30.0)
    parser.add_argument("--env-switch-time", type=float, default=10.0)
    parser.add_argument("--force-amp", type=float, default=10.0)
    parser.add_argument("--force-bias", type=float, default=5.0)
    parser.add_argument("--force-freq-rad", type=float, default=float(math.pi))
    parser.add_argument("--force-phase", type=float, default=0.0)
    parser.add_argument("--force-waveform", default="sine")
    parser.add_argument("--reset-position-mode", default="midpoint")
    parser.add_argument("--q-episodes", type=int, default=600)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    state_variant = get_ql_state_variant("Q0_baseline_reduced4")
    base_reward = baseline_reward_variant()
    study_root = _results_root(str(args.study_name), FE_MODE_GUI)
    study_root.mkdir(parents=True, exist_ok=True)

    base_env_kwargs = {
        "episode_duration": float(args.episode_duration),
        "env_switch_time": float(args.env_switch_time),
        "terminate_on_error": True,
        "reset_position_mode": str(args.reset_position_mode),
        "action_levels": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "reset_options": {
            "force_amp": float(args.force_amp),
            "force_bias": float(args.force_bias),
            "force_freq_rad": float(args.force_freq_rad),
            "force_phase": float(args.force_phase),
            "force_waveform": str(args.force_waveform),
            "fe_mode": FE_MODE_GUI,
            "reset_position_mode": str(args.reset_position_mode),
        },
    }

    variants = [
        {
            "key": "b0_base",
            "label": "baseline",
            "reward_variant": replace(base_reward, name="baseline_cfg"),
            "env_kwargs": dict(base_env_kwargs),
        },
        {
            "key": "p1_term",
            "label": "stronger_terminal_penalty",
            "reward_variant": replace(base_reward, name="stroke_penalty_1000", stroke_limit_penalty=1000.0),
            "env_kwargs": dict(base_env_kwargs),
        },
        {
            "key": "g2_edge",
            "label": "inner_edge_penalty",
            "reward_variant": replace(base_reward, name="edge_guard_30mm", edge_buffer_m=0.03, edge_penalty_weight=40.0),
            "env_kwargs": dict(base_env_kwargs),
        },
        {
            "key": "l3_lowf",
            "label": "low_force_edge_penalty",
            "reward_variant": replace(
                base_reward,
                name="low_force_edge_30mm",
                edge_buffer_m=0.03,
                low_force_threshold_n=2.5,
                low_force_edge_penalty_weight=80.0,
            ),
            "env_kwargs": dict(base_env_kwargs),
        },
        {
            "key": "a4_damp",
            "label": "edge_action_damping",
            "reward_variant": replace(base_reward, name="baseline_cfg"),
            "env_kwargs": {
                **dict(base_env_kwargs),
                "edge_action_damping_buffer_m": 0.03,
                "edge_action_min_scale": 0.25,
            },
        },
        {
            "key": "f5_bias",
            "label": "higher_bias_force",
            "reward_variant": replace(base_reward, name="baseline_cfg"),
            "env_kwargs": {
                **dict(base_env_kwargs),
                "reset_options": {
                    **dict(base_env_kwargs["reset_options"]),
                    "force_bias": 15.0,
                },
            },
        },
    ]

    baseline_summary: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []

    for idx, variant in enumerate(variants, start=1):
        run_dir = study_root / variant["key"]
        summary_path = run_dir / "l" / "summary.json"
        print(f"[edge_fix_screen] {idx}/{len(variants)} {variant['label']} -> {run_dir}", flush=True)
        if args.skip_existing and summary_path.exists():
            summary = _load_summary(run_dir)
        else:
            train_qlearning_variant(
                out_dir=run_dir,
                env_mode=str(args.env_mode),
                env_kwargs=dict(variant["env_kwargs"]),
                state_variant=state_variant,
                reward_variant=variant["reward_variant"],
                total_episodes=int(args.q_episodes),
                test_episodes=int(args.test_episodes),
                seed=int(args.seed),
                label=f"QL_edge_fix_{variant['label']}",
            )
            summary = _load_summary(run_dir)
        if baseline_summary is None:
            baseline_summary = summary
        row = _variant_row(str(variant["label"]), summary, baseline_summary if variant["label"] != "baseline" else None)
        row["variant_key"] = str(variant["key"])
        rows.append(row)
        manifests.append(
            {
                "variant_key": str(variant["key"]),
                "label": str(variant["label"]),
                "reward_variant_name": variant["reward_variant"].name,
                "env_kwargs": variant["env_kwargs"],
                "summary": summary,
            }
        )

    save_json(
        study_root / "comparison.json",
        {
            "study_name": str(args.study_name),
            "env_mode": str(args.env_mode),
            "q_episodes": int(args.q_episodes),
            "test_episodes": int(args.test_episodes),
            "variants": manifests,
            "rows": rows,
        },
    )
    stage_summary_rows_to_csv(rows, study_root / "comparison.csv")
    print(f"[edge_fix_screen] wrote {study_root / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
