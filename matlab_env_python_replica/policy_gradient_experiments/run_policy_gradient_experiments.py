from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL import config as cfg
    from TeleopWithRL.matlab_env_python_replica.policy_gradient_experiments.paths import suite_root as policy_gradient_suite_root
    from TeleopWithRL.matlab_env_python_replica.simuoriginal_replica import FE_MODE_CHOICES, FE_MODE_GUI
    from TeleopWithRL.matlab_env_python_replica.scripts import run_replica_studies as runner
    from TeleopWithRL.matlab_env_python_replica.studies.common import save_json, stage_completed, stage_summary_rows_to_csv
    from TeleopWithRL.matlab_env_python_replica.studies.policy_gradient import (
        PG_ALGO_CHOICES,
        algo_display_name,
        algo_notebook_tag,
        algo_output_dir_name,
        get_policy_gradient_reward_variant,
        get_policy_gradient_state_variant,
        require_sb3,
        train_policy_gradient_variant,
    )
else:
    from ... import config as cfg
    from .paths import suite_root as policy_gradient_suite_root
    from ..simuoriginal_replica import FE_MODE_CHOICES, FE_MODE_GUI
    from ..scripts import run_replica_studies as runner
    from ..studies.common import save_json, stage_completed, stage_summary_rows_to_csv
    from ..studies.policy_gradient import (
        PG_ALGO_CHOICES,
        algo_display_name,
        algo_notebook_tag,
        algo_output_dir_name,
        get_policy_gradient_reward_variant,
        get_policy_gradient_state_variant,
        require_sb3,
        train_policy_gradient_variant,
    )


DEFAULT_PARALLEL_WORKERS = max(1, min(8, os.cpu_count() or 1))


def _suite_root(fe_mode: str, study_name: str) -> Path:
    return policy_gradient_suite_root(str(fe_mode), str(study_name))


def _load_reset_options_json(path: str | None) -> list[dict]:
    if path is None:
        return []
    with open(Path(path), "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        for key in ("signals", "reset_options", "scenarios"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise TypeError(f"Reset-options JSON must contain a list, got {type(payload).__name__}")
    options = []
    for row in payload:
        row = dict(row)
        if isinstance(row.get("reset_options"), dict):
            merged = dict(row["reset_options"])
            if "name" in row and "name" not in merged:
                merged["name"] = row["name"]
            row = merged
        options.append(row)
    return options


def _row_from_summary(summary: dict, variant_name: str) -> dict:
    return {
        "agent": str(summary["algo"]),
        "study_family": "baselines",
        "variant_name": variant_name,
        "state_variant": str(summary["state_variant"]),
        "reward_variant": str(summary["reward_variant"]),
        "tracking_rmse_m": float(summary["tracking_rmse_m"]),
        "transparency_rmse_w": float(summary["transparency_rmse_w"]),
        "pre_switch_tracking_rmse_m": float(summary["pre_switch_tracking_rmse_m"]),
        "post_switch_tracking_rmse_m": float(summary["post_switch_tracking_rmse_m"]),
        "pre_switch_transparency_rmse_w": float(summary["pre_switch_transparency_rmse_w"]),
        "post_switch_transparency_rmse_w": float(summary["post_switch_transparency_rmse_w"]),
        "mean_reward": float(summary["mean_reward"]),
        "invalid_episode_rate": float(summary["invalid_episode_rate"]),
        "model_path": str(summary["model_path"]),
        "out_dir": str(summary["out_dir"]),
    }


def main() -> None:
    require_sb3()

    parser = argparse.ArgumentParser(description="Run policy-gradient baselines for matlab_env_python_replica.")
    parser.add_argument("--algo", choices=PG_ALGO_CHOICES, required=True)
    parser.add_argument("--study-name", default="pg_run_01")
    parser.add_argument("--env-mode", choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING], default=cfg.ENV_MODE_CHANGING)
    parser.add_argument("--episode-duration", type=float, default=30.0)
    parser.add_argument("--env-switch-time", type=float, default=10.0)
    parser.add_argument("--force-amp", type=float, default=5.0)
    parser.add_argument("--force-bias", type=float, default=15.0)
    parser.add_argument("--force-freq", type=float, default=cfg.FORCE_INPUT_FREQ)
    parser.add_argument("--force-freq-rad", type=float, default=6.0)
    parser.add_argument("--force-phase", type=float, default=0.0)
    parser.add_argument("--force-waveform", choices=["sine", "cosine", "square", "ramp", "multisine"], default="sine")
    parser.add_argument("--fe-mode", choices=FE_MODE_CHOICES, default=FE_MODE_GUI)
    parser.add_argument("--reset-position-mode", choices=["midpoint", "zero"], default="midpoint")
    parser.add_argument("--stroke-limit-mode", choices=["terminate", "clamp"], default="clamp")
    parser.add_argument("--reward-variant", default="eqgrad_t40_tr40_nojerk")
    parser.add_argument("--state-variant", default="S0_baseline_full10")
    parser.add_argument("--reward-spec-json", default=None)
    parser.add_argument("--state-spec-json", default=None)
    parser.add_argument("--train-reset-options-json", default=None)
    parser.add_argument("--eval-reset-options-json", default=None)
    parser.add_argument("--train-episodes", type=int, default=2500)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--parallel-envs", type=int, default=None)
    parser.add_argument("--vec-env", choices=["auto", "dummy", "subproc"], default="auto")
    parser.add_argument("--ppo-n-steps", type=int, default=None)
    parser.add_argument("--ppo-batch-size", type=int, default=None)
    parser.add_argument("--ppo-n-epochs", type=int, default=None)
    parser.add_argument("--ppo-device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--eval-every-episodes", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    parser.add_argument("--disable-stroke-limit", action="store_true")
    args = parser.parse_args()

    worker_torch_threads = runner._configure_process_env(int(args.worker_torch_threads))
    train_reset_options_pool = _load_reset_options_json(args.train_reset_options_json)
    eval_reset_options_schedule = _load_reset_options_json(args.eval_reset_options_json)
    suite_root = _suite_root(str(args.fe_mode), str(args.study_name))
    suite_root.mkdir(parents=True, exist_ok=True)
    env_kwargs = runner._canonical_env_kwargs(args)
    runner._log(f"policy_gradient_experiment_root={suite_root}")
    if suite_root.name != str(args.study_name):
        runner._log(
            f"policy_gradient_experiment_root_alias requested={args.study_name} resolved={suite_root.name}"
        )
    runner._log(f"canonical_env={env_kwargs}")
    runner._log(
        f"algo={args.algo} | parallel_workers={int(args.parallel_workers)} | "
        f"worker_torch_threads={worker_torch_threads}"
    )
    save_json(
        suite_root / "suite_manifest.json",
        {
            "study_name": args.study_name,
            "algo": str(args.algo),
            "env_kwargs": env_kwargs,
            "train_episodes": int(args.train_episodes),
            "total_timesteps": None if args.total_timesteps is None else int(args.total_timesteps),
            "parallel_envs": args.parallel_envs,
            "vec_env": str(args.vec_env),
            "ppo_n_steps": args.ppo_n_steps,
            "ppo_batch_size": args.ppo_batch_size,
            "ppo_n_epochs": args.ppo_n_epochs,
            "ppo_device": str(args.ppo_device),
            "test_episodes": int(args.test_episodes),
            "seed": int(args.seed),
            "fe_mode": str(args.fe_mode),
            "reward_variant": str(args.reward_variant),
            "state_variant": str(args.state_variant),
            "reward_spec_json": None if args.reward_spec_json is None else str(args.reward_spec_json),
            "state_spec_json": None if args.state_spec_json is None else str(args.state_spec_json),
            "train_reset_options_json": None if args.train_reset_options_json is None else str(args.train_reset_options_json),
            "eval_reset_options_json": None if args.eval_reset_options_json is None else str(args.eval_reset_options_json),
            "train_signal_count": int(len(train_reset_options_pool)),
            "eval_signal_count": int(len(eval_reset_options_schedule)),
            "parallel_workers": int(args.parallel_workers),
            "worker_torch_threads": int(worker_torch_threads),
        },
    )

    stage_dir = suite_root / "00b"
    runner._log("stage baselines")
    if args.skip_existing and stage_completed(stage_dir):
        runner._log(f"skip stage baselines: {stage_dir}")
        return

    state_variant = get_policy_gradient_state_variant(str(args.state_variant), args.state_spec_json)
    reward_variant = get_policy_gradient_reward_variant(str(args.reward_variant), args.reward_spec_json)
    variant_name = f"{algo_display_name(args.algo)}_{reward_variant.name}_30s10s"
    out_dir = stage_dir / algo_output_dir_name(str(args.algo))
    result = train_policy_gradient_variant(
        algo=str(args.algo),
        out_dir=out_dir,
        env_mode=str(args.env_mode),
        env_kwargs=env_kwargs,
        state_variant=state_variant,
        reward_variant=reward_variant,
        total_episodes=int(args.train_episodes),
        test_episodes=int(args.test_episodes),
        seed=int(args.seed),
        label=f"{algo_display_name(args.algo)}_baseline_{reward_variant.name}",
        total_timesteps=args.total_timesteps,
        parallel_envs=args.parallel_envs,
        eval_every_episodes=int(args.eval_every_episodes),
        vec_env_type=str(args.vec_env),
        ppo_n_steps=args.ppo_n_steps,
        ppo_batch_size=args.ppo_batch_size,
        ppo_n_epochs=args.ppo_n_epochs,
        ppo_device=str(args.ppo_device),
        train_reset_options_pool=train_reset_options_pool,
        eval_reset_options_schedule=eval_reset_options_schedule,
    )
    summary_path = out_dir / "l" / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    row = _row_from_summary(summary, variant_name)
    save_json(
        stage_dir / "study_manifest.json",
        {
            "stage": "baselines",
            "rows": [row],
            "algo": str(args.algo),
            "algo_tag": algo_notebook_tag(str(args.algo)),
            "baseline": row,
        },
    )
    stage_summary_rows_to_csv([row], stage_dir / "study_summary.csv")
    runner._log(
        f"policy-gradient baseline complete | algo={args.algo} | "
        f"track={result.tracking_rmse_m:.4f} m | transp={result.transparency_rmse_w:.4f}"
    )


if __name__ == "__main__":
    main()
