"""CLI entry point for DQN baselines, ablations, and saved-policy evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL.matlab_env_python_replica.config import config as cfg
    from TeleopWithRL.matlab_env_python_replica.environment.simuoriginal_replica import FE_MODE_CHOICES, FE_MODE_GUI
    from TeleopWithRL.matlab_env_python_replica.common import runner
    from TeleopWithRL.matlab_env_python_replica.common.study_utils import save_json, stage_completed, stage_summary_rows_to_csv
    from TeleopWithRL.matlab_env_python_replica.dqn.training import train_dqn_variant
    from TeleopWithRL.matlab_env_python_replica.dqn.state_variants import get_dqn_state_variant
    from TeleopWithRL.matlab_env_python_replica.common.rewarding import baseline_reward_variant, reward_variant_from_name
    from TeleopWithRL.matlab_env_python_replica.common.saved_policy_eval import evaluate_saved_policy, save_evaluation_bundle
else:
    from ...config import config as cfg
    from ...environment.simuoriginal_replica import FE_MODE_CHOICES, FE_MODE_GUI
    from ...common import runner
    from ...common.study_utils import save_json, stage_completed, stage_summary_rows_to_csv
    from ..training import train_dqn_variant
    from ..state_variants import get_dqn_state_variant
    from ...common.rewarding import baseline_reward_variant, reward_variant_from_name
    from ...common.saved_policy_eval import evaluate_saved_policy, save_evaluation_bundle


DEFAULT_SWEEP_DQN_EPISODES = 2_500
DEFAULT_PARALLEL_WORKERS = max(1, min(8, os.cpu_count() or 1))
DEFAULT_DQN_PARALLEL_ENVS = max(1, min(8, os.cpu_count() or 1))


def _duration_switch_tag(args) -> str:
    return f"{float(args.episode_duration):g}s_{float(args.env_switch_time):g}s"


def _dqn_suite_root(fe_mode: str, study_name: str) -> Path:
    # Results remain in the established data directory so existing notebooks
    # and saved-policy evaluations continue to find previous runs.
    base_dir = Path(__file__).resolve().parents[2] / "dqn_experiments"
    fe_dir = "dyn" if str(fe_mode) == "switched_dynamics" else "gui"
    return base_dir / "results" / fe_dir / study_name


def _run_dqn_baseline(suite_root: Path, args, env_kwargs: dict[str, Any]) -> dict[str, Any]:
    stage_dir = suite_root / "00b"
    runner._log("stage dqn_baseline")
    if args.skip_existing and stage_completed(stage_dir):
        runner._log(f"skip stage dqn_baseline: {stage_dir}")
        with open(stage_dir / "study_manifest.json", "r", encoding="utf-8") as fh:
            return json.load(fh)

    dqn_variant = get_dqn_state_variant("S0_baseline_full10")
    reward_variant = reward_variant_from_name(str(args.reward_variant))
    baseline_label = f"DQN_baseline_{reward_variant.name}_{_duration_switch_tag(args)}"
    tasks = [
        {
            "order": 1,
            "name": baseline_label,
            "family": "dqn",
            "out_dir": str(stage_dir / "dqn"),
            "env_mode": args.env_mode,
            "env_kwargs": env_kwargs,
            "state_variant": dqn_variant.name,
            "reward_variant": reward_variant.name,
            "total_episodes": args.dqn_episodes,
            "test_episodes": args.test_episodes,
            "seed": args.seed,
            "label": baseline_label,
            "resume": args.resume,
            "parallel_envs": int(args.dqn_parallel_envs),
        }
    ]
    payloads = runner._run_stage_train_tasks(
        stage_name="dqn_baseline",
        tasks=tasks,
        parallel_workers=1,
        worker_torch_threads=int(args.worker_torch_threads),
    )
    summary = payloads[0]["summary"]
    row = runner._row_from_summary("dqn", "baselines", baseline_label, summary)
    manifest = {
        "stage": "dqn_baseline",
        "rows": [row],
        "dqn_baseline": row,
    }
    save_json(stage_dir / "study_manifest.json", manifest)
    stage_summary_rows_to_csv([row], stage_dir / "study_summary.csv")
    runner._log(f"dqn_baseline complete | transp={row['transparency_rmse_w']:.4f}")
    return manifest


def _run_saved_policy_eval_dqn(suite_root: Path, args, dqn_best: dict[str, Any]) -> dict[str, Any]:
    stage_dir = suite_root / "30ev"
    runner._log("stage dqn_eval")
    if args.skip_existing and stage_completed(stage_dir):
        runner._log(f"skip stage dqn_eval: {stage_dir}")
        with open(stage_dir / "study_manifest.json", "r", encoding="utf-8") as fh:
            return json.load(fh)

    out_dir = stage_dir / "dqn"
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_rows, summary, policy_rows, policy_summary_payload, env_switch_time = evaluate_saved_policy(
        model_path=str(dqn_best["model_path"]),
        episodes=int(args.test_episodes),
        seed=int(args.seed) + 50_000,
        scenario_set=None,
        noise_std=float(args.noise_std),
    )
    save_evaluation_bundle(
        out_dir=out_dir,
        prefix="canon",
        episode_rows=episode_rows,
        summary=summary,
        policy_rows=policy_rows,
        policy_summary_payload=policy_summary_payload,
        env_switch_time=env_switch_time,
    )
    transfer_outputs = {}
    for scenario_set, prefix in (
        ("force_generalization_10", "gen10"),
        ("force_square_10", "square10"),
        ("force_noise_10", "noise10"),
    ):
        ep_rows, bundle_summary, bundle_policy_rows, bundle_policy_summary, switch_time = evaluate_saved_policy(
            model_path=str(dqn_best["model_path"]),
            episodes=int(args.test_episodes),
            seed=int(args.seed) + 60_000,
            scenario_set=scenario_set,
            noise_std=float(args.noise_std),
        )
        save_evaluation_bundle(
            out_dir=out_dir,
            prefix=prefix,
            episode_rows=ep_rows,
            summary=bundle_summary,
            policy_rows=bundle_policy_rows,
            policy_summary_payload=bundle_policy_summary,
            env_switch_time=switch_time,
        )
        transfer_outputs[scenario_set] = bundle_summary

    manifest = {
        "stage": "dqn_best_policy_eval",
        "agent_name": "dqn",
        "best_row": dqn_best,
        "canonical_summary": summary,
        "transfer_summaries": transfer_outputs,
        "out_dir": str(out_dir),
    }
    save_json(stage_dir / "study_manifest.json", manifest)
    stage_summary_rows_to_csv(
        [
            {
                "agent": "dqn",
                "study_family": "best_policy_evals",
                "variant_name": dqn_best["variant_name"],
                "state_variant": dqn_best["state_variant"],
                "reward_variant": dqn_best["reward_variant"],
                "tracking_rmse_m": dqn_best["tracking_rmse_m"],
                "transparency_rmse_w": dqn_best["transparency_rmse_w"],
                "pre_switch_tracking_rmse_m": dqn_best["pre_switch_tracking_rmse_m"],
                "post_switch_tracking_rmse_m": dqn_best["post_switch_tracking_rmse_m"],
                "pre_switch_transparency_rmse_w": dqn_best["pre_switch_transparency_rmse_w"],
                "post_switch_transparency_rmse_w": dqn_best["post_switch_transparency_rmse_w"],
                "mean_reward": dqn_best["mean_reward"],
                "invalid_episode_rate": dqn_best["invalid_episode_rate"],
                "model_path": dqn_best["model_path"],
                "out_dir": str(out_dir),
            }
        ],
        stage_dir / "study_summary.csv",
    )
    runner._log("dqn_eval complete")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DQN-only experiments for matlab_env_python_replica.")
    parser.add_argument("--study-name", default="run_01")
    parser.add_argument("--stage", choices=["all", "baselines", "dqn_reward", "dqn_state", "eval"], default="all")
    parser.add_argument("--env-mode", choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING], default=cfg.ENV_MODE_CHANGING)
    parser.add_argument("--episode-duration", type=float, default=float(cfg.PAPER_EPISODE_DURATION))
    parser.add_argument("--env-switch-time", type=float, default=float(cfg.PAPER_ENV_SWITCH_TIME))
    parser.add_argument("--force-amp", type=float, default=5.0)
    parser.add_argument("--force-bias", type=float, default=5.0)
    parser.add_argument("--force-freq", type=float, default=cfg.FORCE_INPUT_FREQ)
    parser.add_argument("--force-freq-rad", type=float, default=0.5)
    parser.add_argument("--force-phase", type=float, default=0.0)
    parser.add_argument("--force-waveform", choices=["sine", "cosine", "square", "ramp", "multisine"], default="sine")
    parser.add_argument("--fe-mode", choices=FE_MODE_CHOICES, default=FE_MODE_GUI)
    parser.add_argument("--reset-position-mode", choices=["midpoint", "zero"], default=None)
    parser.add_argument("--stroke-limit-mode", choices=["terminate", "clamp"], default="terminate")
    parser.add_argument("--reward-variant", default="baseline_cfg")
    parser.add_argument("--dqn-episodes", type=int, default=DEFAULT_SWEEP_DQN_EPISODES)
    parser.add_argument("--dqn-parallel-envs", type=int, default=DEFAULT_DQN_PARALLEL_ENVS)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    parser.add_argument("--disable-stroke-limit", action="store_true")
    args = parser.parse_args()

    worker_torch_threads = runner._configure_process_env(int(args.worker_torch_threads))
    suite_root = _dqn_suite_root(str(args.fe_mode), str(args.study_name))
    suite_root.mkdir(parents=True, exist_ok=True)
    env_kwargs = runner._canonical_env_kwargs(args)
    runner._log(f"dqn_experiment_root={suite_root}")
    runner._log(f"canonical_env={env_kwargs}")
    runner._log(
        f"parallel_workers={int(args.parallel_workers)} | "
        f"worker_torch_threads={worker_torch_threads}"
    )
    save_json(
        suite_root / "suite_manifest.json",
        {
            "study_name": args.study_name,
            "stage": args.stage,
            "env_kwargs": env_kwargs,
            "dqn_episodes": args.dqn_episodes,
            "dqn_parallel_envs": int(args.dqn_parallel_envs),
            "test_episodes": args.test_episodes,
            "seed": args.seed,
            "full_grid": bool(args.full_grid),
            "fe_mode": str(args.fe_mode),
            "reward_variant": str(args.reward_variant),
            "parallel_workers": int(args.parallel_workers),
            "worker_torch_threads": int(worker_torch_threads),
        },
    )

    baseline = _run_dqn_baseline(suite_root, args, env_kwargs)
    if args.stage == "baselines":
        return
    dqn_reward = runner._run_dqn_reward_study(suite_root, args, env_kwargs, baseline["dqn_baseline"])
    if args.stage == "dqn_reward":
        return
    dqn_state = runner._run_dqn_state_study(
        suite_root,
        args,
        env_kwargs,
        baseline["dqn_baseline"],
        dqn_reward["best"]["reward_variant"],
    )
    if args.stage == "dqn_state":
        return
    _run_saved_policy_eval_dqn(suite_root, args, dqn_state["best"])


if __name__ == "__main__":
    main()
