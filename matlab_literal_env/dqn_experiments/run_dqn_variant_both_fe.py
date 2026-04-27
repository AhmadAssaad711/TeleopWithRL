from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL import config as cfg
    from TeleopWithRL.matlab_literal_env.simuoriginal_replica import FE_MODE_CHOICES
    from TeleopWithRL.matlab_literal_env.scripts import run_replica_studies as runner
    from TeleopWithRL.matlab_literal_env.studies.common import save_json
    from TeleopWithRL.matlab_literal_env.studies.dqn import train_dqn_variant
    from TeleopWithRL.matlab_literal_env.studies.dqn_state_variants import get_dqn_state_variant
    from TeleopWithRL.matlab_literal_env.studies.rewarding import reward_variant_from_name
else:
    from ... import config as cfg
    from ..simuoriginal_replica import FE_MODE_CHOICES
    from ..scripts import run_replica_studies as runner
    from ..studies.common import save_json
    from ..studies.dqn import train_dqn_variant
    from ..studies.dqn_state_variants import get_dqn_state_variant
    from ..studies.rewarding import reward_variant_from_name


def _suite_root(fe_mode: str, study_name: str) -> Path:
    base_dir = Path(__file__).resolve().parent / "results"
    fe_dir = "dyn" if str(fe_mode) == "switched_dynamics" else "gui"
    return base_dir / fe_dir / study_name


def _train_one(task: dict) -> dict:
    state_variant = get_dqn_state_variant(str(task["state_variant"]))
    reward_variant = reward_variant_from_name(str(task["reward_variant"]))
    out_dir = Path(str(task["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_dqn_variant(
        out_dir=out_dir,
        env_mode=str(task["env_mode"]),
        env_kwargs=dict(task["env_kwargs"]),
        state_variant=state_variant,
        reward_variant=reward_variant,
        total_episodes=int(task["dqn_episodes"]),
        test_episodes=int(task["test_episodes"]),
        seed=int(task["seed"]),
        label=str(task["label"]),
        parallel_envs=int(task["dqn_parallel_envs"]),
    )
    manifest = {
        "study_name": str(task["study_name"]),
        "state_variant": state_variant.name,
        "feature_names": list(state_variant.feature_names),
        "reward_variant": reward_variant.name,
        "fe_mode": str(task["fe_mode"]),
        "dqn_episodes": int(task["dqn_episodes"]),
        "dqn_parallel_envs": int(task["dqn_parallel_envs"]),
        "test_episodes": int(task["test_episodes"]),
        "seed": int(task["seed"]),
        "summary": {
            "mean_reward": float(result.mean_reward),
            "tracking_rmse_m": float(result.tracking_rmse_m),
            "transparency_rmse_w": float(result.transparency_rmse_w),
            "pre_switch_tracking_rmse_m": float(result.pre_switch_tracking_rmse_m),
            "post_switch_tracking_rmse_m": float(result.post_switch_tracking_rmse_m),
            "pre_switch_transparency_rmse_w": float(result.pre_switch_transparency_rmse_w),
            "post_switch_transparency_rmse_w": float(result.post_switch_transparency_rmse_w),
            "invalid_episode_rate": float(result.invalid_episode_rate),
            "model_path": str(result.model_path),
            "out_dir": str(result.out_dir),
        },
    }
    save_json(out_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one isolated DQN state variant on both FE modes."
    )
    parser.add_argument("--study-name", default="s8_both_fe_01")
    parser.add_argument("--state-variant", default="S8_absolute_posvel_forces")
    parser.add_argument("--reward-variant", default="baseline_cfg")
    parser.add_argument("--env-mode", choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING], default=cfg.ENV_MODE_CHANGING)
    parser.add_argument("--episode-duration", type=float, default=float(cfg.PAPER_EPISODE_DURATION))
    parser.add_argument("--env-switch-time", type=float, default=float(cfg.PAPER_ENV_SWITCH_TIME))
    parser.add_argument("--force-amp", type=float, default=5.0)
    parser.add_argument("--force-bias", type=float, default=5.0)
    parser.add_argument("--force-freq", type=float, default=cfg.FORCE_INPUT_FREQ)
    parser.add_argument("--force-freq-rad", type=float, default=0.5)
    parser.add_argument("--force-phase", type=float, default=0.0)
    parser.add_argument("--force-waveform", choices=["sine", "cosine", "square", "multisine"], default="sine")
    parser.add_argument("--dqn-episodes", type=int, default=2500)
    parser.add_argument("--dqn-parallel-envs", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--run-parallel", action="store_true")
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    args = parser.parse_args()

    runner._configure_process_env(int(args.worker_torch_threads))
    env_kwargs = runner._canonical_env_kwargs(args)
    get_dqn_state_variant(str(args.state_variant))
    reward_variant_from_name(str(args.reward_variant))

    tasks = []
    for fe_mode, suffix in (("switched_dynamics", "dyn"), ("gui_skin_locked", "gui")):
        run_name = f"{args.study_name}_{suffix}"
        suite_root = _suite_root(fe_mode, run_name)
        suite_root.mkdir(parents=True, exist_ok=True)
        env_kwargs_mode = dict(env_kwargs)
        env_kwargs_mode["reset_options"] = dict(env_kwargs["reset_options"])
        env_kwargs_mode["reset_options"]["fe_mode"] = fe_mode
        task = {
            "study_name": run_name,
            "state_variant": str(args.state_variant),
            "reward_variant": str(args.reward_variant),
            "fe_mode": fe_mode,
            "env_mode": args.env_mode,
            "env_kwargs": env_kwargs_mode,
            "dqn_episodes": int(args.dqn_episodes),
            "dqn_parallel_envs": int(args.dqn_parallel_envs),
            "test_episodes": int(args.test_episodes),
            "seed": int(args.seed),
            "label": f"{args.state_variant}_{fe_mode}",
            "out_dir": str(suite_root),
        }
        tasks.append(task)
        print(f"[dqn variant both-fe] {fe_mode} -> {suite_root}", flush=True)

    if args.run_parallel:
        with cf.ProcessPoolExecutor(
            max_workers=2,
            initializer=runner._worker_runtime_init,
            initargs=(int(args.worker_torch_threads),),
        ) as executor:
            list(executor.map(_train_one, tasks))
        return

    for task in tasks:
        _train_one(task)


if __name__ == "__main__":
    main()
