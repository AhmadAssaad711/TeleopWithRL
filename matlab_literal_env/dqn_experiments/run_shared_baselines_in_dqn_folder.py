from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL import config as cfg
    from TeleopWithRL.matlab_literal_env.simuoriginal_replica import FE_MODE_CHOICES, FE_MODE_GUI
    from TeleopWithRL.matlab_literal_env.scripts import run_replica_studies as runner
    from TeleopWithRL.matlab_literal_env.studies.common import save_json
    from TeleopWithRL.matlab_literal_env.studies.dqn_state_variants import get_dqn_state_variant
    from TeleopWithRL.matlab_literal_env.studies.rewarding import reward_variant_from_name
else:
    from ... import config as cfg
    from ..simuoriginal_replica import FE_MODE_CHOICES, FE_MODE_GUI
    from ..scripts import run_replica_studies as runner
    from ..studies.common import save_json
    from ..studies.dqn_state_variants import get_dqn_state_variant
    from ..studies.rewarding import reward_variant_from_name


DEFAULT_SWEEP_Q_EPISODES = 5_000
DEFAULT_SWEEP_DQN_EPISODES = 2_500
DEFAULT_PARALLEL_WORKERS = max(1, min(8, os.cpu_count() or 1))
DEFAULT_DQN_PARALLEL_ENVS = max(1, min(8, os.cpu_count() or 1))
DEFAULT_BASELINE_EPISODE_DURATION = 20.0
DEFAULT_BASELINE_ENV_SWITCH_TIME = 10.0


def _suite_root(fe_mode: str, study_name: str) -> Path:
    base_dir = Path(__file__).resolve().parent / "results"
    fe_dir = "dyn" if str(fe_mode) == "switched_dynamics" else "gui"
    return base_dir / fe_dir / study_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the shared matlab_literal_env baselines stage, but save under dqn_experiments."
    )
    parser.add_argument("--study-name", default="shared_baseline_01")
    parser.add_argument("--env-mode", choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING], default=cfg.ENV_MODE_CHANGING)
    parser.add_argument("--episode-duration", type=float, default=DEFAULT_BASELINE_EPISODE_DURATION)
    parser.add_argument("--env-switch-time", type=float, default=DEFAULT_BASELINE_ENV_SWITCH_TIME)
    parser.add_argument("--force-amp", type=float, default=5.0)
    parser.add_argument("--force-bias", type=float, default=5.0)
    parser.add_argument("--force-freq", type=float, default=cfg.FORCE_INPUT_FREQ)
    parser.add_argument("--force-freq-rad", type=float, default=0.5)
    parser.add_argument("--force-phase", type=float, default=0.0)
    parser.add_argument("--force-waveform", choices=["sine", "cosine", "square", "ramp", "multisine"], default="sine")
    parser.add_argument("--fe-mode", choices=FE_MODE_CHOICES, default=FE_MODE_GUI)
    parser.add_argument("--reward-variant", default="baseline_cfg")
    parser.add_argument("--state-variant", default="S0_baseline_full10")
    parser.add_argument("--legacy-baseline-env", action="store_true")
    parser.add_argument("--q-episodes", type=int, default=DEFAULT_SWEEP_Q_EPISODES)
    parser.add_argument("--dqn-episodes", type=int, default=DEFAULT_SWEEP_DQN_EPISODES)
    parser.add_argument("--dqn-parallel-envs", type=int, default=DEFAULT_DQN_PARALLEL_ENVS)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    args = parser.parse_args()

    worker_torch_threads = runner._configure_process_env(int(args.worker_torch_threads))
    suite_root = _suite_root(str(args.fe_mode), str(args.study_name))
    suite_root.mkdir(parents=True, exist_ok=True)
    env_kwargs = runner._canonical_env_kwargs(args)

    runner._log(f"dqn_experiment_shared_baseline_root={suite_root}")
    runner._log(f"canonical_env={env_kwargs}")
    runner._log(
        f"parallel_workers={int(args.parallel_workers)} | "
        f"worker_torch_threads={worker_torch_threads}"
    )
    save_json(
        suite_root / "suite_manifest.json",
        {
            "study_name": args.study_name,
            "stage": "baselines",
            "env_kwargs": env_kwargs,
            "q_episodes": args.q_episodes,
            "dqn_episodes": args.dqn_episodes,
            "dqn_parallel_envs": int(args.dqn_parallel_envs),
            "test_episodes": args.test_episodes,
            "seed": args.seed,
            "fe_mode": str(args.fe_mode),
            "reward_variant": str(args.reward_variant),
            "state_variant": str(args.state_variant),
            "parallel_workers": int(args.parallel_workers),
            "worker_torch_threads": int(worker_torch_threads),
            "shared_runner_source": "TeleopWithRL.matlab_literal_env.scripts.run_replica_studies",
        },
    )
    reward_variant = reward_variant_from_name(str(args.reward_variant))
    state_variant = get_dqn_state_variant(str(args.state_variant))
    stage_dir = suite_root / "00b"
    runner._log("stage baselines")
    if args.skip_existing and (stage_dir / "study_manifest.json").exists():
        runner._log(f"skip stage baselines: {stage_dir}")
        return

    tasks = [
        {
            "order": 2,
            "name": f"DQN_baseline_{state_variant.name}_{reward_variant.name}",
            "family": "dqn",
            "out_dir": str(stage_dir / "dqn"),
            "env_mode": args.env_mode,
            "env_kwargs": env_kwargs,
            "state_variant": state_variant.name,
            "reward_variant": reward_variant.name,
            "total_episodes": args.dqn_episodes,
            "test_episodes": args.test_episodes,
            "seed": args.seed,
            "label": f"DQN_baseline_{state_variant.name}_{reward_variant.name}",
            "resume": args.resume,
            "parallel_envs": int(args.dqn_parallel_envs),
        },
    ]
    payloads = runner._run_stage_train_tasks(
        stage_name="baselines",
        tasks=tasks,
        parallel_workers=1,
        worker_torch_threads=int(args.worker_torch_threads),
    )
    summary = payloads[0]["summary"]
    row = runner._row_from_summary(
        "dqn",
        "baselines",
        f"DQN_baseline_{state_variant.name}_{reward_variant.name}",
        summary,
    )
    save_json(
        stage_dir / "study_manifest.json",
        {
            "stage": "baselines",
            "rows": [row],
            "dqn_baseline": row,
        },
    )
    runner.stage_summary_rows_to_csv([row], stage_dir / "study_summary.csv")


if __name__ == "__main__":
    main()
