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
else:
    from ... import config as cfg
    from ..simuoriginal_replica import FE_MODE_CHOICES, FE_MODE_GUI
    from . import run_replica_studies as runner


DEFAULT_SWEEP_Q_EPISODES = 5_000
DEFAULT_SWEEP_DQN_EPISODES = 2_500
DEFAULT_PARALLEL_WORKERS = max(1, min(8, os.cpu_count() or 1))
DEFAULT_DQN_PARALLEL_ENVS = max(1, min(8, os.cpu_count() or 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full matlab_literal_env study suite in series with canonical defaults."
    )
    parser.add_argument("--study-name", default="r60s")
    parser.add_argument("--stage", choices=["all", "baselines", "ql_state", "ql_reward", "dqn_reward", "dqn_state", "eval"], default="all")
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
    parser.add_argument("--q-episodes", type=int, default=DEFAULT_SWEEP_Q_EPISODES)
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
    args = parser.parse_args()

    forwarded = [
        "--study-name",
        args.study_name,
        "--stage",
        args.stage,
        "--env-mode",
        args.env_mode,
        "--episode-duration",
        str(args.episode_duration),
        "--env-switch-time",
        str(args.env_switch_time),
        "--force-amp",
        str(args.force_amp),
        "--force-bias",
        str(args.force_bias),
        "--force-freq",
        str(args.force_freq),
        "--force-freq-rad",
        str(args.force_freq_rad),
        "--force-phase",
        str(args.force_phase),
        "--force-waveform",
        args.force_waveform,
        "--fe-mode",
        args.fe_mode,
        "--q-episodes",
        str(args.q_episodes),
        "--dqn-episodes",
        str(args.dqn_episodes),
        "--dqn-parallel-envs",
        str(args.dqn_parallel_envs),
        "--test-episodes",
        str(args.test_episodes),
        "--seed",
        str(args.seed),
        "--noise-std",
        str(args.noise_std),
        "--parallel-workers",
        str(args.parallel_workers),
        "--worker-torch-threads",
        str(args.worker_torch_threads),
    ]
    if args.resume:
        forwarded.append("--resume")
    if args.skip_existing:
        forwarded.append("--skip-existing")
    if args.full_grid:
        forwarded.append("--full-grid")
    if args.disable_terminate_on_error:
        forwarded.append("--disable-terminate-on-error")

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *forwarded]
        runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
