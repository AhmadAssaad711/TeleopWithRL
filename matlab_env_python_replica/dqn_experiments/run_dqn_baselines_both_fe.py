from __future__ import annotations

import argparse
import subprocess
import sys

from TeleopWithRL import config as cfg


def _python_exe() -> str:
    return sys.executable


def _module_cmd(fe_mode: str, study_name: str, args) -> list[str]:
    cmd = [
        _python_exe(),
        "-m",
        "TeleopWithRL.matlab_env_python_replica.dqn_experiments.run_dqn_experiments",
        "--study-name",
        study_name,
        "--stage",
        "baselines",
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
        "--reward-variant",
        args.reward_variant,
        "--fe-mode",
        fe_mode,
        "--reset-position-mode",
        str(args.reset_position_mode),
        "--stroke-limit-mode",
        str(args.stroke_limit_mode),
        "--dqn-episodes",
        str(args.dqn_episodes),
        "--dqn-parallel-envs",
        str(args.dqn_parallel_envs),
        "--test-episodes",
        str(args.test_episodes),
        "--seed",
        str(args.seed),
        "--parallel-workers",
        str(args.parallel_workers),
        "--worker-torch-threads",
        str(args.worker_torch_threads),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.disable_terminate_on_error:
        cmd.append("--disable-terminate-on-error")
    if args.disable_stroke_limit:
        cmd.append("--disable-stroke-limit")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DQN baseline on both FE modes in isolated dqn_experiments folders."
    )
    parser.add_argument("--study-name", default="baseline_both_fe_01")
    parser.add_argument("--env-mode", choices=["constant_skin", "changing_skin_fat"], default="changing_skin_fat")
    parser.add_argument("--episode-duration", type=float, default=60.0)
    parser.add_argument("--env-switch-time", type=float, default=cfg.PAPER_ENV_SWITCH_TIME)
    parser.add_argument("--force-amp", type=float, default=5.0)
    parser.add_argument("--force-bias", type=float, default=5.0)
    parser.add_argument("--force-freq", type=float, default=0.07957747154594767)
    parser.add_argument("--force-freq-rad", type=float, default=0.5)
    parser.add_argument("--force-phase", type=float, default=0.0)
    parser.add_argument("--force-waveform", choices=["sine", "cosine", "square", "multisine"], default="sine")
    parser.add_argument("--reward-variant", default="baseline_cfg")
    parser.add_argument("--reset-position-mode", choices=["midpoint", "zero"], default="midpoint")
    parser.add_argument("--stroke-limit-mode", choices=["terminate", "clamp"], default="terminate")
    parser.add_argument("--dqn-episodes", type=int, default=2500)
    parser.add_argument("--dqn-parallel-envs", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    parser.add_argument("--disable-stroke-limit", action="store_true")
    parser.add_argument("--run-parallel", action="store_true")
    args = parser.parse_args()

    switched_name = f"{args.study_name}_dyn"
    gui_name = f"{args.study_name}_gui"
    dyn_cmd = _module_cmd("switched_dynamics", switched_name, args)
    gui_cmd = _module_cmd("gui_skin_locked", gui_name, args)

    print(f"[dqn both-fe] dyn -> {' '.join(dyn_cmd)}", flush=True)
    print(f"[dqn both-fe] gui -> {' '.join(gui_cmd)}", flush=True)

    if args.run_parallel:
        procs = [
            subprocess.Popen(dyn_cmd),
            subprocess.Popen(gui_cmd),
        ]
        codes = [proc.wait() for proc in procs]
        if any(code != 0 for code in codes):
            raise SystemExit(max(codes))
        return

    subprocess.run(dyn_cmd, check=True)
    subprocess.run(gui_cmd, check=True)


if __name__ == "__main__":
    main()
