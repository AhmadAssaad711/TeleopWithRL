"""CLI entry point that launches policy-gradient baselines for both FE modes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _python_exe() -> str:
    python_rel = Path(".venv") / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    script_dir = Path(__file__).resolve().parent
    search_roots: list[Path] = []
    for start in (Path.cwd().resolve(), script_dir):
        for candidate in [start, *start.parents]:
            if candidate not in search_roots:
                search_roots.append(candidate)
    for root in search_roots:
        candidate = (root / python_rel).resolve()
        if candidate.exists() and _python_runs(candidate):
            return str(candidate)
    return sys.executable


def _python_runs(path: Path) -> bool:
    try:
        completed = subprocess.run(
            [str(path), "-c", "import sys"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _module_cmd(fe_mode: str, study_name: str, args) -> list[str]:
    cmd = [
        _python_exe(),
        "-m",
        "TeleopWithRL.matlab_env_python_replica.policy_gradient.scripts.run_experiments",
        "--algo",
        args.algo,
        "--study-name",
        study_name,
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
        "--state-variant",
        args.state_variant,
        "--fe-mode",
        fe_mode,
        "--reset-position-mode",
        str(args.reset_position_mode),
        "--stroke-limit-mode",
        str(args.stroke_limit_mode),
        "--train-episodes",
        str(args.train_episodes),
        "--eval-every-episodes",
        str(args.eval_every_episodes),
        "--test-episodes",
        str(args.test_episodes),
        "--seed",
        str(args.seed),
        "--parallel-workers",
        str(args.parallel_workers),
        "--worker-torch-threads",
        str(args.worker_torch_threads),
        "--vec-env",
        str(args.vec_env),
    ]
    if args.ppo_n_steps is not None:
        cmd.extend(["--ppo-n-steps", str(args.ppo_n_steps)])
    if args.ppo_batch_size is not None:
        cmd.extend(["--ppo-batch-size", str(args.ppo_batch_size)])
    if args.ppo_n_epochs is not None:
        cmd.extend(["--ppo-n-epochs", str(args.ppo_n_epochs)])
    cmd.extend(["--ppo-device", str(args.ppo_device)])
    if args.reward_spec_json is not None:
        cmd.extend(["--reward-spec-json", str(args.reward_spec_json)])
    if args.state_spec_json is not None:
        cmd.extend(["--state-spec-json", str(args.state_spec_json)])
    if args.total_timesteps is not None:
        cmd.extend(["--total-timesteps", str(args.total_timesteps)])
    if args.parallel_envs is not None:
        cmd.extend(["--parallel-envs", str(args.parallel_envs)])
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.disable_terminate_on_error:
        cmd.append("--disable-terminate-on-error")
    if args.disable_stroke_limit:
        cmd.append("--disable-stroke-limit")
    return cmd


def main() -> None:
    """Run the selected policy-gradient baseline once for each FE mode."""
    parser = argparse.ArgumentParser(
        description="Run policy-gradient baselines on both FE modes."
    )
    parser.add_argument("--algo", required=True)
    parser.add_argument("--study-name", default="pg_both_fe_01")
    parser.add_argument("--env-mode", choices=["constant_skin", "changing_skin_fat"], default="changing_skin_fat")
    parser.add_argument("--episode-duration", type=float, default=30.0)
    parser.add_argument("--env-switch-time", type=float, default=10.0)
    parser.add_argument("--force-amp", type=float, default=5.0)
    parser.add_argument("--force-bias", type=float, default=15.0)
    parser.add_argument("--force-freq", type=float, default=0.954929658551372)
    parser.add_argument("--force-freq-rad", type=float, default=6.0)
    parser.add_argument("--force-phase", type=float, default=0.0)
    parser.add_argument("--force-waveform", choices=["sine", "cosine", "square", "multisine"], default="sine")
    parser.add_argument("--reward-variant", default="eqgrad_t40_tr40_nojerk")
    parser.add_argument("--state-variant", default="S0_baseline_full10")
    parser.add_argument("--reward-spec-json", default=None)
    parser.add_argument("--state-spec-json", default=None)
    parser.add_argument("--reset-position-mode", choices=["midpoint", "zero"], default="midpoint")
    parser.add_argument("--stroke-limit-mode", choices=["terminate", "clamp"], default="clamp")
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
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    parser.add_argument("--disable-stroke-limit", action="store_true")
    args = parser.parse_args()

    switched_name = f"{args.study_name}_dyn"
    gui_name = f"{args.study_name}_gui"
    dyn_cmd = _module_cmd("switched_dynamics", switched_name, args)
    gui_cmd = _module_cmd("gui_skin_locked", gui_name, args)

    print(f"[pg both-fe] dyn -> {' '.join(dyn_cmd)}", flush=True)
    print(f"[pg both-fe] gui -> {' '.join(gui_cmd)}", flush=True)
    subprocess.run(dyn_cmd, check=True)
    subprocess.run(gui_cmd, check=True)


if __name__ == "__main__":
    main()
