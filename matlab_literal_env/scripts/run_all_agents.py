from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL import config as cfg
    from TeleopWithRL.matlab_literal_env.scripts._common import DEFAULT_RESULTS_ROOT
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    from ... import config as cfg
    from ._common import DEFAULT_RESULTS_ROOT


def _append_flag(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _append_bool_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def _shared_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = [
        "--results-root",
        args.results_root,
        "--env-mode",
        args.env_mode,
        "--force-amp",
        str(args.force_amp),
        "--force-bias",
        str(args.force_bias),
        "--force-phase",
        str(args.force_phase),
        "--force-waveform",
        args.force_waveform,
    ]
    _append_flag(flags, "--episode-duration", args.episode_duration)
    _append_flag(flags, "--env-switch-time", args.env_switch_time)
    _append_bool_flag(flags, "--disable-terminate-on-error", args.disable_terminate_on_error)
    if args.force_freq_rad is not None:
        flags.extend(["--force-freq-rad", str(args.force_freq_rad)])
    else:
        flags.extend(["--force-freq", str(args.force_freq)])
    return flags


def _run(cmd: list[str]) -> None:
    print(f"\n[run_all_agents] Running: {shlex.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(_PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MRAC, Q-learning, and DQN sequentially on the SimuOriginal replica env."
    )
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--env-mode",
        choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING],
        default=cfg.ENV_MODE_CHANGING,
    )
    parser.add_argument("--seed", type=int, default=0, help="MRAC seed.")
    parser.add_argument("--q-episodes", type=int, default=cfg.NUM_EPISODES)
    parser.add_argument("--dqn-episodes", type=int, default=cfg.DQN_NUM_EPISODES)
    parser.add_argument("--episode-duration", type=float, default=None)
    parser.add_argument("--env-switch-time", type=float, default=None)
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    parser.add_argument("--force-amp", type=float, default=cfg.FORCE_INPUT_AMP)
    parser.add_argument("--force-bias", type=float, default=0.0)
    parser.add_argument("--force-freq", type=float, default=cfg.FORCE_INPUT_FREQ)
    parser.add_argument("--force-freq-rad", type=float, default=None)
    parser.add_argument("--force-phase", type=float, default=cfg.FORCE_INPUT_PHASE)
    parser.add_argument(
        "--force-waveform",
        choices=["sine", "cosine", "square", "multisine"],
        default="sine",
    )
    args = parser.parse_args()

    shared_flags = _shared_flags(args)
    python = sys.executable

    mrac_cmd = [
        python,
        "-m",
        "TeleopWithRL.matlab_literal_env.scripts.run_mrac",
        "--seed",
        str(args.seed),
        *shared_flags,
    ]
    q_cmd = [
        python,
        "-m",
        "TeleopWithRL.matlab_literal_env.scripts.train_q_learning",
        "--episodes",
        str(args.q_episodes),
        *shared_flags,
    ]
    dqn_cmd = [
        python,
        "-m",
        "TeleopWithRL.matlab_literal_env.scripts.train_dqn",
        "--episodes",
        str(args.dqn_episodes),
        *shared_flags,
    ]

    _run(mrac_cmd)
    _run(q_cmd)
    _run(dqn_cmd)

    print("\n[run_all_agents] All replica-agent runs completed.")


if __name__ == "__main__":
    main()
