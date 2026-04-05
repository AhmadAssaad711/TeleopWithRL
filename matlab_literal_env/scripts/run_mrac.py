from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL import config as cfg
    from TeleopWithRL.mrac_controller import run_mrac_on_teleop_env
    from TeleopWithRL.matlab_literal_env.simuoriginal_env import SimuOriginalReplicaEnv
    from TeleopWithRL.matlab_literal_env.scripts._common import (
        DEFAULT_RESULTS_ROOT,
        configure_results_root,
        replica_env_kwargs_from_args,
    )
else:
    from ... import config as cfg
    from ...mrac_controller import run_mrac_on_teleop_env
    from ..simuoriginal_env import SimuOriginalReplicaEnv
    from ._common import DEFAULT_RESULTS_ROOT, configure_results_root, replica_env_kwargs_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standard MRAC controller on the SimuOriginal replica env.")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--env-mode", choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING], default=cfg.ENV_MODE_CHANGING)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-duration", type=float, default=cfg.PAPER_EPISODE_DURATION)
    parser.add_argument("--env-switch-time", type=float, default=cfg.PAPER_ENV_SWITCH_TIME)
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    parser.add_argument("--force-amp", type=float, default=cfg.PAPER_FORCE_AMP)
    parser.add_argument("--force-bias", type=float, default=0.0)
    parser.add_argument("--force-freq", type=float, default=cfg.PAPER_FORCE_FREQ)
    parser.add_argument("--force-freq-rad", type=float, default=None)
    parser.add_argument("--force-phase", type=float, default=cfg.PAPER_FORCE_PHASE)
    parser.add_argument("--force-waveform", choices=["sine", "cosine", "square", "multisine"], default="sine")
    args = parser.parse_args()

    configure_results_root(args.results_root)
    env_kwargs = replica_env_kwargs_from_args(
        args,
        episode_duration=cfg.PAPER_EPISODE_DURATION,
        env_switch_time=cfg.PAPER_ENV_SWITCH_TIME,
    )
    env_kwargs["env_mode"] = args.env_mode
    run_mrac_on_teleop_env(
        seed=args.seed,
        env_cls=SimuOriginalReplicaEnv,
        env_kwargs=env_kwargs,
        run_subdir="replica",
        run_label="SimuOriginalReplicaEnv",
        configure_default_inputs=False,
    )


if __name__ == "__main__":
    main()
