from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL import config as cfg
    from TeleopWithRL.train_q_learning import train_q_learning
    from TeleopWithRL.matlab_literal_env.simuoriginal_env import SimuOriginalReplicaEnv
    from TeleopWithRL.matlab_literal_env.scripts._common import (
        DEFAULT_RESULTS_ROOT,
        configure_results_root,
        replica_env_kwargs_from_args,
    )
else:
    from ... import config as cfg
    from ...train_q_learning import train_q_learning
    from ..simuoriginal_env import SimuOriginalReplicaEnv
    from ._common import DEFAULT_RESULTS_ROOT, configure_results_root, replica_env_kwargs_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test tabular Q-learning on the SimuOriginal replica env.")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--env-mode", choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING], default=cfg.ENV_MODE_CHANGING)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--episode-duration", type=float, default=None)
    parser.add_argument("--env-switch-time", type=float, default=None)
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    parser.add_argument("--force-amp", type=float, default=cfg.FORCE_INPUT_AMP)
    parser.add_argument("--force-bias", type=float, default=0.0)
    parser.add_argument("--force-freq", type=float, default=cfg.FORCE_INPUT_FREQ)
    parser.add_argument("--force-freq-rad", type=float, default=None)
    parser.add_argument("--force-phase", type=float, default=cfg.FORCE_INPUT_PHASE)
    parser.add_argument("--force-waveform", choices=["sine", "cosine", "square", "multisine"], default="sine")
    args = parser.parse_args()

    configure_results_root(args.results_root)
    env_kwargs = replica_env_kwargs_from_args(args)
    train_q_learning(
        total_episodes=args.episodes if args.episodes is not None else cfg.NUM_EPISODES,
        env_mode=args.env_mode,
        master_input_mode=cfg.MASTER_INPUT_FORCE,
        env_cls=SimuOriginalReplicaEnv,
        env_kwargs=env_kwargs,
    )


if __name__ == "__main__":
    main()
