from __future__ import annotations

import argparse

from ._common import configure_results_root, DEFAULT_RESULTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test tabular Q-learning on the active Python TeleopEnv.")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--env-mode", choices=["constant_skin", "changing_skin_fat"], default="changing_skin_fat")
    parser.add_argument("--master-input-mode", choices=["reference", "force"], default="force")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    configure_results_root(args.results_root)

    from ... import config as cfg
    from ...train_q_learning import train_q_learning

    train_q_learning(
        total_episodes=args.episodes if args.episodes is not None else cfg.NUM_EPISODES,
        env_mode=args.env_mode,
        master_input_mode=args.master_input_mode,
    )


if __name__ == "__main__":
    main()
