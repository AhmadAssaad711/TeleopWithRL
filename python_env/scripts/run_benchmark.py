from __future__ import annotations

import argparse

from ._common import configure_results_root, DEFAULT_RESULTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standard DQN/Q-learning/MRAC benchmark on the active Python TeleopEnv.")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()

    configure_results_root(args.results_root)

    from ...benchmark_agents import run_full_benchmark

    run_full_benchmark()


if __name__ == "__main__":
    main()
