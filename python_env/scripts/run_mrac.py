from __future__ import annotations

import argparse

from ._common import configure_results_root, DEFAULT_RESULTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standard MRAC controller on the active Python TeleopEnv.")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configure_results_root(args.results_root)

    from ...mrac_controller import run_mrac_on_teleop_env

    run_mrac_on_teleop_env(seed=args.seed)


if __name__ == "__main__":
    main()
