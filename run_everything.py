"""One-command entrypoint for all agents and all comparisons.

Runs:
1) RL (constant environment) training + evaluation
2) RL (changing/adaptive environment) training + evaluation
3) MRAC evaluation (constant + changing)
4) Head-to-head comparison plots + summary table
"""

from __future__ import annotations

from run_benchmark import main


if __name__ == "__main__":
    main()
