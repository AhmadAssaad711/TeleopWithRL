"""One-command runner for all requested comparisons.

Runs:
1) Normal RL (constant environment) vs MRAC
2) Adaptive RL (changing environment) vs MRAC
"""

from __future__ import annotations

from benchmark_agents import run_full_benchmark


if __name__ == "__main__":
    run_full_benchmark()
