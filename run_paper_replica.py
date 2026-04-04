"""Compatibility wrapper for the organized paper replica runner."""

from __future__ import annotations

try:
    from .experiments.paper_replica import *  # noqa: F401,F403
    from .experiments.paper_replica import run_paper_replica
except ImportError:  # pragma: no cover - direct script execution
    from experiments.paper_replica import *  # type: ignore # noqa: F401,F403
    from experiments.paper_replica import run_paper_replica  # type: ignore


if __name__ == "__main__":
    run_paper_replica(seed=0)
