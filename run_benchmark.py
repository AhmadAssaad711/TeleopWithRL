"""Compatibility wrapper for the organized benchmark runner."""

from __future__ import annotations

try:
    from .experiments.benchmark import *  # noqa: F401,F403
    from .experiments.benchmark import main
except ImportError:  # pragma: no cover - direct script execution
    from experiments.benchmark import *  # type: ignore # noqa: F401,F403
    from experiments.benchmark import main  # type: ignore


if __name__ == "__main__":
    main()
