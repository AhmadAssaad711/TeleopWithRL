"""Compatibility wrapper for the organized long study runner."""

from __future__ import annotations

try:
    from .experiments.long_study import *  # noqa: F401,F403
    from .experiments.long_study import main
except ImportError:  # pragma: no cover - direct script execution
    from experiments.long_study import *  # type: ignore # noqa: F401,F403
    from experiments.long_study import main  # type: ignore


if __name__ == "__main__":
    main()
