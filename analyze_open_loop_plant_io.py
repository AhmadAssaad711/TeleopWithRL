"""Compatibility wrapper for the organized open-loop I/O analysis runner."""

from __future__ import annotations

try:
    from .experiments.open_loop_io import *  # noqa: F401,F403
    from .experiments.open_loop_io import main
except ImportError:  # pragma: no cover - direct script execution
    from experiments.open_loop_io import *  # type: ignore # noqa: F401,F403
    from experiments.open_loop_io import main  # type: ignore


if __name__ == "__main__":
    main()
