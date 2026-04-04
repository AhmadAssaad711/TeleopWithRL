"""Compatibility wrapper for the organized state-ablation runner."""

from __future__ import annotations

try:
    from .experiments.state_ablation import *  # noqa: F401,F403
    from .experiments.state_ablation import main
except ImportError:  # pragma: no cover - direct script execution
    from experiments.state_ablation import *  # type: ignore # noqa: F401,F403
    from experiments.state_ablation import main  # type: ignore


if __name__ == "__main__":
    main()
