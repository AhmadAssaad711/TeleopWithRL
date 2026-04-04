"""Compatibility wrapper for the organized saved-DQN evaluation runner."""

from __future__ import annotations

try:
    from .experiments.saved_dqn_eval import *  # noqa: F401,F403
    from .experiments.saved_dqn_eval import main
except ImportError:  # pragma: no cover - direct script execution
    from experiments.saved_dqn_eval import *  # type: ignore # noqa: F401,F403
    from experiments.saved_dqn_eval import main  # type: ignore


if __name__ == "__main__":
    main()
