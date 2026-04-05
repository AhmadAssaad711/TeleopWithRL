"""Compatibility wrapper for the organized MATLAB-literal replica runner."""

from __future__ import annotations

try:
    from ..matlab_literal_env.scripts.run_simuoriginal_replica import *  # noqa: F401,F403
    from ..matlab_literal_env.scripts.run_simuoriginal_replica import main
except ImportError:  # pragma: no cover - direct script execution
    from matlab_literal_env.scripts.run_simuoriginal_replica import *  # type: ignore # noqa: F401,F403
    from matlab_literal_env.scripts.run_simuoriginal_replica import main  # type: ignore


if __name__ == "__main__":
    main()
