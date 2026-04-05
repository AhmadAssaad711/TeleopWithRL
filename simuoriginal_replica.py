"""Compatibility wrapper for the organized MATLAB-literal SimuOriginal replica."""

from __future__ import annotations

try:
    from .matlab_literal_env.simuoriginal_replica import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - direct script execution
    from matlab_literal_env.simuoriginal_replica import *  # type: ignore # noqa: F401,F403
