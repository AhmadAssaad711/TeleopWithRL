"""Stable result-path helpers for policy-gradient experiment artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


# Keep the existing artifact location stable while the source code is grouped
# under ``policy_gradient``. The old directory is data storage, not a second
# implementation of the policy-gradient code.
_REPLICA_ROOT = Path(__file__).resolve().parents[1]


_WINDOWS_SAFE_DIR_LIMIT = 240
_PG_LONGEST_SUFFIX = Path("00b") / "ppod" / "m"


def results_root() -> Path:
    return _REPLICA_ROOT / "policy_gradient_experiments" / "results"


def _fe_dir(fe_mode: str) -> str:
    return "dyn" if str(fe_mode) == "switched_dynamics" else "gui"


def _study_slug(study_name: str) -> str:
    text = str(study_name).strip() or "pg_run"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("._-")
    if not safe:
        safe = "pg_run"
    if len(safe) <= 24:
        return safe
    digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:15]}_{digest}"


def suite_root(fe_mode: str, study_name: str) -> Path:
    base = results_root() / _fe_dir(fe_mode)
    candidate = base / str(study_name)
    if len(str(candidate / _PG_LONGEST_SUFFIX)) < _WINDOWS_SAFE_DIR_LIMIT:
        return candidate
    return base / _study_slug(str(study_name))
