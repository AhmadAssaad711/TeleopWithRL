from __future__ import annotations

import os

from ... import config as cfg


DEFAULT_RESULTS_ROOT = "python_env/results/standard_agents_new_env"


def configure_results_root(results_root: str | None = None) -> str:
    root = results_root or DEFAULT_RESULTS_ROOT
    os.environ["TELEOP_RESULTS_ROOT_DIR"] = root
    cfg.RESULTS_ROOT_DIR = root
    return root
