"""Shared command-line normalization helpers for replica experiment runners."""

from __future__ import annotations

import os
from pathlib import Path

from ..config import config as cfg
from ..environment.simuoriginal_replica import FE_MODE_GUI


DEFAULT_RESULTS_ROOT = "matlab_env_python_replica/results/standard_agents_simuoriginal_env"
FE_MODE_DIR_ALIASES = {
    "gui_skin_locked": "gui",
    "switched_dynamics": "dyn",
}


def results_root_for_fe_mode(results_root: str | None = None, fe_mode: str = FE_MODE_GUI) -> str:
    """Return a normalized result root with the FE-mode directory appended.

    Parameters
    ----------
    results_root:
        Base path supplied by a caller. When omitted, use the shared default.
    fe_mode:
        FE mode name or its short directory alias (``gui``/``dyn``).

    Returns
    -------
    str
        A POSIX-style path suitable for configuration and result manifests.
    """
    base = Path(results_root or DEFAULT_RESULTS_ROOT)
    fe_mode = FE_MODE_DIR_ALIASES.get(str(fe_mode), str(fe_mode))
    root = base if base.name == fe_mode else base / fe_mode
    return root.as_posix()


def configure_results_root(results_root: str | None = None, fe_mode: str = FE_MODE_GUI) -> str:
    """Set and return the process-wide result root used by experiment code.

    The value is written both to ``TELEOP_RESULTS_ROOT_DIR`` and to
    ``config.RESULTS_ROOT_DIR`` so legacy runners and new package code agree.
    """
    root = results_root_for_fe_mode(results_root, fe_mode)
    os.environ["TELEOP_RESULTS_ROOT_DIR"] = root
    cfg.RESULTS_ROOT_DIR = root
    return root


def replica_env_kwargs_from_args(args, *, episode_duration: float | None = None, env_switch_time: float | None = None) -> dict:
    """Translate an argparse namespace into replica-environment settings.

    The returned dictionary is accepted by ``SimuOriginalReplicaEnv``. Force
    and FE settings are placed inside its ``reset_options`` mapping; duration,
    termination, stroke, and action settings remain constructor keywords.
    Missing namespace attributes fall back to configuration constants, which
    lets different algorithm launchers share this adapter.
    """
    kwargs = {
        "episode_duration": float(args.episode_duration if getattr(args, "episode_duration", None) is not None else episode_duration or cfg.EPISODE_DURATION),
        "env_switch_time": float(args.env_switch_time if getattr(args, "env_switch_time", None) is not None else env_switch_time or cfg.PAPER_ENV_SWITCH_TIME),
        "terminate_on_error": bool(not getattr(args, "disable_terminate_on_error", False)),
        "legacy_baseline_env": bool(getattr(args, "legacy_baseline_env", False)),
        "enforce_stroke_limit": bool(not getattr(args, "disable_stroke_limit", False)),
        "stroke_limit_mode": str(getattr(args, "stroke_limit_mode", "terminate")),
    }
    reset_position_mode = getattr(args, "reset_position_mode", None)
    if reset_position_mode is not None:
        kwargs["reset_position_mode"] = str(reset_position_mode)
    action_levels = getattr(args, "action_levels", None)
    if action_levels:
        kwargs["action_levels"] = [float(level) for level in action_levels]

    reset_options = {
        "force_amp": float(getattr(args, "force_amp", cfg.FORCE_INPUT_AMP)),
        "force_bias": float(getattr(args, "force_bias", 0.0)),
        "force_phase": float(getattr(args, "force_phase", cfg.FORCE_INPUT_PHASE)),
        "force_waveform": str(getattr(args, "force_waveform", "sine")),
        "fe_mode": str(getattr(args, "fe_mode", FE_MODE_GUI)),
    }
    if getattr(args, "legacy_baseline_env", False):
        reset_options["legacy_baseline_env"] = True
        reset_options["reset_position_mode"] = "zero"
        reset_options["enforce_stroke_limit"] = False
    elif reset_position_mode is not None:
        reset_options["reset_position_mode"] = str(reset_position_mode)
    if getattr(args, "disable_stroke_limit", False):
        reset_options["enforce_stroke_limit"] = False
    stroke_limit_mode = getattr(args, "stroke_limit_mode", None)
    if stroke_limit_mode is not None:
        reset_options["stroke_limit_mode"] = str(stroke_limit_mode)
    force_freq_rad = getattr(args, "force_freq_rad", None)
    if force_freq_rad is not None:
        reset_options["force_freq_rad"] = float(force_freq_rad)
    else:
        reset_options["force_freq"] = float(getattr(args, "force_freq", cfg.FORCE_INPUT_FREQ))

    kwargs["reset_options"] = reset_options
    return kwargs
