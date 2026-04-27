from __future__ import annotations

import os
from pathlib import Path

from ... import config as cfg
from ..simuoriginal_replica import FE_MODE_GUI


DEFAULT_RESULTS_ROOT = "matlab_literal_env/results/standard_agents_simuoriginal_env"
FE_MODE_DIR_ALIASES = {
    "gui_skin_locked": "gui",
    "switched_dynamics": "dyn",
}


def results_root_for_fe_mode(results_root: str | None = None, fe_mode: str = FE_MODE_GUI) -> str:
    base = Path(results_root or DEFAULT_RESULTS_ROOT)
    fe_mode = FE_MODE_DIR_ALIASES.get(str(fe_mode), str(fe_mode))
    root = base if base.name == fe_mode else base / fe_mode
    return root.as_posix()


def configure_results_root(results_root: str | None = None, fe_mode: str = FE_MODE_GUI) -> str:
    root = results_root_for_fe_mode(results_root, fe_mode)
    os.environ["TELEOP_RESULTS_ROOT_DIR"] = root
    cfg.RESULTS_ROOT_DIR = root
    return root


def replica_env_kwargs_from_args(args, *, episode_duration: float | None = None, env_switch_time: float | None = None) -> dict:
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
