from __future__ import annotations

import os

from ... import config as cfg


DEFAULT_RESULTS_ROOT = "matlab_literal_env/results/standard_agents_simuoriginal_env"


def configure_results_root(results_root: str | None = None) -> str:
    root = results_root or DEFAULT_RESULTS_ROOT
    os.environ["TELEOP_RESULTS_ROOT_DIR"] = root
    cfg.RESULTS_ROOT_DIR = root
    return root


def replica_env_kwargs_from_args(args, *, episode_duration: float | None = None, env_switch_time: float | None = None) -> dict:
    kwargs = {
        "episode_duration": float(args.episode_duration if getattr(args, "episode_duration", None) is not None else episode_duration or cfg.EPISODE_DURATION),
        "env_switch_time": float(args.env_switch_time if getattr(args, "env_switch_time", None) is not None else env_switch_time or cfg.ENV_SWITCH_TIME),
        "terminate_on_error": bool(not getattr(args, "disable_terminate_on_error", False)),
    }

    reset_options = {
        "force_amp": float(getattr(args, "force_amp", cfg.FORCE_INPUT_AMP)),
        "force_bias": float(getattr(args, "force_bias", 0.0)),
        "force_phase": float(getattr(args, "force_phase", cfg.FORCE_INPUT_PHASE)),
        "force_waveform": str(getattr(args, "force_waveform", "sine")),
    }
    force_freq_rad = getattr(args, "force_freq_rad", None)
    if force_freq_rad is not None:
        reset_options["force_freq_rad"] = float(force_freq_rad)
    else:
        reset_options["force_freq"] = float(getattr(args, "force_freq", cfg.FORCE_INPUT_FREQ))

    kwargs["reset_options"] = reset_options
    return kwargs
