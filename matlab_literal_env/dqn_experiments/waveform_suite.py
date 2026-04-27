from __future__ import annotations

from typing import Any
import numpy as np


DEFAULT_WAVEFORMS = ("sine", "cosine", "square", "ramp", "multisine")
WAVEFORM_DIR_ALIASES = {
    "sine": "sin",
    "cosine": "cos",
    "square": "sqr",
    "ramp": "rmp",
    "multisine": "mul",
}


def parse_waveform_forms(raw: str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_WAVEFORMS)
    items = [item.strip().lower() for item in str(raw).split(",")]
    forms = [item for item in items if item]
    if not forms:
        raise ValueError("Waveform list must not be empty.")
    allowed = set(DEFAULT_WAVEFORMS)
    unknown = [item for item in forms if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown waveform(s): {unknown}")
    return forms


def parse_waveform_stages(raw: str | None) -> list[list[str]]:
    if raw is None:
        raw = "sine,cosine;multisine;square,ramp"
    stages: list[list[str]] = []
    for chunk in str(raw).split(";"):
        forms = parse_waveform_forms(chunk)
        if forms:
            stages.append(forms)
    if not stages:
        raise ValueError("Waveform stages must not be empty.")
    return stages


def suite_reset_options(
    *,
    waveforms: list[str],
    force_amp: float,
    force_bias: float,
    force_freq_rad: float,
    force_phase: float,
) -> list[dict[str, Any]]:
    return [
        {
            "name": waveform,
            "dir_name": WAVEFORM_DIR_ALIASES.get(str(waveform), str(waveform)),
            "reset_options": {
                "force_amp": float(force_amp),
                "force_bias": float(force_bias),
                "force_freq_rad": float(force_freq_rad),
                "force_phase": float(force_phase),
                "force_waveform": str(waveform),
            },
        }
        for waveform in waveforms
    ]


def curriculum_schedule(
    *,
    total_episodes: int,
    stages: list[list[str]],
    force_amp: float,
    force_bias: float,
    force_freq_rad: float,
    force_phase: float,
    rng_seed: int,
) -> list[dict[str, Any]]:
    total_episodes = max(1, int(total_episodes))
    stages = list(stages)
    if not stages:
        raise ValueError("Curriculum stages must not be empty.")
    rng = np.random.default_rng(int(rng_seed))
    schedule: list[dict[str, Any]] = []
    stage_count = len(stages)
    base = total_episodes // stage_count
    remainder = total_episodes % stage_count
    for idx, stage_forms in enumerate(stages):
        count = base + (1 if idx < remainder else 0)
        options = suite_reset_options(
            waveforms=stage_forms,
            force_amp=float(force_amp),
            force_bias=float(force_bias),
            force_freq_rad=float(force_freq_rad),
            force_phase=float(force_phase),
        )
        for _ in range(count):
            choice = options[int(rng.integers(0, len(options)))]
            schedule.append(dict(choice["reset_options"]))
    if len(schedule) < total_episodes:
        pad_options = suite_reset_options(
            waveforms=stages[-1],
            force_amp=float(force_amp),
            force_bias=float(force_bias),
            force_freq_rad=float(force_freq_rad),
            force_phase=float(force_phase),
        )
        while len(schedule) < total_episodes:
            choice = pad_options[int(rng.integers(0, len(pad_options)))]
            schedule.append(dict(choice["reset_options"]))
    return schedule[:total_episodes]
