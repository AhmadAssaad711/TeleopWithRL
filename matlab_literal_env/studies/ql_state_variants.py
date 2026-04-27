from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ... import config as cfg


Discretizer = Callable[[np.ndarray, dict[str, Any]], tuple[int, ...]]


FORCE_BIN_EDGES = np.array([-0.6, -0.2, 0.2, 0.6], dtype=np.float64)
FINE_TRACKING_ERROR_BINS = np.array([
    -0.06, -0.045, -0.032, -0.022, -0.014, -0.008, -0.004, -0.001,
     0.001,  0.004,  0.008,  0.014,  0.022,  0.032,  0.045,  0.06
], dtype=np.float64)
FINE_VELOCITY_ERROR_BINS = np.array([
    -0.40, -0.28, -0.18, -0.10, -0.05, -0.02,
     0.02,  0.05,  0.10,  0.18,  0.28,  0.40
], dtype=np.float64)
FINE_PRESSURE_DIFF_BINS = np.array([
    -200_000, -140_000, -90_000, -45_000, -18_000, -6_000,
       6_000,   18_000,  45_000,  90_000, 140_000, 200_000
], dtype=np.float64)


def _fh_scale() -> float:
    return max(float(getattr(cfg, "F_H_SCALE_EST", cfg.FORCE_INPUT_AMP)), 1e-6)


def _fe_scale() -> float:
    return max(float(getattr(cfg, "F_E_MAX_THEORETICAL", cfg.FORCE_INPUT_AMP)), 1e-6)


def _delta_force_scale() -> float:
    return max(_fh_scale() + _fe_scale(), 1e-6)


def _tracking_error(obs: np.ndarray) -> float:
    slave_pos = float(obs[0]) * float(cfg.OBS_SCALE_POS)
    master_pos = float(obs[1]) * float(cfg.OBS_SCALE_POS)
    return master_pos - slave_pos


def _slave_centered_position(obs: np.ndarray) -> float:
    return float(obs[0]) * float(cfg.OBS_SCALE_POS)


def _master_centered_position(obs: np.ndarray) -> float:
    return float(obs[1]) * float(cfg.OBS_SCALE_POS)


def _slave_velocity(obs: np.ndarray) -> float:
    return float(obs[2]) * float(cfg.OBS_SCALE_VEL)


def _master_velocity(obs: np.ndarray) -> float:
    return float(obs[3]) * float(cfg.OBS_SCALE_VEL)


def _velocity_error(obs: np.ndarray) -> float:
    v_s = float(obs[2]) * float(cfg.OBS_SCALE_VEL)
    v_m = float(obs[3]) * float(cfg.OBS_SCALE_VEL)
    return v_m - v_s


def _slave_pressure_diff(obs: np.ndarray) -> float:
    P_s1 = float(obs[4]) * float(cfg.OBS_SCALE_PRESSURE)
    P_s2 = float(obs[5]) * float(cfg.OBS_SCALE_PRESSURE)
    return P_s1 - P_s2


def _master_pressure_diff(obs: np.ndarray) -> float:
    P_m1 = float(obs[6]) * float(cfg.OBS_SCALE_PRESSURE)
    P_m2 = float(obs[7]) * float(cfg.OBS_SCALE_PRESSURE)
    return P_m1 - P_m2


def _digitize_force(raw_force: float, scale: float) -> int:
    normalized = float(raw_force) / max(float(scale), 1e-6)
    return int(np.digitize(normalized, FORCE_BIN_EDGES))


def _baseline_reduced4(obs: np.ndarray, info: dict[str, Any]) -> tuple[int, ...]:
    del info
    return (
        int(np.digitize(_tracking_error(obs), cfg.REDUCED_TRACKING_ERROR_BINS)),
        int(np.digitize(_velocity_error(obs), cfg.REDUCED_VELOCITY_ERROR_BINS)),
        int(np.digitize(_slave_pressure_diff(obs), cfg.REDUCED_SLAVE_PRESSURE_DIFF_BINS)),
        int(np.digitize(_master_pressure_diff(obs), cfg.REDUCED_MASTER_PRESSURE_DIFF_BINS)),
    )


def _baseline_finer4(obs: np.ndarray, info: dict[str, Any]) -> tuple[int, ...]:
    del info
    return (
        int(np.digitize(_tracking_error(obs), FINE_TRACKING_ERROR_BINS)),
        int(np.digitize(_velocity_error(obs), FINE_VELOCITY_ERROR_BINS)),
        int(np.digitize(_slave_pressure_diff(obs), FINE_PRESSURE_DIFF_BINS)),
        int(np.digitize(_master_pressure_diff(obs), FINE_PRESSURE_DIFF_BINS)),
    )


def _forceenv_reduced5(obs: np.ndarray, info: dict[str, Any]) -> tuple[int, ...]:
    return (
        int(np.digitize(_tracking_error(obs), cfg.REDUCED_TRACKING_ERROR_BINS)),
        int(np.digitize(_velocity_error(obs), cfg.REDUCED_VELOCITY_ERROR_BINS)),
        int(np.digitize(_slave_pressure_diff(obs), cfg.REDUCED_SLAVE_PRESSURE_DIFF_BINS)),
        int(np.digitize(_master_pressure_diff(obs), cfg.REDUCED_MASTER_PRESSURE_DIFF_BINS)),
        _digitize_force(float(info.get("F_e", 0.0)), _fe_scale()),
    )


def _operatorforce_reduced5(obs: np.ndarray, info: dict[str, Any]) -> tuple[int, ...]:
    return (
        int(np.digitize(_tracking_error(obs), cfg.REDUCED_TRACKING_ERROR_BINS)),
        int(np.digitize(_velocity_error(obs), cfg.REDUCED_VELOCITY_ERROR_BINS)),
        int(np.digitize(_slave_pressure_diff(obs), cfg.REDUCED_SLAVE_PRESSURE_DIFF_BINS)),
        int(np.digitize(_master_pressure_diff(obs), cfg.REDUCED_MASTER_PRESSURE_DIFF_BINS)),
        _digitize_force(float(info.get("F_h", 0.0)), _fh_scale()),
    )


def _forcepair_reduced5(obs: np.ndarray, info: dict[str, Any]) -> tuple[int, ...]:
    return (
        int(np.digitize(_tracking_error(obs), cfg.REDUCED_TRACKING_ERROR_BINS)),
        int(np.digitize(_velocity_error(obs), cfg.REDUCED_VELOCITY_ERROR_BINS)),
        int(np.digitize(_slave_pressure_diff(obs), cfg.REDUCED_SLAVE_PRESSURE_DIFF_BINS)),
        _digitize_force(float(info.get("F_h", 0.0)), _fh_scale()),
        _digitize_force(float(info.get("F_e", 0.0)), _fe_scale()),
    )


def _forcecompact_reduced5(obs: np.ndarray, info: dict[str, Any]) -> tuple[int, ...]:
    F_h = float(info.get("F_h", 0.0))
    F_e = float(info.get("F_e", 0.0))
    return (
        int(np.digitize(_tracking_error(obs), cfg.REDUCED_TRACKING_ERROR_BINS)),
        int(np.digitize(_velocity_error(obs), cfg.REDUCED_VELOCITY_ERROR_BINS)),
        int(np.digitize(_slave_pressure_diff(obs), cfg.REDUCED_SLAVE_PRESSURE_DIFF_BINS)),
        _digitize_force(F_h - F_e, _delta_force_scale()),
        _digitize_force(F_e, _fe_scale()),
    )


def _relative_positions_velocities_forces_reduced6(obs: np.ndarray, info: dict[str, Any]) -> tuple[int, ...]:
    return (
        int(np.digitize(_slave_centered_position(obs), cfg.SLAVE_POS_ERROR_BINS)),
        int(np.digitize(_master_centered_position(obs), cfg.MASTER_POS_ERROR_BINS)),
        int(np.digitize(_slave_velocity(obs), cfg.VEL_ERROR_BINS)),
        int(np.digitize(_master_velocity(obs), cfg.VEL_ERROR_BINS)),
        _digitize_force(float(info.get("F_h", 0.0)), _fh_scale()),
        _digitize_force(float(info.get("F_e", 0.0)), _fe_scale()),
    )


@dataclass(frozen=True)
class QLStateVariant:
    name: str
    feature_names: tuple[str, ...]
    description: str
    discretizer: Discretizer
    state_dims: tuple[int, ...]


def build_ql_state_variants() -> list[QLStateVariant]:
    reduced_track = len(cfg.REDUCED_TRACKING_ERROR_BINS) + 1
    reduced_vel = len(cfg.REDUCED_VELOCITY_ERROR_BINS) + 1
    reduced_slave_p = len(cfg.REDUCED_SLAVE_PRESSURE_DIFF_BINS) + 1
    reduced_master_p = len(cfg.REDUCED_MASTER_PRESSURE_DIFF_BINS) + 1
    fine_track = len(FINE_TRACKING_ERROR_BINS) + 1
    fine_vel = len(FINE_VELOCITY_ERROR_BINS) + 1
    fine_pressure = len(FINE_PRESSURE_DIFF_BINS) + 1
    centered_slave_pos = len(cfg.SLAVE_POS_ERROR_BINS) + 1
    centered_master_pos = len(cfg.MASTER_POS_ERROR_BINS) + 1
    velocity_bins = len(cfg.VEL_ERROR_BINS) + 1
    force_bins = len(FORCE_BIN_EDGES) + 1

    return [
        QLStateVariant(
            name="Q0_baseline_reduced4",
            feature_names=("tracking_error", "velocity_error", "delta_P_s", "delta_P_m"),
            description="Current 4-D reduced tabular state.",
            discretizer=_baseline_reduced4,
            state_dims=(reduced_track, reduced_vel, reduced_slave_p, reduced_master_p),
        ),
        QLStateVariant(
            name="Q0f_baseline_finer4",
            feature_names=("tracking_error", "velocity_error", "delta_P_s", "delta_P_m"),
            description="Slightly finer 4-D baseline state with denser tracking/velocity/pressure bins.",
            discretizer=_baseline_finer4,
            state_dims=(fine_track, fine_vel, fine_pressure, fine_pressure),
        ),
        QLStateVariant(
            name="Q1_forceenv_reduced5",
            feature_names=("tracking_error", "velocity_error", "delta_P_s", "delta_P_m", "F_e"),
            description="Adds environment force to the current reduced state.",
            discretizer=_forceenv_reduced5,
            state_dims=(reduced_track, reduced_vel, reduced_slave_p, reduced_master_p, force_bins),
        ),
        QLStateVariant(
            name="Q2_operatorforce_reduced5",
            feature_names=("tracking_error", "velocity_error", "delta_P_s", "delta_P_m", "F_h"),
            description="Adds operator-applied force to the current reduced state.",
            discretizer=_operatorforce_reduced5,
            state_dims=(reduced_track, reduced_vel, reduced_slave_p, reduced_master_p, force_bins),
        ),
        QLStateVariant(
            name="Q3_forcepair_reduced5",
            feature_names=("tracking_error", "velocity_error", "delta_P_s", "F_h", "F_e"),
            description="Uses direct human/environment force pair cues.",
            discretizer=_forcepair_reduced5,
            state_dims=(reduced_track, reduced_vel, reduced_slave_p, force_bins, force_bins),
        ),
        QLStateVariant(
            name="Q4_forcecompact_reduced5",
            feature_names=("tracking_error", "velocity_error", "delta_P_s", "delta_F", "F_e"),
            description="Uses force mismatch and environment force as compact contact cues.",
            discretizer=_forcecompact_reduced5,
            state_dims=(reduced_track, reduced_vel, reduced_slave_p, force_bins, force_bins),
        ),
        QLStateVariant(
            name="Q5_relative_posvel_forces_reduced6",
            feature_names=("x_s_centered", "x_m_centered", "v_s", "v_m", "F_h", "F_e"),
            description="Centered slave/master positions, individual velocities, and force pair cues.",
            discretizer=_relative_positions_velocities_forces_reduced6,
            state_dims=(centered_slave_pos, centered_master_pos, velocity_bins, velocity_bins, force_bins, force_bins),
        ),
    ]


_VARIANTS = {variant.name: variant for variant in build_ql_state_variants()}


def get_ql_state_variant(name: str) -> QLStateVariant:
    if name not in _VARIANTS:
        raise KeyError(f"Unknown Q-learning state variant: {name}")
    return _VARIANTS[name]
