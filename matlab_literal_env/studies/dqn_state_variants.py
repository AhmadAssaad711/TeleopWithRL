from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from gymnasium import spaces

from ... import config as cfg


FeatureExtractor = Callable[[np.ndarray, dict[str, Any]], np.ndarray]


def _fh_scale() -> float:
    return max(float(getattr(cfg, "F_H_SCALE_EST", cfg.FORCE_INPUT_AMP)), 1e-6)


def _fe_scale() -> float:
    return max(float(getattr(cfg, "F_E_MAX_THEORETICAL", cfg.FORCE_INPUT_AMP)), 1e-6)


def _arr(values: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class DQNStateVariant:
    name: str
    feature_names: tuple[str, ...]
    description: str
    extractor: FeatureExtractor

    @property
    def obs_dim(self) -> int:
        return len(self.feature_names)


def _baseline_full10(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    del info
    return _arr(obs)


def _no_mass_flow(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    del info
    return _arr(obs[:8])


def _relative_mechanics(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    del info
    tracking_error = float(obs[1] - obs[0])
    velocity_error = float(obs[3] - obs[2])
    return _arr([
        tracking_error,
        velocity_error,
        float(obs[4]),
        float(obs[5]),
        float(obs[6]),
        float(obs[7]),
        float(obs[8]),
        float(obs[9]),
    ])


def _actuator_pressure_compact2(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    del info
    delta_ps = float(obs[4] - obs[5])
    delta_pm = float(obs[6] - obs[7])
    return _arr([
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        delta_ps,
        delta_pm,
    ])


def _tube_coupling_pressure_compact2(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    del info
    delta_pl1 = float(obs[6] - obs[5])
    delta_pl2 = float(obs[7] - obs[4])
    return _arr([
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        delta_pl1,
        delta_pl2,
    ])


def _force_mechanics_minimal(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    return _arr([
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        float(info.get("F_h", 0.0)) / _fh_scale(),
        float(info.get("F_e", 0.0)) / _fe_scale(),
    ])


def _full10_plus_forces(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    return _arr([
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        float(obs[4]),
        float(obs[5]),
        float(obs[6]),
        float(obs[7]),
        float(obs[8]),
        float(obs[9]),
        float(info.get("F_h", 0.0)) / _fh_scale(),
        float(info.get("F_e", 0.0)) / _fe_scale(),
    ])


def _relative_mechanics_plus_forces(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    tracking_error = float(obs[1] - obs[0])
    velocity_error = float(obs[3] - obs[2])
    delta_ps = float(obs[4] - obs[5])
    delta_pm = float(obs[6] - obs[7])
    return _arr([
        tracking_error,
        velocity_error,
        delta_ps,
        delta_pm,
        float(info.get("F_h", 0.0)) / _fh_scale(),
        float(info.get("F_e", 0.0)) / _fe_scale(),
    ])


def _absolute_posvel_forces(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    x_m = float(info.get("x_m", 0.0)) / float(cfg.OBS_SCALE_POS)
    x_s = float(info.get("x_s", 0.0)) / float(cfg.OBS_SCALE_POS)
    v_s = float(obs[2])
    v_m = float(obs[3])
    f_e = float(info.get("F_e", 0.0)) / _fe_scale()
    f_h = float(info.get("F_h", 0.0)) / _fh_scale()
    return _arr([x_m, x_s, v_m, v_s, f_e, f_h])


def _coupled_pressure_errors_plus_forces(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    pressure_error_ch1 = float(obs[6] - obs[4])  # P_m1 - P_s1
    pressure_error_ch2 = float(obs[7] - obs[5])  # P_m2 - P_s2
    return _arr([
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        pressure_error_ch1,
        pressure_error_ch2,
        float(obs[8]),
        float(obs[9]),
        float(info.get("F_h", 0.0)) / _fh_scale(),
        float(info.get("F_e", 0.0)) / _fe_scale(),
    ])


def build_dqn_state_variants() -> list[DQNStateVariant]:
    return [
        DQNStateVariant(
            name="S0_baseline_full10",
            feature_names=(
                "x_s_eq",
                "x_m_eq",
                "v_s",
                "v_m",
                "P_s1",
                "P_s2",
                "P_m1",
                "P_m2",
                "mdot_L1",
                "mdot_L2",
            ),
            description="Current 10-D replica observation.",
            extractor=_baseline_full10,
        ),
        DQNStateVariant(
            name="S1_no_mass_flow",
            feature_names=(
                "x_s_eq",
                "x_m_eq",
                "v_s",
                "v_m",
                "P_s1",
                "P_s2",
                "P_m1",
                "P_m2",
            ),
            description="Removes tube mass-flow states.",
            extractor=_no_mass_flow,
        ),
        DQNStateVariant(
            name="S2_relative_mechanics",
            feature_names=(
                "tracking_error",
                "velocity_error",
                "P_s1",
                "P_s2",
                "P_m1",
                "P_m2",
                "mdot_L1",
                "mdot_L2",
            ),
            description="Uses relative mechanics with raw pressures and flows.",
            extractor=_relative_mechanics,
        ),
        DQNStateVariant(
            name="S3_actuator_pressure_compact2",
            feature_names=(
                "x_s_eq",
                "x_m_eq",
                "v_s",
                "v_m",
                "delta_P_s",
                "delta_P_m",
            ),
            description="Compresses raw pressures into actuator pressure differences.",
            extractor=_actuator_pressure_compact2,
        ),
        DQNStateVariant(
            name="S4_tube_coupling_pressure_compact2",
            feature_names=(
                "x_s_eq",
                "x_m_eq",
                "v_s",
                "v_m",
                "delta_P_L1",
                "delta_P_L2",
            ),
            description="Compresses raw pressures into tube-coupling pressure differences.",
            extractor=_tube_coupling_pressure_compact2,
        ),
        DQNStateVariant(
            name="S5_force_mechanics_minimal",
            feature_names=("x_s_eq", "x_m_eq", "v_s", "v_m", "F_h", "F_e"),
            description="Uses mechanics plus direct force cues.",
            extractor=_force_mechanics_minimal,
        ),
        DQNStateVariant(
            name="S6_full10_plus_forces",
            feature_names=(
                "x_s_eq",
                "x_m_eq",
                "v_s",
                "v_m",
                "P_s1",
                "P_s2",
                "P_m1",
                "P_m2",
                "mdot_L1",
                "mdot_L2",
                "F_h",
                "F_e",
            ),
            description="Adds human and environment force to the baseline full state.",
            extractor=_full10_plus_forces,
        ),
        DQNStateVariant(
            name="S7_relative_mechanics_plus_forces",
            feature_names=("tracking_error", "velocity_error", "delta_P_s", "delta_P_m", "F_h", "F_e"),
            description="Pairs compact mechanics with direct force cues.",
            extractor=_relative_mechanics_plus_forces,
        ),
        DQNStateVariant(
            name="S8_absolute_posvel_forces",
            feature_names=("x_m", "x_s", "v_m", "v_s", "F_e", "F_h"),
            description="Absolute master/slave position, velocities, and force cues with scale-only normalization.",
            extractor=_absolute_posvel_forces,
        ),
        DQNStateVariant(
            name="S9_coupled_pressure_errors_plus_forces",
            feature_names=(
                "x_s_eq",
                "x_m_eq",
                "v_s",
                "v_m",
                "P_m1_minus_P_s1",
                "P_m2_minus_P_s2",
                "mdot_L1",
                "mdot_L2",
                "F_h",
                "F_e",
            ),
            description="Positions/velocities with cross-piston chamber pressure errors and direct force cues.",
            extractor=_coupled_pressure_errors_plus_forces,
        ),
    ]


_VARIANTS = {variant.name: variant for variant in build_dqn_state_variants()}


def get_dqn_state_variant(name: str) -> DQNStateVariant:
    if name not in _VARIANTS:
        raise KeyError(f"Unknown DQN state variant: {name}")
    return _VARIANTS[name]


class ReplicaStateVariantEnv:
    """Observation wrapper for replica DQN studies without feature clipping."""

    def __init__(self, base_env: Any, state_variant: DQNStateVariant):
        self.base_env = base_env
        self.state_variant = state_variant
        self.action_space = base_env.action_space
        low = np.full(state_variant.obs_dim, -np.inf, dtype=np.float32)
        high = np.full(state_variant.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.obs_dim = state_variant.obs_dim

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    def _transform(self, obs: np.ndarray, info: dict[str, Any] | None) -> np.ndarray:
        return self.state_variant.extractor(np.asarray(obs, dtype=np.float32), info or {})

    def reset(self, *args, **kwargs):
        obs, info = self.base_env.reset(*args, **kwargs)
        return self._transform(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        return self._transform(obs, info), reward, terminated, truncated, info

    def step_voltage(self, u_v: float):
        obs, reward, terminated, truncated, info = self.base_env.step_voltage(u_v)
        return self._transform(obs, info), reward, terminated, truncated, info

    def render(self):
        history = self.base_env.render() or {}
        merged = dict(history)
        merged["state_variant_name"] = self.state_variant.name
        merged["state_variant_features"] = list(self.state_variant.feature_names)
        return merged
