"""State-variant definitions and wrappers for DQN ablation studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from gymnasium import spaces

try:
    from . import config as cfg
except ImportError:  # pragma: no cover - direct script execution
    import config as cfg


FeatureExtractor = Callable[[np.ndarray, dict[str, Any]], np.ndarray]


def _clip_obs(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.clip(arr, -2.0, 2.0).astype(np.float32, copy=False)


def _force_scales() -> tuple[float, float]:
    fh_scale = max(float(getattr(cfg, "F_H_SCALE_EST", cfg.FORCE_INPUT_AMP)), 1e-6)
    fe_scale = max(float(getattr(cfg, "F_E_MAX_THEORETICAL", cfg.FORCE_INPUT_AMP)), 1e-6)
    return fh_scale, fe_scale


@dataclass(frozen=True)
class StateVariant:
    name: str
    feature_names: tuple[str, ...]
    description: str
    extractor: FeatureExtractor

    @property
    def obs_dim(self) -> int:
        return len(self.feature_names)


def _baseline_full10(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    del info
    return _clip_obs(obs)


def _no_mass_flow(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    del info
    return _clip_obs(obs[:8])


def _relative_mechanics(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    del info
    tracking_error = float(obs[1] - obs[0])
    velocity_error = float(obs[3] - obs[2])
    return _clip_obs([
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
    return _clip_obs([
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        delta_ps,
        delta_pm,
    ])


def _tube_coupling_pressure_compact2(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    del info
    delta_pl1 = float(obs[6] - obs[5])  # P_m1 - P_s2
    delta_pl2 = float(obs[7] - obs[4])  # P_m2 - P_s1
    return _clip_obs([
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        delta_pl1,
        delta_pl2,
    ])


def _force_mechanics_minimal(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    fh_scale, fe_scale = _force_scales()
    fh = float(info.get("F_h", 0.0)) / fh_scale
    fe = float(info.get("F_e", 0.0)) / fe_scale
    return _clip_obs([
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        fh,
        fe,
    ])


def build_state_variants() -> list[StateVariant]:
    return [
        StateVariant(
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
            description="Current 10-D baseline observation.",
            extractor=_baseline_full10,
        ),
        StateVariant(
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
            description="Removes tube mass-flow states to test whether chamber pressures are enough.",
            extractor=_no_mass_flow,
        ),
        StateVariant(
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
            description="Replaces absolute mechanics with relative tracking and velocity errors.",
            extractor=_relative_mechanics,
        ),
        StateVariant(
            name="S3_actuator_pressure_compact2",
            feature_names=(
                "x_s_eq",
                "x_m_eq",
                "v_s",
                "v_m",
                "delta_P_s",
                "delta_P_m",
            ),
            description="Compresses raw pressures into actuator-driving pressure differences.",
            extractor=_actuator_pressure_compact2,
        ),
        StateVariant(
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
        StateVariant(
            name="S5_force_mechanics_minimal",
            feature_names=(
                "x_s_eq",
                "x_m_eq",
                "v_s",
                "v_m",
                "F_h",
                "F_e",
            ),
            description="Uses only mechanics plus human/environment forces as a compact redesign.",
            extractor=_force_mechanics_minimal,
        ),
    ]


_STATE_VARIANTS = {variant.name: variant for variant in build_state_variants()}


def get_state_variant(name: str) -> StateVariant:
    if name not in _STATE_VARIANTS:
        raise KeyError(f"Unknown state variant: {name}")
    return _STATE_VARIANTS[name]


class StateVariantEnv:
    """Observation wrapper that projects the base env state into a named variant."""

    def __init__(self, base_env: Any, state_variant: StateVariant):
        self.base_env = base_env
        self.state_variant = state_variant
        self.action_space = base_env.action_space
        low = np.full(state_variant.obs_dim, -2.0, dtype=np.float32)
        high = np.full(state_variant.obs_dim, 2.0, dtype=np.float32)
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


def unwrap_base_env(env: Any) -> Any:
    current = env
    while hasattr(current, "base_env"):
        current = current.base_env
    return current
