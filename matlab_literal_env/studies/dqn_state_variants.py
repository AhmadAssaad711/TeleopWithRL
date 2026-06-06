from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from gymnasium import spaces

from ... import config as cfg


FeatureExtractor = Callable[[np.ndarray, dict[str, Any]], np.ndarray]


def _fh_scale() -> float:
    return max(float(getattr(cfg, "F_H_SCALE_EST", cfg.FORCE_INPUT_AMP)), 1e-6)


def _fe_scale() -> float:
    return max(float(getattr(cfg, "F_E_MAX_THEORETICAL", cfg.FORCE_INPUT_AMP)), 1e-6)


def _action_scale() -> float:
    levels = np.asarray(getattr(cfg, "V_LEVELS", [-5.0, 5.0]), dtype=np.float64).reshape(-1)
    return max(float(np.max(np.abs(levels))) if levels.size else 1.0, 1e-6)


def _valve_position_scale() -> float:
    return max(float(getattr(cfg, "KV", 0.2)) * _action_scale(), 1.0)


def _valve_velocity_scale() -> float:
    return max(150.0 * _valve_position_scale(), 1.0)


def _safe_scale(value: float) -> float:
    return max(abs(float(value)), 1e-6)


def _info_float(info: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = info.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _arr(values: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class StateFeatureSpec:
    name: str
    description: str
    scale_note: str
    extractor: FeatureExtractor


@dataclass(frozen=True)
class DQNStateVariant:
    name: str
    feature_names: tuple[str, ...]
    description: str
    extractor: FeatureExtractor
    metadata: dict[str, Any] | None = None

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


def _absolute_posvel_only(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    x_m = float(info.get("x_m", 0.0)) / float(cfg.OBS_SCALE_POS)
    x_s = float(info.get("x_s", 0.0)) / float(cfg.OBS_SCALE_POS)
    v_s = float(obs[2])
    v_m = float(obs[3])
    return _arr([x_m, x_s, v_m, v_s])


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


def _custom_tracking_error(obs: np.ndarray, info: dict[str, Any]) -> float:
    del info
    return float(obs[1] - obs[0])


def _custom_velocity_error(obs: np.ndarray, info: dict[str, Any]) -> float:
    del info
    return float(obs[3] - obs[2])


def _custom_force_diff(obs: np.ndarray, info: dict[str, Any]) -> float:
    del obs
    return (
        _info_float(info, "F_e") - _info_float(info, "F_h")
    ) / _safe_scale(25.0)


def _custom_time_fraction(obs: np.ndarray, info: dict[str, Any]) -> float:
    del obs
    duration = _safe_scale(_info_float(info, "episode_duration", float(cfg.EPISODE_DURATION)))
    return _info_float(info, "time") / duration


_CUSTOM_STATE_FEATURES: dict[str, StateFeatureSpec] = {
    "x_s_eq": StateFeatureSpec(
        "x_s_eq",
        "Slave position relative to the reset equilibrium.",
        "x_s_centered / OBS_SCALE_POS",
        lambda obs, info: float(obs[0]),
    ),
    "x_m_eq": StateFeatureSpec(
        "x_m_eq",
        "Master position relative to the reset equilibrium.",
        "x_m_centered / OBS_SCALE_POS",
        lambda obs, info: float(obs[1]),
    ),
    "v_s": StateFeatureSpec(
        "v_s",
        "Slave velocity.",
        "v_s / OBS_SCALE_VEL",
        lambda obs, info: float(obs[2]),
    ),
    "v_m": StateFeatureSpec(
        "v_m",
        "Master velocity.",
        "v_m / OBS_SCALE_VEL",
        lambda obs, info: float(obs[3]),
    ),
    "P_s1": StateFeatureSpec(
        "P_s1",
        "Slave chamber 1 pressure.",
        "P_s1 / OBS_SCALE_PRESSURE",
        lambda obs, info: float(obs[4]),
    ),
    "P_s2": StateFeatureSpec(
        "P_s2",
        "Slave chamber 2 pressure.",
        "P_s2 / OBS_SCALE_PRESSURE",
        lambda obs, info: float(obs[5]),
    ),
    "P_m1": StateFeatureSpec(
        "P_m1",
        "Master chamber 1 pressure.",
        "P_m1 / OBS_SCALE_PRESSURE",
        lambda obs, info: float(obs[6]),
    ),
    "P_m2": StateFeatureSpec(
        "P_m2",
        "Master chamber 2 pressure.",
        "P_m2 / OBS_SCALE_PRESSURE",
        lambda obs, info: float(obs[7]),
    ),
    "mdot_L1": StateFeatureSpec(
        "mdot_L1",
        "Line 1 mass-flow state.",
        "mdot_L1 / OBS_SCALE_FLOW",
        lambda obs, info: float(obs[8]),
    ),
    "mdot_L2": StateFeatureSpec(
        "mdot_L2",
        "Line 2 mass-flow state.",
        "mdot_L2 / OBS_SCALE_FLOW",
        lambda obs, info: float(obs[9]),
    ),
    "x_s": StateFeatureSpec(
        "x_s",
        "Absolute slave position.",
        "x_s / OBS_SCALE_POS",
        lambda obs, info: _info_float(info, "x_s") / _safe_scale(cfg.OBS_SCALE_POS),
    ),
    "x_m": StateFeatureSpec(
        "x_m",
        "Absolute master position.",
        "x_m / OBS_SCALE_POS",
        lambda obs, info: _info_float(info, "x_m") / _safe_scale(cfg.OBS_SCALE_POS),
    ),
    "x_s_centered": StateFeatureSpec(
        "x_s_centered",
        "Slave position relative to the reset equilibrium.",
        "x_s_centered / OBS_SCALE_POS",
        lambda obs, info: _info_float(info, "x_s_centered") / _safe_scale(cfg.OBS_SCALE_POS),
    ),
    "x_m_centered": StateFeatureSpec(
        "x_m_centered",
        "Master position relative to the reset equilibrium.",
        "x_m_centered / OBS_SCALE_POS",
        lambda obs, info: _info_float(info, "x_m_centered") / _safe_scale(cfg.OBS_SCALE_POS),
    ),
    "tracking_error": StateFeatureSpec(
        "tracking_error",
        "Master minus slave position.",
        "(x_m - x_s) / OBS_SCALE_POS",
        _custom_tracking_error,
    ),
    "pos_error": StateFeatureSpec(
        "pos_error",
        "Alias for tracking_error.",
        "(x_m - x_s) / OBS_SCALE_POS",
        _custom_tracking_error,
    ),
    "velocity_error": StateFeatureSpec(
        "velocity_error",
        "Master minus slave velocity.",
        "(v_m - v_s) / OBS_SCALE_VEL",
        _custom_velocity_error,
    ),
    "F_h": StateFeatureSpec(
        "F_h",
        "Human/master force input.",
        "F_h / F_H_SCALE_EST",
        lambda obs, info: _info_float(info, "F_h") / _fh_scale(),
    ),
    "F_e": StateFeatureSpec(
        "F_e",
        "Environment/slave interaction force.",
        "F_e / F_E_MAX_THEORETICAL",
        lambda obs, info: _info_float(info, "F_e") / _fe_scale(),
    ),
    "force_diff": StateFeatureSpec(
        "force_diff",
        "Environment force minus human force.",
        "(F_e - F_h) / 25 N",
        _custom_force_diff,
    ),
    "transparency_error": StateFeatureSpec(
        "transparency_error",
        "Power mismatch used by the transparency metric.",
        "transparency_error / MAX_POWER_ERROR",
        lambda obs, info: _info_float(info, "transparency_error") / _safe_scale(cfg.MAX_POWER_ERROR),
    ),
    "u_v": StateFeatureSpec(
        "u_v",
        "Previously applied continuous control voltage.",
        "u_v / max(abs(V_LEVELS))",
        lambda obs, info: _info_float(info, "u_v") / _action_scale(),
    ),
    "requested_u_v": StateFeatureSpec(
        "requested_u_v",
        "Voltage requested before edge damping or clipping.",
        "requested_u_v / max(abs(V_LEVELS))",
        lambda obs, info: _info_float(info, "requested_u_v") / _action_scale(),
    ),
    "x_v": StateFeatureSpec(
        "x_v",
        "Valve spool position state.",
        "x_v / valve_position_scale",
        lambda obs, info: _info_float(info, "x_v") / _valve_position_scale(),
    ),
    "x_v_dot": StateFeatureSpec(
        "x_v_dot",
        "Valve spool velocity state.",
        "x_v_dot / valve_velocity_scale",
        lambda obs, info: _info_float(info, "x_v_dot") / _valve_velocity_scale(),
    ),
    "delta_P_s": StateFeatureSpec(
        "delta_P_s",
        "Slave actuator pressure difference.",
        "(P_s1 - P_s2) / OBS_SCALE_PRESSURE",
        lambda obs, info: float(obs[4] - obs[5]),
    ),
    "delta_P_m": StateFeatureSpec(
        "delta_P_m",
        "Master actuator pressure difference.",
        "(P_m1 - P_m2) / OBS_SCALE_PRESSURE",
        lambda obs, info: float(obs[6] - obs[7]),
    ),
    "P_m1_minus_P_s1": StateFeatureSpec(
        "P_m1_minus_P_s1",
        "Cross-piston chamber 1 pressure error.",
        "(P_m1 - P_s1) / OBS_SCALE_PRESSURE",
        lambda obs, info: float(obs[6] - obs[4]),
    ),
    "P_m2_minus_P_s2": StateFeatureSpec(
        "P_m2_minus_P_s2",
        "Cross-piston chamber 2 pressure error.",
        "(P_m2 - P_s2) / OBS_SCALE_PRESSURE",
        lambda obs, info: float(obs[7] - obs[5]),
    ),
    "time_fraction": StateFeatureSpec(
        "time_fraction",
        "Episode progress; useful when the environment switches at a fixed time.",
        "time / episode_duration",
        _custom_time_fraction,
    ),
    "env_id": StateFeatureSpec(
        "env_id",
        "Current environment label encoded as skin=0, fat=1.",
        "0 or 1",
        lambda obs, info: _info_float(info, "env_id"),
    ),
}


def available_custom_state_feature_rows() -> list[dict[str, str]]:
    return [
        {
            "feature": spec.name,
            "description": spec.description,
            "scale": spec.scale_note,
        }
        for spec in _CUSTOM_STATE_FEATURES.values()
    ]


def available_custom_state_features() -> tuple[str, ...]:
    return tuple(_CUSTOM_STATE_FEATURES.keys())


def build_custom_dqn_state_variant(
    *,
    name: str,
    feature_names: list[str] | tuple[str, ...],
    description: str = "Notebook-defined custom state.",
    metadata: dict[str, Any] | None = None,
) -> DQNStateVariant:
    selected = tuple(str(feature).strip() for feature in feature_names if str(feature).strip())
    if not selected:
        raise ValueError("A custom state variant needs at least one selected feature.")

    unknown = [feature for feature in selected if feature not in _CUSTOM_STATE_FEATURES]
    if unknown:
        known = ", ".join(available_custom_state_features())
        raise KeyError(f"Unknown custom state feature(s): {unknown}. Known features: {known}")

    def _extractor(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        return _arr([
            float(_CUSTOM_STATE_FEATURES[feature].extractor(obs, info))
            for feature in selected
        ])

    return DQNStateVariant(
        name=str(name),
        feature_names=selected,
        description=str(description),
        extractor=_extractor,
        metadata=dict(metadata or {}),
    )


def _features_from_spec(spec: Mapping[str, Any]) -> list[str]:
    if "selected_features" in spec:
        features = spec["selected_features"]
    elif "features" in spec:
        features = spec["features"]
    else:
        raise KeyError("Custom state spec must contain 'selected_features' or 'features'.")

    if isinstance(features, Mapping):
        return [str(name) for name, enabled in features.items() if bool(enabled)]
    return [str(name) for name in features]


def build_custom_dqn_state_variant_from_spec(spec: Mapping[str, Any]) -> DQNStateVariant:
    selected = _features_from_spec(spec)
    name = str(spec.get("name") or "custom_state")
    description = str(spec.get("description") or "Notebook-defined custom state.")
    metadata = {
        "kind": "custom_state_spec",
        "selected_features": selected,
        "source_spec": dict(spec),
    }
    return build_custom_dqn_state_variant(
        name=name,
        feature_names=selected,
        description=description,
        metadata=metadata,
    )


def load_custom_dqn_state_variant(path: str | Path) -> DQNStateVariant:
    with open(Path(path), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    if not isinstance(spec, Mapping):
        raise TypeError("Custom state spec JSON must contain an object.")
    return build_custom_dqn_state_variant_from_spec(spec)


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
        DQNStateVariant(
            name="S11_absolute_posvel_only",
            feature_names=("x_m", "x_s", "v_m", "v_s"),
            description="Absolute master/slave positions and velocities only; force cues removed.",
            extractor=_absolute_posvel_only,
        ),
    ]


_VARIANTS = {variant.name: variant for variant in build_dqn_state_variants()}


def get_dqn_state_variant(name: str) -> DQNStateVariant:
    candidate = Path(str(name))
    if candidate.suffix.lower() == ".json" and candidate.exists():
        return load_custom_dqn_state_variant(candidate)
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
