from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ... import config as cfg

_ACTION_LEVELS = np.asarray(getattr(cfg, "V_LEVELS", [-5.0, 5.0]), dtype=np.float64).reshape(-1)
DEFAULT_ACTION_SCALE_V = float(np.max(np.abs(_ACTION_LEVELS))) if _ACTION_LEVELS.size else 1.0
DEFAULT_ACTION_DELTA_SCALE_V = float(max(1.0, 2.0 * DEFAULT_ACTION_SCALE_V))
DEFAULT_TRACKING_SCALE_M = float(cfg.L_CYL)
DEFAULT_VELOCITY_ERROR_SCALE_MPS = 3.0
DEFAULT_FORCE_DIFF_SCALE_N = 25.0
DEFAULT_TRANSPARENCY_SCALE_W = 60.0


@dataclass(frozen=True)
class RewardVariant:
    name: str
    tracking_weight: float
    transparency_weight: float
    jerk_weight: float
    use_jerk: bool
    tracking_scale_m: float = float(cfg.MAX_POSITION_ERROR)
    transparency_scale_w: float = float(cfg.MAX_POWER_ERROR)
    velocity_weight: float = 0.0
    velocity_scale_mps: float = 1.0
    force_diff_weight: float = 0.0
    force_diff_scale_n: float = 1.0
    effort_weight: float = cfg.GAMMA_EFFORT
    effort_scale_v: float = 1.0
    jerk_scale_v: float = 1.0
    stroke_limit_penalty: float = 250.0
    invalid_state_penalty: float = 100.0
    tracking_error_fail_penalty: float = 1000.0
    edge_buffer_m: float = 0.0
    edge_penalty_weight: float = 0.0
    low_force_threshold_n: float = 0.0
    low_force_edge_penalty_weight: float = 0.0


def baseline_reward_variant() -> RewardVariant:
    return RewardVariant(
        name="baseline_cfg",
        tracking_weight=float(cfg.ALPHA_TRACKING),
        transparency_weight=float(cfg.BETA_TRANSPARENCY),
        jerk_weight=0.0,
        use_jerk=False,
        effort_weight=float(cfg.GAMMA_EFFORT),
        tracking_error_fail_penalty=1000.0,
    )


def equal_gradient_reward_variant() -> RewardVariant:
    baseline = baseline_reward_variant()
    return replace(
        baseline,
        name="eqgrad_t40_tr40_nojerk",
        transparency_weight=float(baseline.tracking_weight),
    )


def normalized_force_shape_reward_variant() -> RewardVariant:
    return RewardVariant(
        name="norm_force_shape_v1",
        tracking_weight=1.0,
        transparency_weight=0.0,
        jerk_weight=0.05,
        use_jerk=True,
        tracking_scale_m=DEFAULT_TRACKING_SCALE_M,
        transparency_scale_w=DEFAULT_TRANSPARENCY_SCALE_W,
        velocity_weight=0.5,
        velocity_scale_mps=DEFAULT_VELOCITY_ERROR_SCALE_MPS,
        force_diff_weight=1.0,
        force_diff_scale_n=DEFAULT_FORCE_DIFF_SCALE_N,
        effort_weight=0.0,
        effort_scale_v=DEFAULT_ACTION_SCALE_V,
        jerk_scale_v=DEFAULT_ACTION_DELTA_SCALE_V,
        tracking_error_fail_penalty=1000.0,
    )


def normalized_legacy_transparency_reward_variant() -> RewardVariant:
    return RewardVariant(
        name="norm_legacy_trans_v1",
        tracking_weight=1.0,
        transparency_weight=1.0,
        jerk_weight=0.05,
        use_jerk=True,
        tracking_scale_m=DEFAULT_TRACKING_SCALE_M,
        transparency_scale_w=DEFAULT_TRANSPARENCY_SCALE_W,
        velocity_weight=0.5,
        velocity_scale_mps=DEFAULT_VELOCITY_ERROR_SCALE_MPS,
        force_diff_weight=0.0,
        force_diff_scale_n=DEFAULT_FORCE_DIFF_SCALE_N,
        effort_weight=0.0,
        effort_scale_v=DEFAULT_ACTION_SCALE_V,
        jerk_scale_v=DEFAULT_ACTION_DELTA_SCALE_V,
        tracking_error_fail_penalty=1000.0,
    )


def build_full_reward_variants() -> list[RewardVariant]:
    return [
        equal_gradient_reward_variant(),
        normalized_force_shape_reward_variant(),
        normalized_legacy_transparency_reward_variant(),
        RewardVariant("r01_t40_tr06_j005", 40.0, 6.0, 0.05, True),
        RewardVariant("r02_t50_tr06_j005", 50.0, 6.0, 0.05, True),
        RewardVariant("r03_t50_tr08_j010", 50.0, 8.0, 0.10, True),
        RewardVariant("r04_t60_tr08_j010", 60.0, 8.0, 0.10, True),
        RewardVariant("r05_t60_tr10_j015", 60.0, 10.0, 0.15, True),
        RewardVariant("r06_t70_tr10_j020", 70.0, 10.0, 0.20, True),
        RewardVariant("r07_t80_tr12_j025", 80.0, 12.0, 0.25, True),
        RewardVariant("r08_t50_tr06_nojerk", 50.0, 6.0, 0.00, False),
        RewardVariant("r09_t60_tr08_nojerk", 60.0, 8.0, 0.00, False),
        RewardVariant("r10_t70_tr10_nojerk", 70.0, 10.0, 0.00, False),
        RewardVariant("r11_t40_tr06_nojerk", 40.0, 6.0, 0.00, False),
    ]


def build_core_reward_variants() -> list[RewardVariant]:
    wanted = {
        "eqgrad_t40_tr40_nojerk",
        "r11_t40_tr06_nojerk",
        "r09_t60_tr08_nojerk",
        "r10_t70_tr10_nojerk",
        "r01_t40_tr06_j005",
        "r04_t60_tr08_j010",
        "r06_t70_tr10_j020",
    }
    return [variant for variant in build_full_reward_variants() if variant.name in wanted]


def reward_variant_from_name(name: str) -> RewardVariant:
    variants = {variant.name: variant for variant in build_full_reward_variants()}
    if name == "baseline_cfg":
        return baseline_reward_variant()
    if name not in variants:
        raise KeyError(f"Unknown reward variant: {name}")
    return variants[name]


def _normalized_square(value: float, scale: float) -> float:
    safe_scale = max(abs(float(scale)), 1e-9)
    return (float(value) / safe_scale) ** 2


def compute_reward_terms(
    *,
    pos_error: float,
    velocity_error: float,
    transparency_error: float,
    force_diff: float,
    u_v: float,
    action_delta: float,
    variant: RewardVariant,
) -> tuple[float, float, float, float, float, float, float]:
    track_term = float(variant.tracking_weight) * _normalized_square(pos_error, variant.tracking_scale_m)
    velocity_term = float(variant.velocity_weight) * _normalized_square(
        velocity_error,
        variant.velocity_scale_mps,
    )
    transparency_term = float(variant.transparency_weight) * _normalized_square(
        transparency_error,
        variant.transparency_scale_w,
    )
    force_diff_term = float(variant.force_diff_weight) * _normalized_square(
        force_diff,
        variant.force_diff_scale_n,
    )
    effort_term = float(variant.effort_weight) * _normalized_square(u_v, variant.effort_scale_v)
    jerk_term = (
        float(variant.jerk_weight) * _normalized_square(action_delta, variant.jerk_scale_v)
        if variant.use_jerk
        else 0.0
    )
    reward = -(track_term + velocity_term + transparency_term + force_diff_term + effort_term + jerk_term)
    return reward, track_term, transparency_term, effort_term, jerk_term, velocity_term, force_diff_term


def _edge_severity(position: float, stroke_max: float, buffer_m: float) -> float:
    if buffer_m <= 0.0 or stroke_max <= 0.0:
        return 0.0
    dist_to_edge = min(float(position), float(stroke_max) - float(position))
    severity = 1.0 - (dist_to_edge / float(buffer_m))
    return float(np.clip(severity, 0.0, 1.0))


class ReplicaRewardEnv:
    """Reward wrapper for the SimuOriginal replica env."""

    def __init__(self, base_env: Any, variant: RewardVariant):
        self.base_env = base_env
        self.variant = variant
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        self._prev_u_v = 0.0
        self._reward_history: dict[str, list[Any]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    def reset(self, *args, **kwargs):
        obs, info = self.base_env.reset(*args, **kwargs)
        self._prev_u_v = 0.0
        self._reward_history = {
            "reward": [],
            "reward_track": [],
            "reward_velocity": [],
            "reward_transparency": [],
            "reward_force_diff": [],
            "reward_effort": [],
            "reward_jerk": [],
            "reward_edge": [],
            "reward_low_force_edge": [],
            "reward_terminal_penalty": [],
            "action_delta": [],
            "reward_variant_name": [],
        }
        return obs, info

    def _compute_reward(
        self,
        terminated: bool,
        info: dict[str, Any],
    ) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
        history = self.base_env.render() or {}
        pos_error = float(history.get("pos_error", [0.0])[-1])
        velocity_error = float(history.get("v_m", [0.0])[-1]) - float(history.get("v_s", [0.0])[-1])
        transparency_error = float(history.get("transparency_error", [0.0])[-1])
        force_diff = float(history.get("F_e", [0.0])[-1]) - float(history.get("F_h", [0.0])[-1])
        u_v = float(history.get("u_v", [0.0])[-1])
        action_delta = float(u_v - self._prev_u_v)
        self._prev_u_v = u_v
        reward, track_term, transparency_term, effort_term, jerk_term, velocity_term, force_diff_term = compute_reward_terms(
            pos_error=pos_error,
            velocity_error=velocity_error,
            transparency_error=transparency_error,
            force_diff=force_diff,
            u_v=u_v,
            action_delta=action_delta,
            variant=self.variant,
        )
        edge_penalty = 0.0
        low_force_edge_penalty = 0.0
        edge_buffer_m = float(self.variant.edge_buffer_m)
        if edge_buffer_m > 0.0:
            x_m = float(info.get("x_m", 0.0))
            x_s = float(info.get("x_s", 0.0))
            stroke_max = float(getattr(getattr(self.base_env, "parms", None), "l_cyl", 0.0) or 0.0)
            edge_severity = max(
                _edge_severity(x_m, stroke_max, edge_buffer_m),
                _edge_severity(x_s, stroke_max, edge_buffer_m),
            )
            edge_penalty = float(self.variant.edge_penalty_weight) * (edge_severity ** 2)
            low_force_threshold_n = float(self.variant.low_force_threshold_n)
            if low_force_threshold_n > 0.0:
                low_force_scale = 1.0 - min(abs(float(info.get("F_h", 0.0))) / low_force_threshold_n, 1.0)
                low_force_edge_penalty = (
                    float(self.variant.low_force_edge_penalty_weight)
                    * (edge_severity ** 2)
                    * low_force_scale
                )
        terminal_penalty = 0.0
        if terminated:
            termination_reason = str(info.get("termination_reason") or "")
            invalid_reason = str(info.get("invalid_reason") or "")
            if bool(info.get("tracking_error_fail")) or termination_reason == "tracking_error_fail":
                terminal_penalty = float(self.variant.tracking_error_fail_penalty)
            elif invalid_reason == "stroke_limit" or termination_reason == "stroke_limit":
                terminal_penalty = float(self.variant.stroke_limit_penalty)
            elif bool(info.get("invalid_state")):
                terminal_penalty = float(self.variant.invalid_state_penalty)
        reward -= edge_penalty
        reward -= low_force_edge_penalty
        reward -= terminal_penalty
        return (
            reward,
            track_term,
            transparency_term,
            effort_term,
            jerk_term,
            edge_penalty,
            low_force_edge_penalty,
            terminal_penalty,
            action_delta,
            velocity_term,
            force_diff_term,
        )

    def _record_reward_terms(
        self,
        reward: float,
        track_term: float,
        transparency_term: float,
        effort_term: float,
        jerk_term: float,
        edge_penalty: float,
        low_force_edge_penalty: float,
        terminal_penalty: float,
        action_delta: float,
        velocity_term: float,
        force_diff_term: float,
    ) -> None:
        self._reward_history["reward"].append(float(reward))
        self._reward_history["reward_track"].append(float(track_term))
        self._reward_history["reward_velocity"].append(float(velocity_term))
        self._reward_history["reward_transparency"].append(float(transparency_term))
        self._reward_history["reward_force_diff"].append(float(force_diff_term))
        self._reward_history["reward_effort"].append(float(effort_term))
        self._reward_history["reward_jerk"].append(float(jerk_term))
        self._reward_history["reward_edge"].append(float(edge_penalty))
        self._reward_history["reward_low_force_edge"].append(float(low_force_edge_penalty))
        self._reward_history["reward_terminal_penalty"].append(float(terminal_penalty))
        self._reward_history["action_delta"].append(float(action_delta))
        self._reward_history["reward_variant_name"].append(self.variant.name)

    def step(self, action):
        obs, _, terminated, truncated, info = self.base_env.step(action)
        (
            reward,
            track_term,
            transparency_term,
            effort_term,
            jerk_term,
            edge_penalty,
            low_force_edge_penalty,
            terminal_penalty,
            action_delta,
            velocity_term,
            force_diff_term,
        ) = self._compute_reward(terminated, info)
        self._record_reward_terms(
            reward,
            track_term,
            transparency_term,
            effort_term,
            jerk_term,
            edge_penalty,
            low_force_edge_penalty,
            terminal_penalty,
            action_delta,
            velocity_term,
            force_diff_term,
        )
        return obs, reward, terminated, truncated, info

    def step_voltage(self, u_v: float):
        obs, _, terminated, truncated, info = self.base_env.step_voltage(u_v)
        (
            reward,
            track_term,
            transparency_term,
            effort_term,
            jerk_term,
            edge_penalty,
            low_force_edge_penalty,
            terminal_penalty,
            action_delta,
            velocity_term,
            force_diff_term,
        ) = self._compute_reward(terminated, info)
        self._record_reward_terms(
            reward,
            track_term,
            transparency_term,
            effort_term,
            jerk_term,
            edge_penalty,
            low_force_edge_penalty,
            terminal_penalty,
            action_delta,
            velocity_term,
            force_diff_term,
        )
        return obs, reward, terminated, truncated, info

    def render(self):
        base_history = self.base_env.render() or {}
        merged: dict[str, Any] = {}
        for key, value in base_history.items():
            merged[key] = list(value) if isinstance(value, list) else value
        merged["base_reward"] = list(base_history.get("reward", []))
        merged["base_reward_track"] = list(base_history.get("reward_track", []))
        merged["base_reward_transparency"] = list(base_history.get("reward_transparency", []))
        merged["reward"] = list(self._reward_history.get("reward", []))
        merged["reward_track"] = list(self._reward_history.get("reward_track", []))
        merged["reward_velocity"] = list(self._reward_history.get("reward_velocity", []))
        merged["reward_transparency"] = list(self._reward_history.get("reward_transparency", []))
        merged["reward_force_diff"] = list(self._reward_history.get("reward_force_diff", []))
        merged["reward_effort"] = list(self._reward_history.get("reward_effort", []))
        merged["reward_jerk"] = list(self._reward_history.get("reward_jerk", []))
        merged["reward_edge"] = list(self._reward_history.get("reward_edge", []))
        merged["reward_low_force_edge"] = list(self._reward_history.get("reward_low_force_edge", []))
        merged["reward_terminal_penalty"] = list(self._reward_history.get("reward_terminal_penalty", []))
        merged["action_delta"] = list(self._reward_history.get("action_delta", []))
        merged["reward_variant_name"] = list(self._reward_history.get("reward_variant_name", []))
        return merged
