"""Reward definitions and the reward-wrapping environment adapter.

Reward variants are represented as reusable specifications so each algorithm
and notebook evaluates the same tracking, effort, and transparency terms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import config as cfg
from ..environment.simuoriginal_env import force_velocity_transparency_error, force_velocity_transparency_ratio

_ACTION_LEVELS = np.asarray(getattr(cfg, "V_LEVELS", [-5.0, 5.0]), dtype=np.float64).reshape(-1)
DEFAULT_ACTION_SCALE_V = float(np.max(np.abs(_ACTION_LEVELS))) if _ACTION_LEVELS.size else 1.0
DEFAULT_ACTION_DELTA_SCALE_V = float(max(1.0, 2.0 * DEFAULT_ACTION_SCALE_V))
DEFAULT_TRACKING_SCALE_M = float(cfg.L_CYL)
DEFAULT_VELOCITY_ERROR_SCALE_MPS = 3.0
DEFAULT_FORCE_DIFF_SCALE_N = 25.0
DEFAULT_TRANSPARENCY_SCALE_RATIO = 1.0
DEFAULT_TRANSPARENCY_SCALE_W = float(cfg.MAX_POWER_ERROR)
DEFAULT_SLIDING_LAMBDA = 3.0
DEFAULT_SECOND_ORDER_ZETA = 0.8
DEFAULT_SECOND_ORDER_OMEGA_N = 3.0
DEFAULT_HIGH_PASS_TAU_S = 0.5


@dataclass(frozen=True)
class RewardVariant:
    """Immutable reward specification shared by all algorithm families.

    Weights multiply normalized tracking, transparency, effort, velocity,
    force-difference, jerk, and boundary terms. ``formula_terms`` optionally
    replaces the built-in scalar formula with a JSON-style term list.
    """

    name: str
    tracking_weight: float
    transparency_weight: float
    jerk_weight: float
    use_jerk: bool
    tracking_scale_m: float = float(cfg.MAX_POSITION_ERROR)
    transparency_scale_w: float = DEFAULT_TRANSPARENCY_SCALE_W
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
    formula_terms: tuple[dict[str, Any], ...] = ()


def baseline_reward_variant() -> RewardVariant:
    """Return the configuration-weighted baseline reward."""
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
    """Return the equal tracking/transparency-gradient reward variant."""
    baseline = baseline_reward_variant()
    return replace(
        baseline,
        name="eqgrad_t40_tr40_nojerk",
        transparency_weight=float(baseline.tracking_weight),
    )


def tracking_effort_no_force_transparency_reward_variant() -> RewardVariant:
    """Return tracking-plus-effort reward with transparency disabled."""
    baseline = baseline_reward_variant()
    return replace(
        baseline,
        name="track_effort_no_force_trans",
        transparency_weight=0.0,
        velocity_weight=0.0,
        force_diff_weight=0.0,
        jerk_weight=0.0,
        use_jerk=False,
    )


def tracking_jerk_no_force_transparency_reward_variant() -> RewardVariant:
    """Return tracking-plus-jerk reward without force-transparency terms."""
    return RewardVariant(
        name="track_jerk_no_force_trans",
        tracking_weight=float(cfg.ALPHA_TRACKING),
        tracking_scale_m=float(cfg.POS_ERROR_FAIL_THRESHOLD),
        transparency_weight=0.0,
        jerk_weight=0.05,
        use_jerk=True,
        velocity_weight=0.0,
        force_diff_weight=0.0,
        effort_weight=0.0,
        effort_scale_v=DEFAULT_ACTION_SCALE_V,
        jerk_scale_v=DEFAULT_ACTION_DELTA_SCALE_V,
        tracking_error_fail_penalty=1000.0,
    )


def normalized_force_shape_reward_variant() -> RewardVariant:
    """Return the normalized tracking/velocity/force-shape reward."""
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
    """Return the normalized legacy transparency reward variant."""
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
    """Return the complete built-in reward-variant sweep."""
    return [
        equal_gradient_reward_variant(),
        tracking_effort_no_force_transparency_reward_variant(),
        tracking_jerk_no_force_transparency_reward_variant(),
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
    """Return the smaller reward set used by the core comparison studies."""
    wanted = {
        "eqgrad_t40_tr40_nojerk",
        "track_effort_no_force_trans",
        "track_jerk_no_force_trans",
        "r11_t40_tr06_nojerk",
        "r09_t60_tr08_nojerk",
        "r10_t70_tr10_nojerk",
        "r01_t40_tr06_j005",
        "r04_t60_tr08_j010",
        "r06_t70_tr10_j020",
    }
    return [variant for variant in build_full_reward_variants() if variant.name in wanted]


def _lookup(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def reward_variant_from_spec(spec: Mapping[str, Any]) -> RewardVariant:
    """Build a reward variant from a notebook-friendly JSON-style spec."""

    base = baseline_reward_variant()
    weights = spec.get("weights", {})
    scales = spec.get("scales", {})
    penalties = spec.get("penalties", {})
    scale_catalog = spec.get("scale_catalog", {})
    if not isinstance(weights, Mapping):
        raise TypeError("reward spec 'weights' must be an object when provided.")
    if not isinstance(scales, Mapping):
        raise TypeError("reward spec 'scales' must be an object when provided.")
    if not isinstance(penalties, Mapping):
        raise TypeError("reward spec 'penalties' must be an object when provided.")
    if not isinstance(scale_catalog, Mapping):
        raise TypeError("reward spec 'scale_catalog' must be an object when provided.")

    tracking_weight = float(
        _lookup(weights, "tracking", "tracking_weight", default=_lookup(spec, "tracking_weight", default=base.tracking_weight))
    )
    transparency_weight = float(
        _lookup(
            weights,
            "transparency",
            "transparency_weight",
            default=_lookup(spec, "transparency_weight", default=base.transparency_weight),
        )
    )
    velocity_weight = float(
        _lookup(weights, "velocity", "velocity_weight", default=_lookup(spec, "velocity_weight", default=base.velocity_weight))
    )
    force_diff_weight = float(
        _lookup(
            weights,
            "force_difference",
            "force_diff",
            "force_diff_weight",
            default=_lookup(spec, "force_diff_weight", default=base.force_diff_weight),
        )
    )
    effort_weight = float(
        _lookup(weights, "effort", "effort_weight", default=_lookup(spec, "effort_weight", default=base.effort_weight))
    )
    jerk_weight = float(
        _lookup(weights, "jerk", "jerk_weight", default=_lookup(spec, "jerk_weight", default=base.jerk_weight))
    )
    edge_penalty_weight = float(
        _lookup(
            weights,
            "edge",
            "edge_penalty",
            "edge_penalty_weight",
            default=_lookup(
                penalties,
                "edge_penalty_weight",
                "edge",
                default=_lookup(spec, "edge_penalty_weight", default=base.edge_penalty_weight),
            ),
        )
    )
    low_force_edge_penalty_weight = float(
        _lookup(
            weights,
            "low_force_edge",
            "low_force_edge_penalty",
            "low_force_edge_penalty_weight",
            default=_lookup(
                penalties,
                "low_force_edge_penalty_weight",
                "low_force_edge",
                default=_lookup(spec, "low_force_edge_penalty_weight", default=base.low_force_edge_penalty_weight),
            ),
        )
    )

    payload = {
        "name": str(spec.get("name") or "custom_reward"),
        "tracking_weight": tracking_weight,
        "transparency_weight": transparency_weight,
        "jerk_weight": jerk_weight,
        "use_jerk": bool(spec.get("use_jerk", abs(jerk_weight) > 0.0)),
        "tracking_scale_m": float(
            _lookup(scales, "tracking_m", "tracking", "tracking_scale_m", default=_lookup(spec, "tracking_scale_m", default=base.tracking_scale_m))
        ),
        "transparency_scale_w": float(
            _lookup(
                scales,
                "transparency_w",
                "transparency",
                "transparency_scale_w",
                default=_lookup(spec, "transparency_scale_w", default=base.transparency_scale_w),
            )
        ),
        "velocity_weight": velocity_weight,
        "velocity_scale_mps": float(
            _lookup(
                scales,
                "velocity_mps",
                "velocity",
                "velocity_scale_mps",
                default=_lookup(spec, "velocity_scale_mps", default=base.velocity_scale_mps),
            )
        ),
        "force_diff_weight": force_diff_weight,
        "force_diff_scale_n": float(
            _lookup(
                scales,
                "force_difference_n",
                "force_diff_n",
                "force_diff",
                "force_diff_scale_n",
                default=_lookup(spec, "force_diff_scale_n", default=base.force_diff_scale_n),
            )
        ),
        "effort_weight": effort_weight,
        "effort_scale_v": float(
            _lookup(scales, "effort_v", "effort", "effort_scale_v", default=_lookup(spec, "effort_scale_v", default=base.effort_scale_v))
        ),
        "jerk_scale_v": float(
            _lookup(scales, "jerk_v", "jerk", "jerk_scale_v", default=_lookup(spec, "jerk_scale_v", default=base.jerk_scale_v))
        ),
        "stroke_limit_penalty": float(
            _lookup(penalties, "stroke_limit", "stroke_limit_penalty", default=_lookup(spec, "stroke_limit_penalty", default=base.stroke_limit_penalty))
        ),
        "invalid_state_penalty": float(
            _lookup(penalties, "invalid_state", "invalid_state_penalty", default=_lookup(spec, "invalid_state_penalty", default=base.invalid_state_penalty))
        ),
        "tracking_error_fail_penalty": float(
            _lookup(
                penalties,
                "tracking_error_fail",
                "tracking_error_fail_penalty",
                default=_lookup(spec, "tracking_error_fail_penalty", default=base.tracking_error_fail_penalty),
            )
        ),
        "edge_buffer_m": float(
            _lookup(penalties, "edge_buffer_m", default=_lookup(spec, "edge_buffer_m", default=base.edge_buffer_m))
        ),
        "edge_penalty_weight": edge_penalty_weight,
        "low_force_threshold_n": float(
            _lookup(
                penalties,
                "low_force_threshold_n",
                default=_lookup(spec, "low_force_threshold_n", default=base.low_force_threshold_n),
            )
        ),
        "low_force_edge_penalty_weight": low_force_edge_penalty_weight,
        "formula_terms": tuple(
            _normalize_formula_term(_resolve_formula_term_scale(term, scale_catalog), index)
            for index, term in enumerate(spec.get("terms", spec.get("formula_terms", ())), start=1)
        ),
    }

    valid_fields = {field.name for field in fields(RewardVariant)}
    extras = {key: value for key, value in spec.items() if key in valid_fields and key not in payload}
    payload.update(extras)
    return RewardVariant(**payload)


def load_reward_variant_from_json(path: str | Path) -> RewardVariant:
    """Load one reward specification from a JSON object on disk."""
    with open(Path(path), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    if not isinstance(spec, Mapping):
        raise TypeError("Reward spec JSON must contain an object.")
    return reward_variant_from_spec(spec)


def reward_variant_from_name(name: str) -> RewardVariant:
    """Resolve a built-in reward name or an existing JSON specification path."""
    candidate = Path(str(name))
    if candidate.suffix.lower() == ".json" and candidate.exists():
        return load_reward_variant_from_json(candidate)
    variants = {variant.name: variant for variant in build_full_reward_variants()}
    if name == "baseline_cfg":
        return baseline_reward_variant()
    if name not in variants:
        raise KeyError(f"Unknown reward variant: {name}")
    return variants[name]


def _normalized_square(value: float, scale: float) -> float:
    safe_scale = max(abs(float(scale)), 1e-9)
    return (float(value) / safe_scale) ** 2


def _safe_name(value: Any) -> str:
    raw = str(value).strip() or "term"
    chars = [ch if ch.isalnum() else "_" for ch in raw]
    name = "".join(chars).strip("_").lower()
    return name or "term"


def _normalize_formula_term(term: Mapping[str, Any], index: int) -> dict[str, Any]:
    if "source" not in term:
        raise KeyError(f"Reward term {index} is missing required key 'source'.")
    name = _safe_name(term.get("name") or f"term_{index}")
    sign = str(term.get("sign", "penalty")).strip().lower()
    if sign in {"-", "cost", "loss"}:
        sign = "penalty"
    if sign in {"+", "reward"}:
        sign = "bonus"
    if sign not in {"penalty", "bonus"}:
        raise ValueError(f"Reward term '{name}' has unknown sign '{sign}'. Use 'penalty' or 'bonus'.")

    return {
        "name": name,
        "source": str(term["source"]).strip(),
        "shape": str(term.get("shape", "square")).strip().lower(),
        "sign": sign,
        "scale_name": str(term.get("scale_name", "")).strip(),
        "weight": float(term.get("weight", 1.0)),
        "scale": float(term.get("scale", 1.0)),
        "target": float(term.get("target", 0.0)),
        "deadband": float(term.get("deadband", 0.0)),
        "threshold": float(term.get("threshold", 0.0)),
        "margin": float(term.get("margin", term.get("scale", 1.0))),
        "clip": None if term.get("clip") is None else float(term.get("clip")),
    }


def _scale_catalog_value(scale_catalog: Mapping[str, Any], scale_name: str) -> float:
    if scale_name not in scale_catalog:
        known = ", ".join(sorted(str(key) for key in scale_catalog.keys()))
        raise KeyError(f"Unknown reward scale_name '{scale_name}'. Known scales: {known}")
    entry = scale_catalog[scale_name]
    if isinstance(entry, Mapping):
        if "value" not in entry:
            raise KeyError(f"Reward scale '{scale_name}' must contain a 'value' field.")
        entry = entry["value"]
    return float(entry)


def _resolve_formula_term_scale(term: Mapping[str, Any], scale_catalog: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(term)
    scale_name = str(resolved.get("scale_name", "")).strip()
    if scale_name and "scale" not in resolved:
        resolved["scale"] = _scale_catalog_value(scale_catalog, scale_name)
    margin_name = str(resolved.get("margin_name", "")).strip()
    if margin_name and "margin" not in resolved:
        resolved["margin"] = _scale_catalog_value(scale_catalog, margin_name)
    deadband_name = str(resolved.get("deadband_name", "")).strip()
    if deadband_name and "deadband" not in resolved:
        resolved["deadband"] = _scale_catalog_value(scale_catalog, deadband_name)
    threshold_name = str(resolved.get("threshold_name", "")).strip()
    if threshold_name and "threshold" not in resolved:
        resolved["threshold"] = _scale_catalog_value(scale_catalog, threshold_name)
    return resolved


def _shape_value(raw_value: float, term: Mapping[str, Any]) -> float:
    shape = str(term.get("shape", "square")).strip().lower()
    centered = float(raw_value) - float(term.get("target", 0.0))
    scale = max(abs(float(term.get("scale", 1.0))), 1e-9)
    normalized = centered / scale
    clip = term.get("clip")
    if clip is not None:
        limit = abs(float(clip))
        normalized = float(np.clip(normalized, -limit, limit))

    if shape in {"square", "squared", "l2"}:
        return float(normalized ** 2)
    if shape in {"absolute", "abs", "l1"}:
        return float(abs(normalized))
    if shape in {"linear", "signed"}:
        return float(normalized)
    if shape in {"deadband_square", "deadband_squared"}:
        excess = max(abs(centered) - max(float(term.get("deadband", 0.0)), 0.0), 0.0)
        return float((excess / scale) ** 2)
    if shape in {"deadband_abs", "deadband_absolute"}:
        excess = max(abs(centered) - max(float(term.get("deadband", 0.0)), 0.0), 0.0)
        return float(excess / scale)
    if shape in {"above_threshold_square", "hinge_square"}:
        excess = max(float(raw_value) - float(term.get("threshold", 0.0)), 0.0)
        return float((excess / scale) ** 2)
    if shape in {"above_threshold_abs", "hinge_abs"}:
        excess = max(float(raw_value) - float(term.get("threshold", 0.0)), 0.0)
        return float(excess / scale)
    if shape in {"below_threshold_square"}:
        excess = max(float(term.get("threshold", 0.0)) - float(raw_value), 0.0)
        return float((excess / scale) ** 2)
    if shape in {"below_threshold_abs"}:
        excess = max(float(term.get("threshold", 0.0)) - float(raw_value), 0.0)
        return float(excess / scale)
    if shape in {"tolerance_bonus", "triangle_bonus"}:
        margin = max(abs(float(term.get("margin", scale))), 1e-9)
        return float(max(0.0, 1.0 - (abs(centered) / margin)))
    if shape in {"gaussian_bonus", "rbf_bonus"}:
        return float(np.exp(-0.5 * (centered / scale) ** 2))
    raise ValueError(f"Unknown reward term shape: {shape}")


def _term_group(term: Mapping[str, Any]) -> str:
    source = str(term.get("source", ""))
    name = str(term.get("name", ""))
    text = f"{source} {name}".lower()
    if "pos_error" in text or "tracking" in text:
        return "track"
    if "velocity" in text:
        return "velocity"
    if "transparency" in text:
        return "transparency"
    if "force_diff" in text or "force_difference" in text:
        return "force_diff"
    if source in {"u_v", "requested_u_v"} or "effort" in text:
        return "effort"
    if "action_delta" in text or "jerk" in text:
        return "jerk"
    if "edge" in text:
        return "edge"
    return "other"


def reward_formula_from_context(
    context: Mapping[str, float],
    variant: RewardVariant,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Evaluate a custom formula and return total, grouped, and term rewards."""
    formula_terms = tuple(
        _normalize_formula_term(term, index)
        for index, term in enumerate(variant.formula_terms, start=1)
    )
    return _reward_formula_from_terms(context, formula_terms)


def _reward_formula_from_terms(
    context: Mapping[str, float],
    formula_terms: tuple[Mapping[str, Any], ...],
) -> tuple[float, dict[str, float], dict[str, float]]:
    grouped = {
        "track": 0.0,
        "velocity": 0.0,
        "transparency": 0.0,
        "force_diff": 0.0,
        "effort": 0.0,
        "jerk": 0.0,
        "edge": 0.0,
        "low_force_edge": 0.0,
        "terminal_penalty": 0.0,
    }
    custom_terms: dict[str, float] = {}
    reward = 0.0

    for term in formula_terms:
        source = str(term["source"])
        if source not in context:
            known = ", ".join(sorted(context.keys()))
            raise KeyError(f"Reward term '{term['name']}' uses unknown source '{source}'. Known sources: {known}")
        shaped = _shape_value(float(context[source]), term)
        magnitude = float(term["weight"]) * shaped
        signed = magnitude if term["sign"] == "bonus" else -magnitude
        reward += signed
        custom_terms[str(term["name"])] = float(signed)

        group = _term_group(term)
        if group in grouped:
            grouped[group] += magnitude if term["sign"] == "penalty" else -magnitude

    return float(reward), grouped, custom_terms


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
    """Compute total reward and seven decomposed scalar contributions.

    The return order is ``reward, tracking, transparency, effort, jerk,
    velocity, force_difference``. Inputs are one-transition error/control
    quantities; scales and weights come from ``variant``.
    """
    if variant.formula_terms:
        transparency_ratio = float("nan")
        if abs(transparency_error) < 1e-12:
            transparency_ratio = 1.0
        context = {
            "pos_error": float(pos_error),
            "tracking_error": float(pos_error),
            "abs_pos_error": abs(float(pos_error)),
            "velocity_error": float(velocity_error),
            "abs_velocity_error": abs(float(velocity_error)),
            "transparency_ratio": transparency_ratio,
            "abs_transparency_ratio": abs(transparency_ratio),
            "transparency_error": float(transparency_error),
            "abs_transparency_error": abs(float(transparency_error)),
            "force_diff": float(force_diff),
            "abs_force_diff": abs(float(force_diff)),
            "u_v": float(u_v),
            "abs_u_v": abs(float(u_v)),
            "action_delta": float(action_delta),
            "abs_action_delta": abs(float(action_delta)),
        }
        reward, grouped, _ = reward_formula_from_context(context, variant)
        return (
            reward,
            float(grouped["track"]),
            float(grouped["transparency"]),
            float(grouped["effort"]),
            float(grouped["jerk"]),
            float(grouped["velocity"]),
            float(grouped["force_diff"]),
        )

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
    """Reward wrapper preserving the base environment's reset/step contract.

    The wrapper computes per-transition reward terms from the base history and
    records them in ``render()``. Unknown attributes are forwarded to
    ``base_env`` so runners and state encoders retain the original metadata.
    """

    def __init__(self, base_env: Any, variant: RewardVariant):
        self.base_env = base_env
        self.variant = variant
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        self._prev_u_v = 0.0
        self._prev_action_delta = 0.0
        self._prev_tracking_energy: float | None = None
        self._tracking_error_lp = 0.0
        self._u_v_lp = 0.0
        self._high_pass_initialized = False
        self._reward_history: dict[str, list[Any]] = {}
        self._formula_terms = tuple(
            _normalize_formula_term(term, index)
            for index, term in enumerate(self.variant.formula_terms, start=1)
        )
        self._custom_reward_keys = tuple(
            f"reward_term_{term['name']}"
            for term in self._formula_terms
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    def _custom_term_keys(self) -> list[str]:
        return list(self._custom_reward_keys)

    def reset(self, *args, **kwargs):
        """Reset the base environment and clear reward-term history."""
        obs, info = self.base_env.reset(*args, **kwargs)
        self._prev_u_v = 0.0
        self._prev_action_delta = 0.0
        self._prev_tracking_energy = None
        self._tracking_error_lp = 0.0
        self._u_v_lp = 0.0
        self._high_pass_initialized = False
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
        for key in self._custom_term_keys():
            self._reward_history[key] = []
        return obs, info

    @staticmethod
    def _last(history: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        values = history.get(key, [])
        if isinstance(values, (list, tuple)) and values:
            value = values[-1]
        else:
            value = default
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _formula_context(
        self,
        history: Mapping[str, Any],
        info: Mapping[str, Any],
        *,
        action_delta: float,
        action_delta2: float,
        lyapunov_increase: float,
        tracking_error_hf: float,
        u_v_hf: float,
    ) -> dict[str, float]:
        x_m = float(info.get("x_m", self._last(history, "x_m")))
        x_s = float(info.get("x_s", self._last(history, "x_s")))
        v_m = self._last(history, "v_m")
        v_s = self._last(history, "v_s")
        f_h = float(info.get("F_h", self._last(history, "F_h")))
        f_e = float(info.get("F_e", self._last(history, "F_e")))
        pos_error = self._last(history, "pos_error", x_m - x_s)
        velocity_error = v_m - v_s
        x_m_ddot = float(info.get("a_m_signal", self._last(history, "a_m_signal")))
        x_s_ddot = float(info.get("a_s_signal", self._last(history, "a_s_signal")))
        acceleration_error = x_m_ddot - x_s_ddot
        sliding_error = velocity_error + (DEFAULT_SLIDING_LAMBDA * pos_error)
        second_order_error = (
            acceleration_error
            + (2.0 * DEFAULT_SECOND_ORDER_ZETA * DEFAULT_SECOND_ORDER_OMEGA_N * velocity_error)
            + ((DEFAULT_SECOND_ORDER_OMEGA_N ** 2) * pos_error)
        )
        fallback_transparency_ratio = force_velocity_transparency_ratio(f_h, v_m, f_e, v_s)
        fallback_transparency_error = force_velocity_transparency_error(f_h, v_m, f_e, v_s)
        transparency_ratio = float(
            info.get(
                "transparency_ratio",
                self._last(history, "transparency_ratio", fallback_transparency_ratio),
            )
        )
        transparency_error = float(
            info.get(
                "transparency_error",
                self._last(history, "transparency_error", fallback_transparency_error),
            )
        )
        force_diff = f_e - f_h
        u_v = self._last(history, "u_v", float(info.get("u_v", 0.0)))
        requested_u_v = self._last(history, "requested_u_v", float(info.get("requested_u_v", u_v)))
        time_s = self._last(history, "time", float(info.get("time", 0.0)))
        episode_duration = float(info.get("episode_duration", getattr(self.base_env, "episode_duration", cfg.EPISODE_DURATION)))

        stroke_max = float(getattr(getattr(self.base_env, "parms", None), "l_cyl", 0.0) or 0.0)
        edge_buffer_m = float(self.variant.edge_buffer_m)
        edge_severity = 0.0
        if edge_buffer_m > 0.0:
            edge_severity = max(
                _edge_severity(x_m, stroke_max, edge_buffer_m),
                _edge_severity(x_s, stroke_max, edge_buffer_m),
            )
        low_force_threshold_n = float(self.variant.low_force_threshold_n)
        low_force_scale = (
            1.0 - min(abs(f_h) / low_force_threshold_n, 1.0)
            if low_force_threshold_n > 0.0
            else 0.0
        )

        context = {
            "time": time_s,
            "time_fraction": time_s / max(abs(episode_duration), 1e-9),
            "env_id": float(info.get("env_id", self._last(history, "env_id"))),
            "x_m": x_m,
            "x_s": x_s,
            "x_m_centered": float(info.get("x_m_centered", self._last(history, "x_m_centered"))),
            "x_s_centered": float(info.get("x_s_centered", self._last(history, "x_s_centered"))),
            "v_m": v_m,
            "v_s": v_s,
            "P_m1": self._last(history, "P_m1"),
            "P_m2": self._last(history, "P_m2"),
            "P_s1": self._last(history, "P_s1"),
            "P_s2": self._last(history, "P_s2"),
            "mdot_L1": self._last(history, "mdot_L1"),
            "mdot_L2": self._last(history, "mdot_L2"),
            "x_v": float(info.get("x_v", self._last(history, "x_v"))),
            "x_v_dot": float(info.get("x_v_dot", self._last(history, "x_v_dot"))),
            "F_h": f_h,
            "F_e": f_e,
            "x_m_ddot": x_m_ddot,
            "x_s_ddot": x_s_ddot,
            "a_m_signal": x_m_ddot,
            "a_s_signal": x_s_ddot,
            "u_v": u_v,
            "requested_u_v": requested_u_v,
            "action_delta": float(action_delta),
            "action_delta2": float(action_delta2),
            "pos_error": pos_error,
            "tracking_error": pos_error,
            "velocity_error": velocity_error,
            "acceleration_error": acceleration_error,
            "tracking_error_ddot": acceleration_error,
            "sliding_error": sliding_error,
            "second_order_error": second_order_error,
            "lyapunov_increase": float(lyapunov_increase),
            "phase_lag_proxy": pos_error * v_m,
            "direction_disagreement": max(0.0, -(v_m * v_s)),
            "tracking_error_hf": float(tracking_error_hf),
            "u_v_hf": float(u_v_hf),
            "transparency_ratio": transparency_ratio,
            "transparency_error": transparency_error,
            "force_diff": force_diff,
            "delta_P_m": self._last(history, "P_m1") - self._last(history, "P_m2"),
            "delta_P_s": self._last(history, "P_s1") - self._last(history, "P_s2"),
            "P_m1_minus_P_s1": self._last(history, "P_m1") - self._last(history, "P_s1"),
            "P_m2_minus_P_s2": self._last(history, "P_m2") - self._last(history, "P_s2"),
            "edge_severity": edge_severity,
            "low_force_edge_severity": edge_severity * low_force_scale,
        }
        for key, value in list(context.items()):
            context[f"abs_{key}"] = abs(float(value))
        return context

    def _high_pass_values(self, pos_error: float, u_v: float) -> tuple[float, float]:
        if not self._high_pass_initialized:
            self._tracking_error_lp = float(pos_error)
            self._u_v_lp = float(u_v)
            self._high_pass_initialized = True
            return 0.0, 0.0
        dt = float(getattr(cfg, "RL_DT", 0.02))
        alpha = dt / max(float(DEFAULT_HIGH_PASS_TAU_S) + dt, 1e-9)
        self._tracking_error_lp += alpha * (float(pos_error) - self._tracking_error_lp)
        self._u_v_lp += alpha * (float(u_v) - self._u_v_lp)
        return float(pos_error) - self._tracking_error_lp, float(u_v) - self._u_v_lp

    def _compute_reward(
        self,
        terminated: bool,
        info: dict[str, Any],
    ) -> tuple[float, float, float, float, float, float, float, float, float, float, float, dict[str, float]]:
        history = self.base_env.render() or {}
        pos_error = self._last(history, "pos_error")
        v_m = self._last(history, "v_m")
        v_s = self._last(history, "v_s")
        f_h = self._last(history, "F_h")
        f_e = self._last(history, "F_e")
        velocity_error = v_m - v_s
        fallback_transparency_ratio = force_velocity_transparency_ratio(f_h, v_m, f_e, v_s)
        fallback_transparency_error = force_velocity_transparency_error(f_h, v_m, f_e, v_s)
        transparency_ratio = float(
            info.get(
                "transparency_ratio",
                self._last(history, "transparency_ratio", fallback_transparency_ratio),
            )
        )
        transparency_error = float(
            info.get(
                "transparency_error",
                self._last(history, "transparency_error", fallback_transparency_error),
            )
        )
        force_diff = f_e - f_h
        u_v = self._last(history, "u_v")
        action_delta = float(u_v - self._prev_u_v)
        action_delta2 = float(action_delta - self._prev_action_delta)
        tracking_energy = 0.5 * (float(pos_error) ** 2) + 0.5 * (float(velocity_error) ** 2)
        lyapunov_increase = (
            max(0.0, tracking_energy - float(self._prev_tracking_energy))
            if self._prev_tracking_energy is not None
            else 0.0
        )
        tracking_error_hf, u_v_hf = self._high_pass_values(pos_error, u_v)

        custom_terms: dict[str, float] = {}
        if self.variant.formula_terms:
            context = self._formula_context(
                history,
                info,
                action_delta=action_delta,
                action_delta2=action_delta2,
                lyapunov_increase=lyapunov_increase,
                tracking_error_hf=tracking_error_hf,
                u_v_hf=u_v_hf,
            )
            reward, grouped_terms, custom_terms = _reward_formula_from_terms(context, self._formula_terms)
            track_term = grouped_terms["track"]
            transparency_term = grouped_terms["transparency"]
            effort_term = grouped_terms["effort"]
            jerk_term = grouped_terms["jerk"]
            velocity_term = grouped_terms["velocity"]
            force_diff_term = grouped_terms["force_diff"]
        else:
            reward, track_term, transparency_term, effort_term, jerk_term, velocity_term, force_diff_term = compute_reward_terms(
                pos_error=pos_error,
                velocity_error=velocity_error,
                transparency_error=transparency_error,
                force_diff=force_diff,
                u_v=u_v,
                action_delta=action_delta,
                variant=self.variant,
            )

        self._prev_u_v = u_v
        self._prev_action_delta = action_delta
        self._prev_tracking_energy = tracking_energy

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
            custom_terms,
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
        custom_terms: Mapping[str, float] | None = None,
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
        custom_terms = dict(custom_terms or {})
        for key in self._custom_term_keys():
            name = key.replace("reward_term_", "", 1)
            self._reward_history[key].append(float(custom_terms.get(name, 0.0)))

    def step(self, action):
        """Step the base environment and return the variant-computed reward."""
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
            custom_terms,
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
            custom_terms,
        )
        return obs, reward, terminated, truncated, info

    def step_voltage(self, u_v: float):
        """Apply a scalar voltage through the base environment's direct API."""
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
            custom_terms,
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
            custom_terms,
        )
        return obs, reward, terminated, truncated, info

    def render(self):
        """Return base history merged with decomposed reward histories."""
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
        for key in self._custom_term_keys():
            merged[key] = list(self._reward_history.get(key, []))
        return merged
