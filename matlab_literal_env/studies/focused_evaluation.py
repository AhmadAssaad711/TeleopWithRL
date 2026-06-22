from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ... import config as cfg
from ..simuoriginal_replica import SimuOriginalProfile
from .common import (
    history_array,
    plot_transparency_ratio_monitor,
    save_json,
    transparency_power_error_array,
    transparency_ratio_array,
    transparency_ratio_metrics,
)
from .policy_gradient import (
    PG_ALGO_PPO_CONTINUOUS,
    PG_ALGO_SAC,
    PG_ALGO_TD3,
    build_policy_gradient_env_factory,
    get_policy_gradient_reward_variant,
    get_policy_gradient_state_variant,
    require_sb3,
)
from .dqn_state_variants import build_custom_dqn_state_variant_from_spec
from .rewarding import reward_variant_from_spec


_REPLICA_XM_DOT = 2
_REPLICA_XM = 3
_REPLICA_XS_DOT = 6
_REPLICA_XS = 7


@dataclass(frozen=True)
class EvaluationScenario:
    name: str
    group: str
    force_waveform: str
    force_amp: float
    force_bias: float
    force_freq_rad: float
    force_phase: float
    env_switch_time: float
    pre_switch_Ke: float
    pre_switch_Be: float
    post_switch_Ke: float
    post_switch_Be: float
    initial_state_delta: tuple[float, ...] | None = None
    force_release_time: float | None = None
    force_release_value: float = 0.0
    force_chirp_end_freq_rad: float | None = None
    force_chirp_duration: float | None = None
    force_sequence_times: tuple[float, ...] | None = None
    force_sequence_values: tuple[float, ...] | None = None
    episode_duration: float | None = None
    notes: str = ""

    def reset_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "name": self.name,
            "force_waveform": self.force_waveform,
            "force_amp": float(self.force_amp),
            "force_bias": float(self.force_bias),
            "force_freq_rad": float(self.force_freq_rad),
            "force_phase": float(self.force_phase),
            "env_switch_time": float(self.env_switch_time),
            "pre_switch_Ke": float(self.pre_switch_Ke),
            "pre_switch_Be": float(self.pre_switch_Be),
            "post_switch_Ke": float(self.post_switch_Ke),
            "post_switch_Be": float(self.post_switch_Be),
        }
        if self.episode_duration is not None:
            options["episode_duration"] = float(self.episode_duration)
        if self.initial_state_delta is not None:
            options["initial_state_delta"] = list(self.initial_state_delta)
        if self.force_release_time is not None:
            options["force_release_time"] = float(self.force_release_time)
            options["force_release_value"] = float(self.force_release_value)
        if self.force_chirp_end_freq_rad is not None:
            options["force_chirp_end_freq_rad"] = float(self.force_chirp_end_freq_rad)
        if self.force_chirp_duration is not None:
            options["force_chirp_duration"] = float(self.force_chirp_duration)
        if self.force_sequence_times is not None:
            options["force_sequence_times"] = list(self.force_sequence_times)
        if self.force_sequence_values is not None:
            options["force_sequence_values"] = list(self.force_sequence_values)
        return options


def safe_stem(text: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    return stem.strip("._-") or "scenario"


def load_summary(summary_or_model_path: str | Path) -> dict[str, Any]:
    path = Path(summary_or_model_path)
    if path.is_dir():
        summary_path = path / "l" / "summary.json"
        if not summary_path.exists():
            summary_path = path / "summary.json"
    elif path.name == "summary.json":
        summary_path = path
    else:
        summary_path = path.parent.parent / "l" / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_model_path(path: str | Path, summary: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if summary and summary.get("model_path"):
        candidate = Path(str(summary["model_path"]))
        if candidate.exists():
            return candidate
    for candidate in (
        path / "m" / "ppo_model.zip",
        path / "m" / "sac_model.zip",
        path / "m" / "td3_model.zip",
        path / "ppo_model.zip",
        path / "sac_model.zip",
        path / "td3_model.zip",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find a policy-gradient model under: {path}")


def _state_variant_from_summary(summary: dict[str, Any]):
    state_spec = summary.get("state_spec")
    if isinstance(state_spec, dict):
        source_spec = state_spec.get("source_spec")
        if isinstance(source_spec, dict):
            return build_custom_dqn_state_variant_from_spec(source_spec)
        if "selected_features" in state_spec or "features" in state_spec:
            spec = dict(state_spec)
            spec.setdefault("name", summary.get("state_variant", "custom_state"))
            spec.setdefault(
                "description",
                summary.get("state_variant_description", "Summary-defined custom state."),
            )
            return build_custom_dqn_state_variant_from_spec(spec)

    try:
        return get_policy_gradient_state_variant(str(summary["state_variant"]))
    except KeyError as exc:
        raise KeyError(
            f"Unknown policy-gradient state variant {summary.get('state_variant')!r}. "
            "For notebook-defined variants, the run summary must include 'state_spec'."
        ) from exc


def _reward_variant_from_summary(summary: dict[str, Any]):
    for key in ("reward_spec", "reward_config"):
        spec = summary.get(key)
        if isinstance(spec, dict):
            return reward_variant_from_spec(spec)

    try:
        return get_policy_gradient_reward_variant(str(summary["reward_variant"]))
    except KeyError as exc:
        raise KeyError(
            f"Unknown policy-gradient reward variant {summary.get('reward_variant')!r}. "
            "For notebook-defined rewards, the run summary must include 'reward_config'."
        ) from exc


def build_eval_context(summary: dict[str, Any]):
    env_kwargs = {
        "episode_duration": float(summary["episode_duration"]),
        "env_switch_time": float(summary["env_switch_time"]),
        "terminate_on_error": bool(summary.get("terminate_on_error", True)),
        "enforce_stroke_limit": bool(summary.get("enforce_stroke_limit", True)),
        "stroke_limit_mode": str(summary.get("stroke_limit_mode", "clamp")),
        "reset_position_mode": str(summary.get("reset_options", {}).get("reset_position_mode", "midpoint")),
        "reset_options": dict(summary.get("reset_options", {})),
    }
    if summary.get("action_levels") is not None:
        env_kwargs["action_levels"] = list(summary["action_levels"])
    state_variant = _state_variant_from_summary(summary)
    reward_variant = _reward_variant_from_summary(summary)
    env_factory = build_policy_gradient_env_factory(
        algo=str(summary.get("algo", summary.get("family", PG_ALGO_PPO_CONTINUOUS))),
        env_mode=str(summary["env_mode"]),
        env_kwargs=env_kwargs,
        state_variant=state_variant,
        reward_variant=reward_variant,
    )
    return env_factory, env_kwargs, state_variant, reward_variant


def load_policy_gradient_model(model_path: str | Path, summary: dict[str, Any]):
    require_sb3()
    from stable_baselines3 import PPO, SAC, TD3

    _install_numpy_pickle_compat()
    custom_objects = _sb3_space_custom_objects(summary)
    algo = str(summary.get("algo", summary.get("family", PG_ALGO_PPO_CONTINUOUS)))
    model_path = resolve_model_path(model_path, summary)
    if algo == PG_ALGO_PPO_CONTINUOUS:
        return PPO.load(str(model_path), custom_objects=custom_objects)
    if algo == PG_ALGO_SAC:
        return SAC.load(str(model_path), custom_objects=custom_objects)
    if algo == PG_ALGO_TD3:
        return TD3.load(str(model_path), custom_objects=custom_objects)
    raise ValueError(f"Focused continuous evaluation does not support algo: {algo}")


def _sb3_space_custom_objects(summary: dict[str, Any]) -> dict[str, Any]:
    try:
        from gymnasium import spaces
    except Exception:
        return {}
    obs_dim = int(summary.get("obs_dim", 0) or 0)
    if obs_dim <= 0:
        return {}
    action_levels = np.asarray(summary.get("action_levels", cfg.V_LEVELS), dtype=np.float32).reshape(-1)
    action_limit = float(np.max(np.abs(action_levels))) if action_levels.size else 5.0
    return {
        "observation_space": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        ),
        "action_space": spaces.Box(
            low=np.asarray([-action_limit], dtype=np.float32),
            high=np.asarray([action_limit], dtype=np.float32),
            dtype=np.float32,
        ),
    }


def _install_numpy_pickle_compat() -> None:
    """Allow SB3 checkpoints pickled under NumPy 2.x to load under NumPy 1.x."""
    try:
        import numpy.core as np_core
        import numpy.core.multiarray as np_multiarray
        import numpy.core.numeric as np_numeric
        import numpy.random._pickle as np_random_pickle
    except Exception:
        return
    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)
    original_ctor = getattr(np_random_pickle, "__bit_generator_ctor", None)
    bit_generators = getattr(np_random_pickle, "BitGenerators", {})
    if original_ctor is None or getattr(original_ctor, "_teleop_compat", False):
        return

    def _bit_generator_ctor_compat(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        if bit_generator_name in bit_generators:
            return bit_generators[bit_generator_name]()
        return original_ctor(bit_generator_name)

    _bit_generator_ctor_compat._teleop_compat = True  # type: ignore[attr-defined]
    np_random_pickle.__bit_generator_ctor = _bit_generator_ctor_compat


def _nominal_values(summary: dict[str, Any]) -> dict[str, float | str]:
    reset_options = dict(summary.get("reset_options", {}))
    profile = SimuOriginalProfile(env_switch_time=float(summary.get("env_switch_time", cfg.ENV_SWITCH_TIME)))
    pre_ke = float(reset_options.get("pre_switch_Ke", profile.skin_Ke))
    pre_be = float(reset_options.get("pre_switch_Be", profile.skin_Be))
    post_ke = float(reset_options.get("post_switch_Ke", reset_options.get("K_e", pre_ke + profile.delta_Ke_after_switch)))
    post_be = float(reset_options.get("post_switch_Be", reset_options.get("B_e", pre_be + profile.delta_Be_after_switch)))
    force_freq_rad = float(
        reset_options.get(
            "force_freq_rad",
            _TWO_PI * float(reset_options.get("force_freq", cfg.FORCE_INPUT_FREQ)),
        )
    )
    return {
        "force_waveform": str(reset_options.get("force_waveform", "sine")),
        "force_amp": float(reset_options.get("force_amp", cfg.FORCE_INPUT_AMP)),
        "force_bias": float(reset_options.get("force_bias", 0.0)),
        "force_freq_rad": force_freq_rad,
        "force_phase": float(reset_options.get("force_phase", cfg.FORCE_INPUT_PHASE)),
        "env_switch_time": float(summary.get("env_switch_time", cfg.ENV_SWITCH_TIME)),
        "episode_duration": float(summary.get("episode_duration", cfg.EPISODE_DURATION)),
        "pre_switch_Ke": pre_ke,
        "pre_switch_Be": pre_be,
        "post_switch_Ke": post_ke,
        "post_switch_Be": post_be,
    }


_TWO_PI = 2.0 * math.pi


def _scenario(base: dict[str, float | str], *, name: str, group: str, **overrides: Any) -> EvaluationScenario:
    payload = dict(base)
    payload.update(overrides)
    return EvaluationScenario(
        name=name,
        group=group,
        force_waveform=str(payload["force_waveform"]),
        force_amp=float(payload["force_amp"]),
        force_bias=float(payload["force_bias"]),
        force_freq_rad=float(payload["force_freq_rad"]),
        force_phase=float(payload["force_phase"]),
        env_switch_time=float(payload["env_switch_time"]),
        pre_switch_Ke=float(payload["pre_switch_Ke"]),
        pre_switch_Be=float(payload["pre_switch_Be"]),
        post_switch_Ke=float(payload["post_switch_Ke"]),
        post_switch_Be=float(payload["post_switch_Be"]),
        initial_state_delta=payload.get("initial_state_delta"),
        force_release_time=payload.get("force_release_time"),
        force_release_value=float(payload.get("force_release_value", 0.0)),
        force_chirp_end_freq_rad=payload.get("force_chirp_end_freq_rad"),
        force_chirp_duration=payload.get("force_chirp_duration"),
        force_sequence_times=_tuple_or_none(payload.get("force_sequence_times")),
        force_sequence_values=_tuple_or_none(payload.get("force_sequence_values")),
        episode_duration=payload.get("episode_duration"),
        notes=str(payload.get("notes", "")),
    )


def _delta(**entries: float) -> tuple[float, ...]:
    values = np.zeros(12, dtype=np.float64)
    for key, value in entries.items():
        values[int(key)] = float(value)
    return tuple(float(v) for v in values)


def _tuple_or_none(values: Any) -> tuple[float, ...] | None:
    if values is None:
        return None
    return tuple(float(v) for v in values)


def build_focused_scenarios(summary: dict[str, Any]) -> list[EvaluationScenario]:
    base = _nominal_values(summary)
    k0 = float(base["post_switch_Ke"])
    b0 = float(base["post_switch_Be"])
    contact_t = float(base["env_switch_time"])
    duration = float(base["episode_duration"])
    lit_amp = 10.0
    lit_omega = math.pi
    lit_base = {
        **base,
        "force_waveform": "sine",
        "force_amp": lit_amp,
        "force_bias": 0.0,
        "force_freq_rad": lit_omega,
        "force_phase": 0.0,
    }

    scenarios: list[EvaluationScenario] = [
        _scenario(
            lit_base,
            name="nominal",
            group="nominal",
            notes="Baayoun-style 10 N sine at 0.5 Hz / pi rad-s, zero bias.",
        ),
    ]

    for label, amp in (("1N", 1.0), ("10N", 10.0), ("20N", 20.0)):
        scenarios.append(
            _scenario(
                lit_base,
                name=f"force_amp_{label}",
                group="paper_force_amplitude",
                force_amp=amp,
                notes="Literature force-amplitude set: 1 N, 10 N, 20 N.",
            )
        )

    for label, omega in (("1_rad_s", 1.0), ("pi_rad_s", math.pi), ("5_rad_s", 5.0), ("10_rad_s", 10.0)):
        scenarios.append(
            _scenario(
                lit_base,
                name=f"force_freq_{label}",
                group="paper_force_frequency",
                force_freq_rad=omega,
                notes="Literature frequency set: 1, pi, 5, 10 rad/s.",
            )
        )

    scenarios.extend(
        [
            _scenario(
                lit_base,
                name="signal_sine_10N_pi",
                group="paper_signal_type",
                force_waveform="sine",
                notes="10 N sinusoidal force at pi rad/s.",
            ),
            _scenario(
                lit_base,
                name="signal_pulse_10N_pi",
                group="paper_signal_type",
                force_waveform="square",
                notes="Pulse-like operator input, implemented as a square-like force.",
            ),
            _scenario(
                lit_base,
                name="signal_constant_0p5N",
                group="paper_signal_type",
                force_waveform="constant",
                force_amp=0.0,
                force_bias=0.5,
                notes="Constant 0.5 N forward force used in object simulations.",
            ),
            _scenario(
                lit_base,
                name="signal_sequence_1_10_20N",
                group="paper_signal_type",
                force_waveform="sequence",
                force_amp=0.0,
                force_bias=0.0,
                force_sequence_times=(0.0, duration / 3.0, 2.0 * duration / 3.0),
                force_sequence_values=(1.0, 10.0, 20.0),
                notes="Changing-force sequence inspired by 1 N, 10 N, 20 N tests.",
            ),
            _scenario(
                lit_base,
                name="signal_ramp_generalization",
                group="engineering_signal_generalization",
                force_waveform="ramp",
                notes="Engineering generalization test; not Baayoun-derived.",
            ),
            _scenario(
                lit_base,
                name="signal_chirp_generalization",
                group="engineering_signal_generalization",
                force_waveform="chirp",
                force_freq_rad=1.0,
                force_chirp_end_freq_rad=10.0,
                force_chirp_duration=duration,
                notes="Engineering frequency-sweep generalization test; not Baayoun-derived.",
            ),
        ]
    )

    for label, factor in (("low", 0.5), ("nominal", 1.0), ("high", 1.5)):
        scenarios.append(
            _scenario(
                lit_base,
                name=f"env_K_{label}",
                group="environment_stiffness",
                post_switch_Ke=k0 * factor,
                post_switch_Be=b0,
                notes="One-factor SI-unit stiffness variation around fat-like Baayoun value.",
            )
        )

    for label, factor in (("low", 0.5), ("nominal", 1.0), ("high", 2.0)):
        scenarios.append(
            _scenario(
                lit_base,
                name=f"env_B_{label}",
                group="environment_damping",
                post_switch_Ke=k0,
                post_switch_Be=b0 * factor,
                notes="One-factor SI-unit damping variation around fat-like Baayoun value.",
            )
        )

    scenarios.extend(
        [
            _scenario(
                lit_base,
                name="init_error_pos_10mm",
                group="initial_condition",
                initial_state_delta=_delta(**{str(_REPLICA_XM): 0.005, str(_REPLICA_XS): -0.005}),
            ),
            _scenario(
                lit_base,
                name="init_error_neg_10mm",
                group="initial_condition",
                initial_state_delta=_delta(**{str(_REPLICA_XM): -0.005, str(_REPLICA_XS): 0.005}),
            ),
            _scenario(
                lit_base,
                name="init_master_vel_pos",
                group="initial_condition",
                initial_state_delta=_delta(**{str(_REPLICA_XM_DOT): 0.05}),
            ),
            _scenario(
                lit_base,
                name="init_slave_vel_neg",
                group="initial_condition",
                initial_state_delta=_delta(**{str(_REPLICA_XS_DOT): -0.05}),
            ),
        ]
    )

    release_time = min(duration - cfg.RL_DT, max(contact_t + 5.0, 0.5 * duration))
    scenarios.append(
        _scenario(
            lit_base,
            name=f"sudden_release_constant_10N_t{release_time:g}s".replace(".", "p"),
            group="paper_sudden_release",
            force_waveform="constant",
            force_amp=0.0,
            force_bias=10.0,
            force_release_time=release_time,
            force_release_value=0.0,
            notes="Constant forward force followed by sudden release.",
        )
    )
    return scenarios


def build_bode_scenarios(summary: dict[str, Any], frequencies_rad_s: list[float] | None = None) -> list[EvaluationScenario]:
    base = _nominal_values(summary)
    base = {
        **base,
        "force_amp": 10.0,
        "force_bias": 0.0,
        "force_phase": 0.0,
    }
    frequencies = frequencies_rad_s or [0.1, 0.4, 0.5, 1.0, math.pi, 5.0, 10.0, 12.0]
    contact_t = float(base["env_switch_time"])
    base_duration = float(base["episode_duration"])
    return [
        _scenario(
            base,
            name=f"bode_{omega:g}_rad_s".replace(".", "p"),
            group="empirical_bode",
            force_waveform="sine",
            force_freq_rad=float(omega),
            force_phase=0.0,
            episode_duration=max(base_duration, contact_t + (4.0 * (2.0 * math.pi / max(float(omega), 1e-9)))),
            notes="Paper-matched Bode frequency set; 12 rad/s is retained as extrapolation.",
        )
        for omega in frequencies
    ]


def evaluate_policy_on_scenario(
    policy: Any,
    env_factory: Callable[[], Any],
    scenario: EvaluationScenario,
    *,
    seed: int,
    deterministic: bool = True,
) -> dict[str, Any]:
    env = env_factory()
    obs, info = env.reset(seed=int(seed), options=scenario.reset_options())
    if hasattr(policy, "reset_recurrent_state"):
        policy.reset_recurrent_state()
    done = False
    final_info = dict(info)
    final_terminated = False
    final_truncated = False
    while not done:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        final_info = dict(info)
        final_terminated = bool(terminated)
        final_truncated = bool(truncated)
    history = env.render() or {}
    if hasattr(env, "close"):
        env.close()
    return {
        "scenario": scenario,
        "history": history,
        "final_info": final_info,
        "terminated": final_terminated,
        "truncated": final_truncated,
    }


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2))) if values.size else 0.0


def _peak_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _integral(t: np.ndarray, values: np.ndarray) -> float:
    if t.size < 2 or values.size < 2:
        return 0.0
    n = min(t.size, values.size)
    return float(np.trapz(values[:n], t[:n]))


def settling_time(t: np.ndarray, error: np.ndarray, *, start_time: float, threshold: float, window_s: float) -> float:
    if t.size == 0 or error.size == 0:
        return float("nan")
    n = min(t.size, error.size)
    t = t[:n]
    error = np.abs(error[:n])
    indices = np.flatnonzero(t >= float(start_time))
    if indices.size == 0:
        return float("nan")
    if t.size >= 2:
        dt = float(np.nanmedian(np.diff(t)))
    else:
        dt = float(cfg.RL_DT)
    window = max(1, int(round(float(window_s) / max(dt, 1e-9))))
    for idx in indices:
        end = min(n, idx + window)
        if end <= idx:
            continue
        if np.all(error[idx:end] <= float(threshold)) and end - idx >= min(window, n - idx):
            return float(t[idx] - float(start_time))
    return float("nan")


def compute_non_bode_metrics(result: dict[str, Any], *, action_limit: float = 5.0) -> dict[str, Any]:
    scenario: EvaluationScenario = result["scenario"]
    history = result["history"]
    final_info = result["final_info"]
    t = history_array(history, "time", dtype=np.float64)
    error = history_array(history, "pos_error", dtype=np.float64)
    u_v = history_array(history, "u_v", dtype=np.float64)
    reward = history_array(history, "reward", dtype=np.float64)
    transparency_power_error = transparency_power_error_array(history)
    transparency_ratio = transparency_ratio_array(history)
    transparency_ratio_error = transparency_ratio - 1.0
    ratio_metrics = transparency_ratio_metrics(history)
    n = min(t.size, error.size)
    t = t[:n]
    error = error[:n]
    u_v = u_v[: min(u_v.size, n)]
    post = t >= float(scenario.env_switch_time) if t.size else np.zeros(0, dtype=bool)
    du = np.diff(u_v) if u_v.size >= 2 else np.asarray([], dtype=np.float64)
    ddu = np.diff(u_v, n=2) if u_v.size >= 3 else np.asarray([], dtype=np.float64)
    termination_reason = str(final_info.get("termination_reason", ""))
    failure = bool(
        result["terminated"]
        or final_info.get("invalid_state")
        or final_info.get("tracking_error_fail")
        or termination_reason in {"tracking_error_fail", "stroke_limit", "volume_singularity"}
    )
    return {
        "scenario": scenario.name,
        "group": scenario.group,
        "notes": scenario.notes,
        "force_waveform": scenario.force_waveform,
        "force_amp_N": float(scenario.force_amp),
        "force_bias_N": float(scenario.force_bias),
        "force_freq_rad_s": float(scenario.force_freq_rad),
        "K_e": float(scenario.post_switch_Ke),
        "B_e": float(scenario.post_switch_Be),
        "episode_duration_s": "" if scenario.episode_duration is None else float(scenario.episode_duration),
        "initial_condition_changed": int(scenario.initial_state_delta is not None),
        "release_time_s": "" if scenario.force_release_time is None else float(scenario.force_release_time),
        "episode_return": float(np.sum(reward)) if reward.size else 0.0,
        "rms_error_m": _rms(error),
        "peak_error_m": _peak_abs(error),
        "post_contact_rms_error_m": _rms(error[post]) if post.size == error.size else 0.0,
        "post_contact_peak_error_m": _peak_abs(error[post]) if post.size == error.size else 0.0,
        "transparency_rmse_w": _rms(transparency_power_error),
        "transparency_ratio_raw_mean": float(np.mean(transparency_ratio)) if transparency_ratio.size else 0.0,
        "transparency_ratio_mean": float(ratio_metrics["transparency_ratio_mean"]),
        "transparency_ratio_median": float(ratio_metrics["transparency_ratio_median"]),
        "transparency_ratio_raw_error_rmse": _rms(transparency_ratio_error),
        "transparency_ratio_error_rmse": float(ratio_metrics["transparency_ratio_error_rmse"]),
        "transparency_ratio_valid_fraction": float(ratio_metrics["transparency_ratio_valid_fraction"]),
        "transparency_ratio_within_20pct": float(ratio_metrics["transparency_ratio_within_20pct"]),
        "settling_time_s": settling_time(
            t,
            error,
            start_time=float(scenario.env_switch_time),
            threshold=0.005,
            window_s=1.0,
        ),
        "control_energy_v2_s": _integral(t[: u_v.size], u_v ** 2),
        "mean_abs_u_v": float(np.mean(np.abs(u_v))) if u_v.size else 0.0,
        "rms_u_v": _rms(u_v),
        "control_smoothness_mean_abs_delta_v": float(np.mean(np.abs(du))) if du.size else 0.0,
        "control_smoothness_rms_delta_v": _rms(du),
        "control_smoothness_mean_abs_delta2_v": float(np.mean(np.abs(ddu))) if ddu.size else 0.0,
        "control_smoothness_rms_delta2_v": _rms(ddu),
        "max_abs_u_v": _peak_abs(u_v),
        "max_abs_delta_u_v": _peak_abs(du),
        "max_abs_delta2_u_v": _peak_abs(ddu),
        "saturation_fraction": float(np.mean(np.abs(u_v) >= 0.98 * float(action_limit))) if u_v.size else 0.0,
        "failure_flag": int(failure),
        "termination_reason": termination_reason,
    }


def _fit_sine(t: np.ndarray, y: np.ndarray, omega: float) -> tuple[float, float]:
    if t.size < 4 or y.size < 4:
        return 0.0, 0.0
    n = min(t.size, y.size)
    t = t[:n]
    y = y[:n]
    x = np.column_stack([np.sin(float(omega) * t), np.cos(float(omega) * t), np.ones_like(t)])
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    sin_c, cos_c = float(coeffs[0]), float(coeffs[1])
    amp = math.sqrt((sin_c ** 2) + (cos_c ** 2))
    phase = math.atan2(cos_c, sin_c)
    return amp, phase


def _wrap_phase(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def compute_bode_metrics(result: dict[str, Any]) -> dict[str, Any]:
    scenario: EvaluationScenario = result["scenario"]
    history = result["history"]
    t = history_array(history, "time", dtype=np.float64)
    x_m = history_array(history, "x_m", dtype=np.float64)
    x_s = history_array(history, "x_s", dtype=np.float64)
    transient_s = max(float(scenario.env_switch_time), 2.0 * (2.0 * math.pi / max(float(scenario.force_freq_rad), 1e-9)))
    mask = t >= transient_s if t.size else np.zeros(0, dtype=bool)
    if mask.size == t.size and np.any(mask):
        t_fit = t[mask]
        x_m_fit = x_m[: t.size][mask]
        x_s_fit = x_s[: t.size][mask]
    else:
        n = min(t.size, x_m.size, x_s.size)
        t_fit = t[:n]
        x_m_fit = x_m[:n]
        x_s_fit = x_s[:n]
    amp_m, phase_m = _fit_sine(t_fit, x_m_fit, float(scenario.force_freq_rad))
    amp_s, phase_s = _fit_sine(t_fit, x_s_fit, float(scenario.force_freq_rad))
    gain = float(amp_s / max(amp_m, 1e-12))
    return {
        "scenario": scenario.name,
        "frequency_rad_s": float(scenario.force_freq_rad),
        "frequency_hz": float(scenario.force_freq_rad / (2.0 * math.pi)),
        "gain": gain,
        "gain_dB": float(20.0 * math.log10(max(gain, 1e-12))),
        "phase_lag_rad": _wrap_phase(phase_s - phase_m),
        "phase_lag_deg": float(math.degrees(_wrap_phase(phase_s - phase_m))),
        "master_amp_m": float(amp_m),
        "slave_amp_m": float(amp_s),
    }


def plot_scenario_result(result: dict[str, Any], out_dir: str | Path) -> None:
    scenario: EvaluationScenario = result["scenario"]
    history = result["history"]
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    t = history_array(history, "time", dtype=np.float64)
    x_m = history_array(history, "x_m", dtype=np.float64)
    x_s = history_array(history, "x_s", dtype=np.float64)
    error = history_array(history, "pos_error", dtype=np.float64)
    u_v = history_array(history, "u_v", dtype=np.float64)
    f_h = history_array(history, "F_h", dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    axes[0].plot(t[: x_m.size], x_m * 1000.0, label="x_m")
    axes[0].plot(t[: x_s.size], x_s * 1000.0, label="x_s")
    axes[0].set_ylabel("position [mm]")
    axes[0].legend(loc="best")
    axes[1].plot(t[: error.size], error * 1000.0, color="tab:red")
    axes[1].set_ylabel("error [mm]")
    axes[2].plot(t[: u_v.size], u_v, color="tab:cyan")
    axes[2].set_ylabel("u_v [V]")
    axes[3].plot(t[: f_h.size], f_h, color="tab:blue")
    axes[3].set_ylabel("F_h [N]")
    axes[3].set_xlabel("time [s]")
    for ax in axes:
        ax.axvline(float(scenario.env_switch_time), color="0.4", ls="--", lw=1.0)
        if scenario.force_release_time is not None:
            ax.axvline(float(scenario.force_release_time), color="tab:orange", ls=":", lw=1.2)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{scenario.group}: {scenario.name}")
    stem = safe_stem(scenario.name)
    fig.savefig(_save_path(out_dir / f"{stem}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_transparency_ratio_monitor(
        history,
        out_dir / f"{stem}_transparency_ratio.png",
        f"{scenario.group}: {scenario.name}",
        env_switch_time=float(scenario.env_switch_time),
    )


def _plot_tracking_result(result: dict[str, Any], out_dir: str | Path) -> None:
    scenario: EvaluationScenario = result["scenario"]
    history = result["history"]
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    t = history_array(history, "time", dtype=np.float64)
    x_m = history_array(history, "x_m", dtype=np.float64)
    x_s = history_array(history, "x_s", dtype=np.float64)
    error = history_array(history, "pos_error", dtype=np.float64)
    stem = safe_stem(scenario.name)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(t[: x_m.size], x_m * 1000.0, label="master")
    axes[0].plot(t[: x_s.size], x_s * 1000.0, label="slave")
    axes[0].set_ylabel("position [mm]")
    axes[0].legend(loc="best")
    axes[1].plot(t[: error.size], error * 1000.0, color="tab:red")
    axes[1].axhline(0.0, color="0.4", lw=1.0)
    axes[1].set_ylabel("x_m - x_s [mm]")
    axes[1].set_xlabel("time [s]")
    for ax in axes:
        ax.axvline(float(scenario.env_switch_time), color="0.4", ls="--", lw=1.0)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{scenario.group}: {scenario.name}: tracking")
    fig.savefig(_save_path(out_dir / f"{stem}_tracking.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_force_result(result: dict[str, Any], out_dir: str | Path) -> None:
    scenario: EvaluationScenario = result["scenario"]
    history = result["history"]
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    t = history_array(history, "time", dtype=np.float64)
    f_h = history_array(history, "F_h", dtype=np.float64)
    f_env = history_array(history, "F_env", dtype=np.float64)
    stem = safe_stem(scenario.name)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(t[: f_h.size], f_h, label="F_h")
    if f_env.size:
        ax.plot(t[: f_env.size], f_env, label="F_env")
    ax.axvline(float(scenario.env_switch_time), color="0.4", ls="--", lw=1.0)
    if scenario.force_release_time is not None:
        ax.axvline(float(scenario.force_release_time), color="tab:orange", ls=":", lw=1.2)
    ax.set_ylabel("force [N]")
    ax.set_xlabel("time [s]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(f"{scenario.group}: {scenario.name}: force")
    fig.savefig(_save_path(out_dir / f"{stem}_force.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_control_result(result: dict[str, Any], out_dir: str | Path) -> None:
    scenario: EvaluationScenario = result["scenario"]
    history = result["history"]
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    t = history_array(history, "time", dtype=np.float64)
    u_v = history_array(history, "u_v", dtype=np.float64)
    du = np.diff(u_v) if u_v.size else np.asarray([], dtype=np.float64)
    stem = safe_stem(scenario.name)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(t[: u_v.size], u_v, color="tab:cyan")
    axes[0].set_ylabel("u_v [V]")
    axes[1].plot(t[1 : 1 + du.size], du, color="tab:purple")
    axes[1].axhline(0.0, color="0.4", lw=1.0)
    axes[1].set_ylabel("delta u_v [V]")
    axes[1].set_xlabel("time [s]")
    for ax in axes:
        ax.axvline(float(scenario.env_switch_time), color="0.4", ls="--", lw=1.0)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{scenario.group}: {scenario.name}: control")
    fig.savefig(_save_path(out_dir / f"{stem}_control.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_detail_results(result: dict[str, Any], out_dir: str | Path) -> None:
    _plot_tracking_result(result, out_dir)
    _plot_force_result(result, out_dir)
    _plot_control_result(result, out_dir)


def _npz_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (str, bytes, int, float, bool, np.number)):
        return np.asarray(value)
    try:
        arr = np.asarray(value)
    except Exception:
        return np.asarray(json.dumps(value, default=str))
    if arr.dtype == object:
        try:
            return arr.astype(np.float64)
        except (TypeError, ValueError):
            return np.asarray(json.dumps(value, default=str))
    return arr


def save_scenario_history_npz(
    result: dict[str, Any],
    out_dir: str | Path,
    metrics: dict[str, Any] | None = None,
) -> Path:
    scenario: EvaluationScenario = result["scenario"]
    history = result["history"]
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    stem = safe_stem(scenario.name)
    payload: dict[str, np.ndarray] = {
        "scenario_json": np.asarray(json.dumps(asdict(scenario), default=str)),
    }
    if metrics is not None:
        payload["metrics_json"] = np.asarray(json.dumps(metrics, default=str))
    for key, value in history.items():
        payload[str(key)] = _npz_array(value)
    out_path = out_dir / f"{stem}.npz"
    np.savez_compressed(_save_path(out_path), **payload)
    return out_path


def plot_bode(bode_rows: list[dict[str, Any]], out_path: str | Path) -> None:
    if not bode_rows:
        return
    rows = sorted(bode_rows, key=lambda row: float(row["frequency_rad_s"]))
    omega = np.asarray([row["frequency_rad_s"] for row in rows], dtype=np.float64)
    gain_db = np.asarray([row["gain_dB"] for row in rows], dtype=np.float64)
    phase_deg = np.asarray([row["phase_lag_deg"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].semilogx(omega, gain_db, marker="o")
    axes[0].set_ylabel("gain [dB]")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].semilogx(omega, phase_deg, marker="o", color="tab:orange")
    axes[1].set_ylabel("phase lag [deg]")
    axes[1].set_xlabel("frequency [rad/s]")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.savefig(_save_path(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(_save_path(path), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_path(path: str | Path) -> str | Path:
    path = Path(path)
    if os.name != "nt":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    resolved = path.resolve()
    parent = str(path.parent.resolve())
    if parent.startswith("\\\\?\\"):
        parent_text = parent
    elif parent.startswith("\\\\"):
        parent_text = "\\\\?\\UNC\\" + parent.lstrip("\\")
    else:
        parent_text = "\\\\?\\" + parent
    os.makedirs(parent_text, exist_ok=True)
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text.lstrip("\\")
    return "\\\\?\\" + text


def _ensure_dir(path: str | Path) -> None:
    path = Path(path)
    if os.name != "nt":
        path.mkdir(parents=True, exist_ok=True)
        return
    text = str(path.resolve())
    if text.startswith("\\\\?\\"):
        dir_text = text
    elif text.startswith("\\\\"):
        dir_text = "\\\\?\\UNC\\" + text.lstrip("\\")
    else:
        dir_text = "\\\\?\\" + text
    os.makedirs(dir_text, exist_ok=True)


def run_focused_evaluation(
    *,
    model_path: str | Path,
    out_dir: str | Path,
    seed: int = 42,
    deterministic: bool = True,
    include_bode: bool = True,
    save_plots: bool = True,
) -> dict[str, Any]:
    summary = load_summary(model_path)
    model_path = resolve_model_path(model_path, summary)
    policy = load_policy_gradient_model(model_path, summary)
    env_factory, env_kwargs, state_variant, reward_variant = build_eval_context(summary)
    action_levels = np.asarray(summary.get("action_levels", cfg.V_LEVELS), dtype=np.float64)
    action_limit = float(np.max(np.abs(action_levels))) if action_levels.size else 5.0
    out_dir = Path(out_dir)
    scenarios = build_focused_scenarios(summary)
    normal_rows: list[dict[str, Any]] = []
    normal_results: list[dict[str, Any]] = []
    for idx, scenario in enumerate(scenarios):
        result = evaluate_policy_on_scenario(
            policy,
            env_factory,
            scenario,
            seed=int(seed) + idx,
            deterministic=deterministic,
        )
        normal_results.append(result)
        metrics = compute_non_bode_metrics(result, action_limit=action_limit)
        normal_rows.append(metrics)
        save_scenario_history_npz(result, out_dir / "histories", metrics)
        if save_plots:
            plot_scenario_result(result, out_dir / "plots" / "scenarios")
            plot_scenario_detail_results(result, out_dir / "plots" / "scenarios")

    bode_rows: list[dict[str, Any]] = []
    if include_bode:
        bode_scenarios = build_bode_scenarios(summary)
        for idx, scenario in enumerate(bode_scenarios):
            result = evaluate_policy_on_scenario(
                policy,
                env_factory,
                scenario,
                seed=int(seed) + 10_000 + idx,
                deterministic=deterministic,
            )
            bode_rows.append(compute_bode_metrics(result))
        if save_plots:
            plot_bode(bode_rows, out_dir / "plots" / "empirical_bode.png")

    groups = sorted({row["group"] for row in normal_rows})
    summary_payload = {
        "model_path": str(model_path),
        "source_summary": str(Path(model_path).parent.parent / "l" / "summary.json"),
        "algo": str(summary.get("algo", summary.get("family", ""))),
        "state_variant": state_variant.name,
        "reward_variant": reward_variant.name,
        "deterministic": bool(deterministic),
        "seed": int(seed),
        "methodology": "paper_matched_eval_v1",
        "groups": groups,
        "metrics": [
            "rms_error_m",
            "peak_error_m",
            "post_contact_rms_error_m",
            "post_contact_peak_error_m",
            "settling_time_s",
            "control_energy_v2_s",
            "control_smoothness_mean_abs_delta_v",
            "control_smoothness_mean_abs_delta2_v",
            "transparency_ratio_median",
            "transparency_ratio_valid_fraction",
            "transparency_ratio_within_20pct",
            "saturation_fraction",
            "failure_flag",
        ],
        "bode_metrics": ["frequency_rad_s", "gain", "gain_dB", "phase_lag_deg"],
        "artifacts": {
            "metrics_csv": "focused_eval_metrics.csv",
            "history_npz_dir": "histories",
            "scenario_plots_dir": "plots/scenarios",
            "scenario_dashboard_pattern": "<scenario>.png",
            "tracking_plot_pattern": "<scenario>_tracking.png",
            "force_plot_pattern": "<scenario>_force.png",
            "control_plot_pattern": "<scenario>_control.png",
            "transparency_ratio_plot_pattern": "<scenario>_transparency_ratio.png",
        },
        "env_kwargs": env_kwargs,
        "scenarios": [asdict(scenario) for scenario in scenarios],
    }
    _ensure_dir(out_dir)
    write_csv(out_dir / "focused_eval_metrics.csv", normal_rows)
    save_json(out_dir / "focused_eval_summary.json", summary_payload)
    if bode_rows:
        write_csv(out_dir / "focused_eval_bode.csv", bode_rows)
    return {
        "summary": summary_payload,
        "metrics": normal_rows,
        "bode": bode_rows,
    }
