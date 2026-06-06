from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ... import config as cfg
from ..simuoriginal_replica import SimuOriginalProfile
from .common import history_array, save_json
from .policy_gradient import (
    PG_ALGO_PPO_CONTINUOUS,
    PG_ALGO_SAC,
    PG_ALGO_TD3,
    build_policy_gradient_env_factory,
    get_policy_gradient_reward_variant,
    get_policy_gradient_state_variant,
    require_sb3,
)


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
        if self.initial_state_delta is not None:
            options["initial_state_delta"] = list(self.initial_state_delta)
        if self.force_release_time is not None:
            options["force_release_time"] = float(self.force_release_time)
            options["force_release_value"] = float(self.force_release_value)
        if self.force_chirp_end_freq_rad is not None:
            options["force_chirp_end_freq_rad"] = float(self.force_chirp_end_freq_rad)
        if self.force_chirp_duration is not None:
            options["force_chirp_duration"] = float(self.force_chirp_duration)
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
    state_variant = get_policy_gradient_state_variant(str(summary["state_variant"]))
    reward_variant = get_policy_gradient_reward_variant(str(summary["reward_variant"]))
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

    algo = str(summary.get("algo", summary.get("family", PG_ALGO_PPO_CONTINUOUS)))
    model_path = resolve_model_path(model_path, summary)
    if algo == PG_ALGO_PPO_CONTINUOUS:
        return PPO.load(str(model_path))
    if algo == PG_ALGO_SAC:
        return SAC.load(str(model_path))
    if algo == PG_ALGO_TD3:
        return TD3.load(str(model_path))
    raise ValueError(f"Focused continuous evaluation does not support algo: {algo}")


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
        notes=str(payload.get("notes", "")),
    )


def _delta(**entries: float) -> tuple[float, ...]:
    values = np.zeros(12, dtype=np.float64)
    for key, value in entries.items():
        values[int(key)] = float(value)
    return tuple(float(v) for v in values)


def build_focused_scenarios(summary: dict[str, Any]) -> list[EvaluationScenario]:
    base = _nominal_values(summary)
    a0 = float(base["force_amp"])
    w0 = float(base["force_freq_rad"])
    k0 = float(base["post_switch_Ke"])
    b0 = float(base["post_switch_Be"])
    contact_t = float(base["env_switch_time"])
    duration = float(base["episode_duration"])

    scenarios: list[EvaluationScenario] = [
        _scenario(base, name="nominal", group="nominal"),
    ]

    for label, factor in (("low", 0.6), ("nominal", 1.0), ("high", 1.4)):
        scenarios.append(
            _scenario(base, name=f"force_amp_{label}", group="human_force_amplitude", force_amp=a0 * factor)
        )

    for label, factor in (("low", 0.5), ("nominal", 1.0), ("high", 1.5)):
        scenarios.append(
            _scenario(base, name=f"force_freq_{label}", group="human_force_frequency", force_freq_rad=w0 * factor)
        )

    scenarios.extend(
        [
            _scenario(base, name="signal_sine", group="human_force_signal_type", force_waveform="sine"),
            _scenario(base, name="signal_pulse", group="human_force_signal_type", force_waveform="square"),
            _scenario(base, name="signal_ramp", group="human_force_signal_type", force_waveform="ramp"),
            _scenario(
                base,
                name="signal_chirp",
                group="human_force_signal_type",
                force_waveform="chirp",
                force_freq_rad=0.5 * w0,
                force_chirp_end_freq_rad=1.5 * w0,
                force_chirp_duration=duration,
            ),
        ]
    )

    for label, factor in (("low", 0.5), ("nominal", 1.0), ("high", 1.5)):
        scenarios.append(
            _scenario(
                base,
                name=f"env_K_{label}",
                group="environment_stiffness",
                post_switch_Ke=k0 * factor,
                post_switch_Be=b0,
            )
        )

    for label, factor in (("low", 0.5), ("nominal", 1.0), ("high", 2.0)):
        scenarios.append(
            _scenario(
                base,
                name=f"env_B_{label}",
                group="environment_damping",
                post_switch_Ke=k0,
                post_switch_Be=b0 * factor,
            )
        )

    scenarios.extend(
        [
            _scenario(
                base,
                name="init_error_pos_10mm",
                group="initial_condition",
                initial_state_delta=_delta(**{str(_REPLICA_XM): 0.005, str(_REPLICA_XS): -0.005}),
            ),
            _scenario(
                base,
                name="init_error_neg_10mm",
                group="initial_condition",
                initial_state_delta=_delta(**{str(_REPLICA_XM): -0.005, str(_REPLICA_XS): 0.005}),
            ),
            _scenario(
                base,
                name="init_master_vel_pos",
                group="initial_condition",
                initial_state_delta=_delta(**{str(_REPLICA_XM_DOT): 0.05}),
            ),
            _scenario(
                base,
                name="init_slave_vel_neg",
                group="initial_condition",
                initial_state_delta=_delta(**{str(_REPLICA_XS_DOT): -0.05}),
            ),
        ]
    )

    release_time = min(duration - cfg.RL_DT, max(contact_t + 5.0, 0.5 * duration))
    scenarios.append(
        _scenario(
            base,
            name=f"sudden_release_t{release_time:g}s".replace(".", "p"),
            group="sudden_release",
            force_release_time=release_time,
            force_release_value=0.0,
        )
    )
    return scenarios


def build_bode_scenarios(summary: dict[str, Any], frequencies_rad_s: list[float] | None = None) -> list[EvaluationScenario]:
    base = _nominal_values(summary)
    frequencies = frequencies_rad_s or [1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
    return [
        _scenario(
            base,
            name=f"bode_{omega:g}_rad_s".replace(".", "p"),
            group="empirical_bode",
            force_waveform="sine",
            force_freq_rad=float(omega),
            force_phase=0.0,
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
    n = min(t.size, error.size)
    t = t[:n]
    error = error[:n]
    u_v = u_v[: min(u_v.size, n)]
    post = t >= float(scenario.env_switch_time) if t.size else np.zeros(0, dtype=bool)
    du = np.diff(u_v) if u_v.size >= 2 else np.asarray([], dtype=np.float64)
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
        "initial_condition_changed": int(scenario.initial_state_delta is not None),
        "release_time_s": "" if scenario.force_release_time is None else float(scenario.force_release_time),
        "episode_return": float(np.sum(reward)) if reward.size else 0.0,
        "rms_error_m": _rms(error),
        "peak_error_m": _peak_abs(error),
        "post_contact_rms_error_m": _rms(error[post]) if post.size == error.size else 0.0,
        "post_contact_peak_error_m": _peak_abs(error[post]) if post.size == error.size else 0.0,
        "settling_time_s": settling_time(
            t,
            error,
            start_time=float(scenario.env_switch_time),
            threshold=0.005,
            window_s=1.0,
        ),
        "control_energy_v2_s": _integral(t[: u_v.size], u_v ** 2),
        "control_smoothness_mean_abs_delta_v": float(np.mean(np.abs(du))) if du.size else 0.0,
        "control_smoothness_rms_delta_v": _rms(du),
        "max_abs_u_v": _peak_abs(u_v),
        "max_abs_delta_u_v": _peak_abs(du),
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
    out_dir.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(out_dir / f"{safe_stem(scenario.name)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    fig.savefig(Path(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        normal_rows.append(compute_non_bode_metrics(result, action_limit=action_limit))
        if save_plots:
            plot_scenario_result(result, out_dir / "plots" / "scenarios")

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
        "methodology": "focused_unified_eval_v1",
        "groups": groups,
        "metrics": [
            "rms_error_m",
            "peak_error_m",
            "post_contact_rms_error_m",
            "post_contact_peak_error_m",
            "settling_time_s",
            "control_energy_v2_s",
            "control_smoothness_mean_abs_delta_v",
            "saturation_fraction",
            "failure_flag",
        ],
        "bode_metrics": ["frequency_rad_s", "gain", "gain_dB", "phase_lag_deg"],
        "env_kwargs": env_kwargs,
        "scenarios": [asdict(scenario) for scenario in scenarios],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "focused_eval_metrics.csv", normal_rows)
    save_json(out_dir / "focused_eval_summary.json", summary_payload)
    if bode_rows:
        write_csv(out_dir / "focused_eval_bode.csv", bode_rows)
    return {
        "summary": summary_payload,
        "metrics": normal_rows,
        "bode": bode_rows,
    }
