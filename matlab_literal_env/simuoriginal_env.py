from __future__ import annotations

from dataclasses import replace
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from .. import config as cfg
except ImportError:  # pragma: no cover - direct script execution
    import config as cfg

from .simuoriginal_replica import (
    FE_MODE_CHOICES,
    FE_MODE_DYNAMICS,
    FE_MODE_GUI,
    ParmsOriginal,
    SimuOriginalProfile,
    SimuOriginalState,
    _rk4_step,
    build_saved_simuoriginal_state,
    environment_force,
    gui_environment_force,
    saved_environment,
    simuoriginal_derivatives,
)


_TWO_PI = 2.0 * math.pi
STROKE_LIMIT_MODES = ("terminate", "clamp")


def position_transparency_ratio(x_m: float, x_s: float, eps: float = 1e-9) -> float:
    """Return the position transparency ratio x_m / x_s with a stable zero-zero limit."""
    x_m = float(x_m)
    x_s = float(x_s)
    eps = float(eps)
    if abs(x_s) >= eps:
        return float(x_m / x_s)
    if abs(x_m - x_s) < eps:
        return 1.0
    return float(x_m / (eps if x_s >= 0.0 else -eps))


def _normalize_action_levels(action_levels: list[float] | tuple[float, ...] | np.ndarray | None) -> np.ndarray:
    if action_levels is None:
        levels = np.asarray(cfg.V_LEVELS, dtype=np.float64)
    else:
        levels = np.asarray(action_levels, dtype=np.float64).reshape(-1)
    if levels.size == 0:
        raise ValueError("action_levels must contain at least one voltage level")
    if not np.all(np.isfinite(levels)):
        raise ValueError("action_levels must be finite")
    return levels.astype(np.float64, copy=True)


def _force_waveform_value(phase: float, waveform: str) -> float:
    waveform = str(waveform).strip().lower()
    if waveform in {"constant", "dc"}:
        return 0.0
    if waveform == "sine":
        return math.sin(phase)
    if waveform == "cosine":
        return math.cos(phase)
    if waveform in {"square", "pulse"}:
        return 1.0 if math.sin(phase) >= 0.0 else -1.0
    if waveform == "ramp":
        phase_mod = float(phase) % _TWO_PI
        return -1.0 + (2.0 * phase_mod / _TWO_PI)
    if waveform == "multisine":
        return 0.75 * math.sin(phase) + 0.25 * math.sin((2.0 * phase) + 0.35)
    raise ValueError(f"Unknown force waveform: {waveform}")


def _as_float_tuple(values) -> tuple[float, ...]:
    if values is None:
        return ()
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return tuple(float(v) for v in arr)


def _build_force_noise_components(
    noise_seed: int | None,
    n_components: int = 4,
) -> tuple[tuple[float, float, float], ...]:
    if n_components <= 0:
        return ()

    seed = 0 if noise_seed is None else int(noise_seed)
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.35, 1.0, size=n_components)
    norm = math.sqrt(max(0.5 * float(np.sum(weights ** 2)), 1e-12))
    coeffs = weights / norm
    freq_multipliers = rng.uniform(1.4, 4.5, size=n_components)
    phases = rng.uniform(0.0, _TWO_PI, size=n_components)
    return tuple(
        (float(coeff), float(freq_mul), float(phase))
        for coeff, freq_mul, phase in zip(coeffs, freq_multipliers, phases)
    )


def _force_noise_signal(
    t: float,
    base_freq_hz: float,
    noise_std: float,
    noise_components: tuple[tuple[float, float, float], ...] | tuple[()] = (),
) -> float:
    if noise_std <= 0.0 or not noise_components:
        return 0.0

    freq = abs(float(base_freq_hz))
    if freq <= 1e-9:
        freq = float(cfg.FORCE_INPUT_FREQ)

    total = 0.0
    for coeff, freq_mul, phase in noise_components:
        total += coeff * math.sin((_TWO_PI * freq * freq_mul * t) + phase)
    return float(noise_std) * total


class SimuOriginalReplicaEnv(gym.Env):
    """Gym-style wrapper around the nonlinear SimuOriginal plant replica."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    IX_XM, IX_VM = 0, 1
    IX_XS, IX_VS = 2, 3
    IX_PM1, IX_PM2 = 4, 5
    IX_PS1, IX_PS2 = 6, 7
    IX_ML1, IX_ML2 = 8, 9
    IX_XV, IX_VV = 10, 11
    N_STATE = 12

    def __init__(
        self,
        render_mode: str | None = None,
        env_mode: str | None = None,
        master_input_mode: str | None = None,
        episode_duration: float | None = None,
        env_switch_time: float | None = None,
        terminate_on_error: bool = True,
        legacy_baseline_env: bool = False,
        reset_position_mode: str = "midpoint",
        enforce_stroke_limit: bool = True,
        stroke_limit_mode: str = "terminate",
        edge_action_damping_buffer_m: float = 0.0,
        edge_action_min_scale: float = 1.0,
        action_levels: list[float] | tuple[float, ...] | np.ndarray | None = None,
        parms: ParmsOriginal | None = None,
        profile: SimuOriginalProfile | None = None,
        reset_options: dict | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.env_mode = env_mode or cfg.ENV_MODE_CONSTANT
        self.master_input_mode = master_input_mode or cfg.DEFAULT_MASTER_INPUT_MODE
        if self.master_input_mode != cfg.MASTER_INPUT_FORCE:
            raise ValueError("SimuOriginalReplicaEnv currently supports only force-driven master input.")

        self.parms = parms or ParmsOriginal()
        self._tube_compliance = float(self.parms.tube_compliance)
        self.profile = profile or SimuOriginalProfile(fixed_step=self.parms.Ts)
        self._default_profile = self.profile
        self.episode_duration = float(
            cfg.EPISODE_DURATION if episode_duration is None else episode_duration
        )
        self.env_switch_time = float(
            self.profile.env_switch_time if env_switch_time is None else env_switch_time
        )
        self.max_steps = max(1, int(round(self.episode_duration / cfg.RL_DT)))
        self.terminate_on_error = bool(terminate_on_error)
        self.legacy_baseline_env = bool(legacy_baseline_env)
        self.rl_dt = float(cfg.RL_DT)
        self.internal_dt = float(self.parms.Ts)
        self.sub_steps = max(1, int(round(self.rl_dt / self.internal_dt)))
        self.reset_position_mode = str(reset_position_mode).strip().lower()
        self.enforce_stroke_limit = bool(enforce_stroke_limit)
        self.stroke_limit_mode = str(stroke_limit_mode).strip().lower()
        if self.stroke_limit_mode not in STROKE_LIMIT_MODES:
            raise ValueError(f"Unknown stroke_limit_mode: {stroke_limit_mode}")
        if self.legacy_baseline_env:
            self.reset_position_mode = "zero"
            self.enforce_stroke_limit = False
        # Legacy baseline env used origin-centered positions with no explicit
        # stroke clamp. The newer RL-safe env uses midpoint-centered positions
        # with a hard geometric stroke check.
        self.x_eq = 0.0 if self.reset_position_mode in {"zero", "origin", "legacy"} else 0.5 * float(self.parms.l_cyl)
        self.stroke_min = 0.0
        self.stroke_max = float(self.parms.l_cyl)
        self.edge_action_damping_buffer_m = max(0.0, float(edge_action_damping_buffer_m))
        self.edge_action_min_scale = float(np.clip(edge_action_min_scale, 0.0, 1.0))
        self.default_reset_options = dict(reset_options or {})

        self._action_table = _normalize_action_levels(action_levels)
        self.action_levels = self._action_table.copy()
        self._u_min = float(self._action_table.min())
        self._u_max = float(self._action_table.max())
        self.action_space = spaces.Box(
            low=np.array([self._u_min], dtype=np.float32),
            high=np.array([self._u_max], dtype=np.float32),
            dtype=np.float32,
        )
        low = -np.ones(10, dtype=np.float32) * 2.0
        high = np.ones(10, dtype=np.float32) * 2.0
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.runtime_profile = self.profile
        self.replica_state = build_saved_simuoriginal_state(
            self.parms,
            init_position_mode=self.reset_position_mode,
        ).as_array()
        self.state = np.zeros(self.N_STATE, dtype=np.float64)
        self._history: dict[str, list] | None = None
        self.last_u_v = 0.0
        self.current_env_label = "skin"
        self.current_env_id = 0
        self.invalid_state = False
        self.invalid_reason: str | None = None
        self.singularity_time: float | None = None
        self.termination_reason: str | None = None
        self.tracking_error_fail = False
        self.last_terminated = False
        self.last_truncated = False
        self.F_h_nominal = 0.0
        self.F_h_noise = 0.0
        self.F_h = 0.0
        self.F_e = 0.0
        self.a_m_signal = 0.0
        self.fe_mode = FE_MODE_GUI
        self.requested_u_v = 0.0
        self.hit_stroke_stop = False

    def _sync_force_parameters(self) -> None:
        self.force_amp = abs(float(getattr(self, "force_amp", getattr(self, "fh_amp", cfg.FORCE_INPUT_AMP))))
        self.force_bias = float(getattr(self, "force_bias", getattr(self, "fh_bias", 0.0)))
        if hasattr(self, "force_freq_rad") or hasattr(self, "fh_freq_rad"):
            freq_rad_value = getattr(self, "force_freq_rad", None)
            if freq_rad_value is None:
                freq_rad_value = getattr(self, "fh_freq_rad")
            self.force_freq_rad = float(freq_rad_value)
            self.force_freq = self.force_freq_rad / _TWO_PI
        else:
            self.force_freq = float(getattr(self, "force_freq", getattr(self, "fh_freq", cfg.FORCE_INPUT_FREQ)))
            self.force_freq_rad = _TWO_PI * self.force_freq
        self.force_phase = float(getattr(self, "force_phase", getattr(self, "fh_phase", cfg.FORCE_INPUT_PHASE)))
        self.force_waveform = str(getattr(self, "force_waveform", getattr(self, "fh_waveform", "sine"))).strip().lower()
        self.force_noise_std = abs(float(getattr(self, "force_noise_std", getattr(self, "fh_noise_std", 0.0))))
        self.force_noise_seed = int(getattr(self, "force_noise_seed", getattr(self, "fh_noise_seed", 0)))
        self.force_noise_components = (
            _build_force_noise_components(self.force_noise_seed) if self.force_noise_std > 0.0 else ()
        )
        self.force_release_time = getattr(self, "force_release_time", getattr(self, "fh_release_time", None))
        self.force_release_value = float(getattr(self, "force_release_value", getattr(self, "fh_release_value", 0.0)))
        self.force_chirp_end_freq_rad = float(
            getattr(self, "force_chirp_end_freq_rad", getattr(self, "fh_chirp_end_freq_rad", self.force_freq_rad))
        )
        self.force_chirp_duration = float(
            getattr(self, "force_chirp_duration", getattr(self, "fh_chirp_duration", self.episode_duration))
        )
        self.force_sequence_times = _as_float_tuple(
            getattr(self, "force_sequence_times", getattr(self, "fh_sequence_times", ()))
        )
        self.force_sequence_values = _as_float_tuple(
            getattr(self, "force_sequence_values", getattr(self, "fh_sequence_values", ()))
        )
        self.fh_amp = self.force_amp
        self.fh_bias = self.force_bias
        self.fh_freq = self.force_freq
        self.fh_freq_rad = self.force_freq_rad
        self.fh_phase = self.force_phase
        self.fh_waveform = self.force_waveform
        self.fh_noise_std = self.force_noise_std
        self.fh_noise_seed = self.force_noise_seed
        self.fh_sequence_times = self.force_sequence_times
        self.fh_sequence_values = self.force_sequence_values

    def _update_runtime_profile(self) -> None:
        if self.env_mode == cfg.ENV_MODE_CONSTANT:
            self.runtime_profile = replace(
                self.profile,
                env_switch_time=float("inf"),
                skin_Ke=float(self.profile.skin_Ke),
                skin_Be=float(self.profile.skin_Be),
                delta_Ke_after_switch=0.0,
                delta_Be_after_switch=0.0,
            )
            return
        if self.env_mode == cfg.ENV_MODE_CHANGING:
            self.runtime_profile = replace(
                self.profile,
                env_switch_time=float(self.env_switch_time),
            )
            return
        raise ValueError(f"Unknown env_mode: {self.env_mode}")

    def _update_environment_mode(self) -> None:
        Ke, Be = saved_environment(self.t, self.runtime_profile)
        if abs(Ke - self.profile.skin_Ke) < 1e-12 and abs(Be - self.profile.skin_Be) < 1e-12:
            self.current_env_label = "skin"
            self.current_env_id = 0
        else:
            self.current_env_label = "fat"
            self.current_env_id = 1
        self.Be = float(Be)
        self.Ke = float(Ke)

    def _force_nominal_value(self, t: float) -> float:
        if self.force_waveform in {"sequence", "step_sequence"}:
            if not self.force_sequence_values:
                return self.force_bias
            if len(self.force_sequence_times) == len(self.force_sequence_values):
                idx = int(np.searchsorted(self.force_sequence_times, float(t), side="right") - 1)
                idx = int(np.clip(idx, 0, len(self.force_sequence_values) - 1))
                return float(self.force_sequence_values[idx])
            idx = int(np.searchsorted(self.force_sequence_times, float(t), side="right"))
            idx = int(np.clip(idx, 0, len(self.force_sequence_values) - 1))
            return float(self.force_sequence_values[idx])

        if self.force_waveform == "chirp":
            duration = max(abs(float(self.force_chirp_duration)), 1e-9)
            t_eff = min(max(float(t), 0.0), duration)
            end_freq = float(self.force_chirp_end_freq_rad)
            chirp_rate = (end_freq - float(self.force_freq_rad)) / duration
            phase = self.force_phase + (self.force_freq_rad * t_eff) + (0.5 * chirp_rate * t_eff ** 2)
            if t > duration:
                phase += end_freq * (float(t) - duration)
            return self.force_bias + (self.force_amp * math.sin(phase))

        phase = (self.force_freq_rad * t) + self.force_phase
        return self.force_bias + (self.force_amp * _force_waveform_value(phase, self.force_waveform))

    def _force_components(self, t: float) -> tuple[float, float, float]:
        release_time = self.force_release_time
        if release_time is not None and float(t) >= float(release_time):
            release_value = float(self.force_release_value)
            return release_value, 0.0, release_value

        nominal = self._force_nominal_value(t)
        noise = _force_noise_signal(
            t,
            self.force_freq,
            self.force_noise_std,
            self.force_noise_components,
        )
        return nominal, noise, nominal + noise

    def _force_input(self, t: float) -> float:
        _, _, total = self._force_components(t)
        return total

    def _control_input(self, _t: float) -> float:
        return float(self.last_u_v)

    def _to_env_state(self, replica_state: np.ndarray) -> np.ndarray:
        return np.array(
            [
                replica_state[3],
                replica_state[2],
                replica_state[7],
                replica_state[6],
                replica_state[0],
                replica_state[1],
                replica_state[4],
                replica_state[5],
                replica_state[8],
                replica_state[9],
                replica_state[10],
                replica_state[11],
            ],
            dtype=np.float64,
        )

    def _volumes_are_valid(self, replica_state: np.ndarray) -> bool:
        if not np.all(np.isfinite(replica_state)):
            return False
        xm = float(replica_state[3])
        xs = float(replica_state[7])
        volumes = (
            self.parms.V_md + self.parms.A_p * xm,
            self.parms.V_md + self.parms.A_p * (self.parms.l_cyl - xm),
            self.parms.V_md + self.parms.A_p * xs + self._tube_compliance,
            self.parms.V_sd + self.parms.A_p * (self.parms.l_cyl - xs) + self._tube_compliance,
        )
        return min(volumes) > 0.0

    def _stroke_is_valid(self, replica_state: np.ndarray) -> bool:
        if not self.enforce_stroke_limit:
            return True
        xm = float(replica_state[3])
        xs = float(replica_state[7])
        return bool(
            self.stroke_min <= xm <= self.stroke_max
            and self.stroke_min <= xs <= self.stroke_max
            and np.all(np.isfinite((xm, xs)))
        )

    def _apply_stroke_clamp(self, replica_state: np.ndarray) -> tuple[np.ndarray, bool]:
        clamped = np.asarray(replica_state, dtype=np.float64).copy()

        hit = False
        stroke_min = self.stroke_min
        stroke_max = self.stroke_max
        if clamped[3] < stroke_min:
            clamped[3] = stroke_min
            if clamped[2] < 0.0:
                clamped[2] = 0.0
            hit = True
        elif clamped[3] > stroke_max:
            clamped[3] = stroke_max
            if clamped[2] > 0.0:
                clamped[2] = 0.0
            hit = True

        if clamped[7] < stroke_min:
            clamped[7] = stroke_min
            if clamped[6] < 0.0:
                clamped[6] = 0.0
            hit = True
        elif clamped[7] > stroke_max:
            clamped[7] = stroke_max
            if clamped[6] > 0.0:
                clamped[6] = 0.0
            hit = True

        return clamped, hit

    def _derivative_fn(self, t: float, y: np.ndarray) -> np.ndarray:
        return simuoriginal_derivatives(
            t,
            y,
            parms=self.parms,
            profile=self.runtime_profile,
            F_h_fn=self._force_input,
            u_fn=self._control_input,
        )

    def _update_signals(self) -> None:
        self._update_environment_mode()
        self.state = self._to_env_state(self.replica_state)
        self.F_h_nominal, self.F_h_noise, self.F_h = self._force_components(self.t)
        self.F_h = float(self.F_h)
        if self.fe_mode == FE_MODE_DYNAMICS:
            self.F_e = environment_force(
                self.state[self.IX_XS],
                self.state[self.IX_VS],
                self.Ke,
                self.Be,
            )
        else:
            # Alternate reference path:
            # self.F_e = environment_force(self.state[self.IX_XS], self.state[self.IX_VS], self.Ke, self.Be)
            self.F_e = gui_environment_force(
                self.state[self.IX_XS],
                self.state[self.IX_VS],
                self.runtime_profile,
            )
        deriv = self._derivative_fn(self.t, self.replica_state)
        self.a_m_signal = float(deriv[2]) if np.all(np.isfinite(deriv)) else 0.0

    def _edge_action_scale(self) -> float:
        buffer_m = float(self.edge_action_damping_buffer_m)
        if buffer_m <= 0.0:
            return 1.0
        s = SimuOriginalState.from_array(self.replica_state)
        dist_m = min(float(s.xm), self.stroke_max - float(s.xm))
        dist_s = min(float(s.xs), self.stroke_max - float(s.xs))
        dist_to_edge = min(dist_m, dist_s)
        if dist_to_edge >= buffer_m:
            return 1.0
        severity = 1.0 - float(np.clip(dist_to_edge / buffer_m, 0.0, 1.0))
        return float(self.edge_action_min_scale + ((1.0 - self.edge_action_min_scale) * (1.0 - severity)))

    def get_equilibrium_position(self) -> float:
        return self.x_eq

    def get_centered_positions(self) -> tuple[float, float]:
        return (
            float(self.state[self.IX_XM] - self.x_eq),
            float(self.state[self.IX_XS] - self.x_eq),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        merged_options = dict(self.default_reset_options)
        merged_options.update(options or {})
        options = merged_options
        self.profile = self._default_profile
        if "episode_duration" in options and options["episode_duration"] is not None:
            self.episode_duration = float(options["episode_duration"])
            self.max_steps = max(1, int(round(self.episode_duration / cfg.RL_DT)))
        if "env_switch_time" in options and options["env_switch_time"] is not None:
            self.env_switch_time = float(options["env_switch_time"])

        self.replica_state = build_saved_simuoriginal_state(
            self.parms,
            init_position_mode=self.reset_position_mode,
        ).as_array()
        self.state = self._to_env_state(self.replica_state)
        self.t = 0.0
        self.step_count = 0
        self.invalid_state = False
        self.invalid_reason = None
        self.singularity_time = None
        self.termination_reason = None
        self.tracking_error_fail = False
        self.last_terminated = False
        self.last_truncated = False
        self.last_u_v = 0.0
        self.requested_u_v = 0.0
        self.hit_stroke_stop = False

        self.force_amp = float(cfg.FORCE_INPUT_AMP)
        self.force_bias = 0.0
        self.force_freq = float(cfg.FORCE_INPUT_FREQ)
        self.force_phase = float(cfg.FORCE_INPUT_PHASE)
        self.force_waveform = "sine"
        self.force_noise_std = 0.0
        self.force_noise_seed = 0
        self.force_noise_components: tuple[tuple[float, float, float], ...] | tuple[()] = ()
        self.force_release_time = None
        self.force_release_value = 0.0
        self.force_chirp_end_freq_rad = _TWO_PI * self.force_freq
        self.force_chirp_duration = self.episode_duration
        self.force_sequence_times: tuple[float, ...] = ()
        self.force_sequence_values: tuple[float, ...] = ()
        self.fe_mode = FE_MODE_GUI
        for key in (
            "episode_duration",
            "force_amp",
            "force_bias",
            "force_freq",
            "force_freq_rad",
            "force_phase",
            "force_waveform",
            "force_noise_std",
            "force_noise_seed",
            "force_release_time",
            "force_release_value",
            "force_chirp_end_freq_rad",
            "force_chirp_duration",
            "force_sequence_times",
            "force_sequence_values",
            "fe_mode",
            "legacy_baseline_env",
            "reset_position_mode",
            "enforce_stroke_limit",
            "stroke_limit_mode",
            "edge_action_damping_buffer_m",
            "edge_action_min_scale",
            "fh_amp",
            "fh_bias",
            "fh_freq",
            "fh_freq_rad",
            "fh_phase",
            "fh_waveform",
            "fh_noise_std",
            "fh_noise_seed",
            "fh_release_time",
            "fh_release_value",
            "fh_chirp_end_freq_rad",
            "fh_chirp_duration",
            "fh_sequence_times",
            "fh_sequence_values",
        ):
            if key in options:
                setattr(self, key, options[key])
        if any(
            key in options
            for key in ("pre_switch_Ke", "pre_switch_Be", "post_switch_Ke", "post_switch_Be", "K_e", "B_e")
        ):
            pre_Ke = float(options.get("pre_switch_Ke", self.profile.skin_Ke))
            pre_Be = float(options.get("pre_switch_Be", self.profile.skin_Be))
            default_post_Ke = pre_Ke + float(self.profile.delta_Ke_after_switch)
            default_post_Be = pre_Be + float(self.profile.delta_Be_after_switch)
            post_Ke = float(options.get("post_switch_Ke", options.get("K_e", default_post_Ke)))
            post_Be = float(options.get("post_switch_Be", options.get("B_e", default_post_Be)))
            self.profile = replace(
                self.profile,
                skin_Ke=pre_Ke,
                skin_Be=pre_Be,
                delta_Ke_after_switch=post_Ke - pre_Ke,
                delta_Be_after_switch=post_Be - pre_Be,
            )
        self.legacy_baseline_env = bool(getattr(self, "legacy_baseline_env", False))
        self.reset_position_mode = str(getattr(self, "reset_position_mode", self.reset_position_mode)).strip().lower()
        self.enforce_stroke_limit = bool(getattr(self, "enforce_stroke_limit", self.enforce_stroke_limit))
        self.stroke_limit_mode = str(getattr(self, "stroke_limit_mode", self.stroke_limit_mode)).strip().lower()
        if self.stroke_limit_mode not in STROKE_LIMIT_MODES:
            raise ValueError(f"Unknown stroke_limit_mode: {self.stroke_limit_mode}")
        if self.legacy_baseline_env:
            self.reset_position_mode = "zero"
            self.enforce_stroke_limit = False
        self.x_eq = 0.0 if self.reset_position_mode in {"zero", "origin", "legacy"} else 0.5 * float(self.parms.l_cyl)
        if options.get("initial_state") is not None:
            initial_state = np.asarray(options["initial_state"], dtype=np.float64).reshape(-1)
            if initial_state.size != self.N_STATE:
                raise ValueError(f"initial_state must have {self.N_STATE} entries, got {initial_state.size}")
            self.replica_state = initial_state.copy()
        if options.get("initial_state_delta") is not None:
            initial_delta = np.asarray(options["initial_state_delta"], dtype=np.float64).reshape(-1)
            if initial_delta.size != self.N_STATE:
                raise ValueError(f"initial_state_delta must have {self.N_STATE} entries, got {initial_delta.size}")
            self.replica_state = self.replica_state + initial_delta
        self.state = self._to_env_state(self.replica_state)
        self.fe_mode = str(self.fe_mode).strip().lower()
        if self.fe_mode not in FE_MODE_CHOICES:
            raise ValueError(f"Unknown fe_mode: {self.fe_mode}")
        self._sync_force_parameters()
        self._update_runtime_profile()
        self._update_signals()

        self._history = {
            "time": [],
            "x_m": [],
            "x_s": [],
            "x_m_centered": [],
            "x_s_centered": [],
            "v_m": [],
            "v_s": [],
            "P_m1": [],
            "P_m2": [],
            "P_s1": [],
            "P_s2": [],
            "mdot_L1": [],
            "mdot_L2": [],
            "F_h": [],
            "F_h_nominal": [],
            "F_h_noise": [],
            "a_m_signal": [],
            "F_e": [],
            "u_v": [],
            "requested_u_v": [],
            "x_v": [],
            "x_v_dot": [],
            "env_id": [],
            "env_label": [],
            "pos_error": [],
            "transparency_ratio": [],
            "transparency_error": [],
            "reward_track": [],
            "reward_effort": [],
            "reward_transparency": [],
            "reward": [],
            "invalid_state": [],
            "invalid_reason": [],
            "hit_stroke_stop": [],
            "terminated": [],
            "truncated": [],
            "tracking_error_fail": [],
            "termination_reason": [],
        }

        return self._get_obs(), self._get_info()

    def _log_step(self, reward: float, track_term: float, effort_term: float, transparency_term: float) -> None:
        if self._history is None:
            return
        x_m_centered, x_s_centered = self.get_centered_positions()
        pos_error = float(self.state[self.IX_XM] - self.state[self.IX_XS])
        transparency_ratio = position_transparency_ratio(self.state[self.IX_XM], self.state[self.IX_XS])
        transparency_error = float(transparency_ratio - 1.0)
        self._history["time"].append(self.t)
        self._history["x_m"].append(self.state[self.IX_XM])
        self._history["x_s"].append(self.state[self.IX_XS])
        self._history["x_m_centered"].append(x_m_centered)
        self._history["x_s_centered"].append(x_s_centered)
        self._history["v_m"].append(self.state[self.IX_VM])
        self._history["v_s"].append(self.state[self.IX_VS])
        self._history["P_m1"].append(self.state[self.IX_PM1])
        self._history["P_m2"].append(self.state[self.IX_PM2])
        self._history["P_s1"].append(self.state[self.IX_PS1])
        self._history["P_s2"].append(self.state[self.IX_PS2])
        self._history["mdot_L1"].append(self.state[self.IX_ML1])
        self._history["mdot_L2"].append(self.state[self.IX_ML2])
        self._history["F_h"].append(self.F_h)
        self._history["F_h_nominal"].append(self.F_h_nominal)
        self._history["F_h_noise"].append(self.F_h_noise)
        self._history["a_m_signal"].append(self.a_m_signal)
        self._history["F_e"].append(self.F_e)
        self._history["u_v"].append(self.last_u_v)
        self._history["requested_u_v"].append(self.requested_u_v)
        self._history["x_v"].append(self.state[self.IX_XV])
        self._history["x_v_dot"].append(self.state[self.IX_VV])
        self._history["env_id"].append(self.current_env_id)
        self._history["env_label"].append(self.current_env_label)
        self._history["pos_error"].append(pos_error)
        self._history["transparency_ratio"].append(transparency_ratio)
        self._history["transparency_error"].append(transparency_error)
        self._history["reward_track"].append(track_term)
        self._history["reward_effort"].append(effort_term)
        self._history["reward_transparency"].append(transparency_term)
        self._history["reward"].append(reward)
        self._history["invalid_state"].append(self.invalid_state)
        self._history["invalid_reason"].append(self.invalid_reason)
        self._history["hit_stroke_stop"].append(self.hit_stroke_stop)
        self._history["terminated"].append(self.last_terminated)
        self._history["truncated"].append(self.last_truncated)
        self._history["tracking_error_fail"].append(self.tracking_error_fail)
        self._history["termination_reason"].append(self.termination_reason)

    def _step_with_voltage(self, u_v: float):
        self.requested_u_v = float(u_v)
        self.last_u_v = float(u_v) * self._edge_action_scale()
        self._sync_force_parameters()
        self._update_runtime_profile()
        self.hit_stroke_stop = False
        clamp_stroke = bool(self.enforce_stroke_limit and self.stroke_limit_mode == "clamp")

        for _ in range(self.sub_steps):
            if clamp_stroke:
                self.replica_state, hit_stop = self._apply_stroke_clamp(self.replica_state)
                self.hit_stroke_stop = self.hit_stroke_stop or hit_stop
            if not self._volumes_are_valid(self.replica_state):
                self.invalid_state = True
                self.invalid_reason = "volume_singularity"
                self.singularity_time = float(self.t)
                break
            if not clamp_stroke and not self._stroke_is_valid(self.replica_state):
                self.invalid_state = True
                self.invalid_reason = "stroke_limit"
                self.singularity_time = float(self.t)
                break
            next_state = _rk4_step(self._derivative_fn, self.t, self.replica_state, self.internal_dt)
            self.t += self.internal_dt
            if clamp_stroke:
                next_state, hit_stop = self._apply_stroke_clamp(next_state)
                self.hit_stroke_stop = self.hit_stroke_stop or hit_stop
            if not self._volumes_are_valid(next_state):
                self.invalid_state = True
                self.invalid_reason = "volume_singularity"
                self.singularity_time = float(self.t)
                break
            if not clamp_stroke and not self._stroke_is_valid(next_state):
                self.invalid_state = True
                self.invalid_reason = "stroke_limit"
                self.singularity_time = float(self.t)
                break
            self.replica_state = next_state

        self.step_count += 1
        self._update_signals()

        pos_error = float(self.state[self.IX_XM] - self.state[self.IX_XS])
        norm_pos_error = float(
            np.clip(pos_error / cfg.MAX_POSITION_ERROR, -cfg.POS_ERR_NORM_CLIP, cfg.POS_ERR_NORM_CLIP)
        )
        transparency_ratio = position_transparency_ratio(self.state[self.IX_XM], self.state[self.IX_XS])
        transparency_error = float(transparency_ratio - 1.0)
        norm_transparency_error = transparency_error
        track_term = cfg.ALPHA_TRACKING * norm_pos_error ** 2
        effort_term = cfg.GAMMA_EFFORT * u_v ** 2
        transparency_term = cfg.BETA_TRANSPARENCY * norm_transparency_error ** 2
        reward = -(track_term + effort_term + transparency_term)

        terminated = bool(self.invalid_state)
        self.tracking_error_fail = False
        self.termination_reason = self.invalid_reason if self.invalid_state else None
        if self.terminate_on_error and abs(pos_error) >= cfg.POS_ERROR_FAIL_THRESHOLD:
            terminated = True
            self.tracking_error_fail = True
            if self.termination_reason is None:
                self.termination_reason = "tracking_error_fail"
        truncated = self.step_count >= self.max_steps
        if truncated and self.termination_reason is None:
            self.termination_reason = "max_steps"
        self.last_terminated = terminated
        self.last_truncated = truncated

        self._log_step(reward, track_term, effort_term, transparency_term)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _action_to_voltage(self, action: int | float | np.ndarray) -> float:
        if isinstance(action, (int, np.integer)):
            idx = int(action)
            if idx < 0 or idx >= self._action_table.size:
                raise AssertionError(f"Invalid discrete action index {action}")
            return float(self._action_table[idx])
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 1:
            raise AssertionError(f"Invalid continuous action shape: {np.asarray(action).shape}")
        return float(np.clip(float(arr[0]), self._u_min, self._u_max))

    def step(self, action: int | float | np.ndarray):
        return self._step_with_voltage(self._action_to_voltage(action))

    def step_voltage(self, u_v: float):
        u_v = float(np.clip(u_v, self._u_min, self._u_max))
        return self._step_with_voltage(u_v)

    def render(self):
        return self._history

    def _get_obs(self) -> np.ndarray:
        x_m_centered, x_s_centered = self.get_centered_positions()
        return np.array(
            [
                x_s_centered / cfg.OBS_SCALE_POS,
                x_m_centered / cfg.OBS_SCALE_POS,
                self.state[self.IX_VS] / cfg.OBS_SCALE_VEL,
                self.state[self.IX_VM] / cfg.OBS_SCALE_VEL,
                self.state[self.IX_PS1] / cfg.OBS_SCALE_PRESSURE,
                self.state[self.IX_PS2] / cfg.OBS_SCALE_PRESSURE,
                self.state[self.IX_PM1] / cfg.OBS_SCALE_PRESSURE,
                self.state[self.IX_PM2] / cfg.OBS_SCALE_PRESSURE,
                self.state[self.IX_ML1] / cfg.OBS_SCALE_FLOW,
                self.state[self.IX_ML2] / cfg.OBS_SCALE_FLOW,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        x_m_centered, x_s_centered = self.get_centered_positions()
        transparency_ratio = position_transparency_ratio(self.state[self.IX_XM], self.state[self.IX_XS])
        return {
            "time": self.t,
            "u_v": self.last_u_v,
            "requested_u_v": self.requested_u_v,
            "F_h": self.F_h,
            "F_h_nominal": self.F_h_nominal,
            "F_h_noise": self.F_h_noise,
            "a_m_signal": self.a_m_signal,
            "F_e": self.F_e,
            "env_id": self.current_env_id,
            "env_label": self.current_env_label,
            "x_m": self.state[self.IX_XM],
            "x_s": self.state[self.IX_XS],
            "x_v": self.state[self.IX_XV],
            "x_v_dot": self.state[self.IX_VV],
            "mdot_L1": self.state[self.IX_ML1],
            "mdot_L2": self.state[self.IX_ML2],
            "x_eq": self.x_eq,
            "x_m_centered": x_m_centered,
            "x_s_centered": x_s_centered,
            "pos_error": float(self.state[self.IX_XM] - self.state[self.IX_XS]),
            "transparency_ratio": transparency_ratio,
            "transparency_error": transparency_ratio - 1.0,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "episode_duration": self.episode_duration,
            "env_switch_time": self.env_switch_time,
            "terminate_on_error": self.terminate_on_error,
            "master_input_mode": self.master_input_mode,
            "force_bias": self.force_bias,
            "force_freq": self.force_freq,
            "force_freq_rad": self.force_freq_rad,
            "force_waveform": self.force_waveform,
            "force_release_time": self.force_release_time,
            "force_release_value": self.force_release_value,
            "force_chirp_end_freq_rad": self.force_chirp_end_freq_rad,
            "force_chirp_duration": self.force_chirp_duration,
            "force_sequence_times": self.force_sequence_times,
            "force_sequence_values": self.force_sequence_values,
            "fe_mode": self.fe_mode,
            "legacy_baseline_env": self.legacy_baseline_env,
            "reset_position_mode": self.reset_position_mode,
            "enforce_stroke_limit": self.enforce_stroke_limit,
            "stroke_limit_mode": self.stroke_limit_mode,
            "hit_stroke_stop": self.hit_stroke_stop,
            "edge_action_damping_buffer_m": self.edge_action_damping_buffer_m,
            "edge_action_min_scale": self.edge_action_min_scale,
            "invalid_state": self.invalid_state,
            "invalid_reason": self.invalid_reason,
            "singularity_time": self.singularity_time,
            "tracking_error_fail": self.tracking_error_fail,
            "terminated": self.last_terminated,
            "truncated": self.last_truncated,
            "termination_reason": self.termination_reason,
            "pos_error_fail_threshold": float(cfg.POS_ERROR_FAIL_THRESHOLD),
        }

    def discretise_obs(self, obs: np.ndarray) -> tuple[int, ...]:
        return (
            int(np.digitize(obs[0], cfg.SLAVE_POS_ERROR_BINS)),
            int(np.digitize(obs[1], cfg.MASTER_POS_ERROR_BINS)),
            int(np.digitize(obs[4], cfg.SLAVE_P1_BINS)),
            int(np.digitize(obs[5], cfg.SLAVE_P2_BINS)),
            int(np.digitize(obs[6], cfg.MASTER_P1_BINS)),
            int(np.digitize(obs[7], cfg.MASTER_P2_BINS)),
            int(np.digitize(obs[8], cfg.MASS_FLOW1_BINS)),
            int(np.digitize(obs[9], cfg.MASS_FLOW2_BINS)),
        )

    def get_state_dims(self) -> tuple[int, ...]:
        return (
            len(cfg.SLAVE_POS_ERROR_BINS) + 1,
            len(cfg.MASTER_POS_ERROR_BINS) + 1,
            len(cfg.SLAVE_P1_BINS) + 1,
            len(cfg.SLAVE_P2_BINS) + 1,
            len(cfg.MASTER_P1_BINS) + 1,
            len(cfg.MASTER_P2_BINS) + 1,
            len(cfg.MASS_FLOW1_BINS) + 1,
            len(cfg.MASS_FLOW2_BINS) + 1,
        )

    def discretise_obs_reduced(self, obs: np.ndarray) -> tuple[int, ...]:
        slave_pos = obs[0] * cfg.OBS_SCALE_POS
        master_pos = obs[1] * cfg.OBS_SCALE_POS
        v_s = obs[2] * cfg.OBS_SCALE_VEL
        v_m = obs[3] * cfg.OBS_SCALE_VEL
        P_s1 = obs[4] * cfg.OBS_SCALE_PRESSURE
        P_s2 = obs[5] * cfg.OBS_SCALE_PRESSURE
        P_m1 = obs[6] * cfg.OBS_SCALE_PRESSURE
        P_m2 = obs[7] * cfg.OBS_SCALE_PRESSURE

        tracking_error = master_pos - slave_pos
        velocity_error = v_m - v_s
        slave_pdiff = P_s1 - P_s2
        master_pdiff = P_m1 - P_m2
        return (
            int(np.digitize(tracking_error, cfg.REDUCED_TRACKING_ERROR_BINS)),
            int(np.digitize(velocity_error, cfg.REDUCED_VELOCITY_ERROR_BINS)),
            int(np.digitize(slave_pdiff, cfg.REDUCED_SLAVE_PRESSURE_DIFF_BINS)),
            int(np.digitize(master_pdiff, cfg.REDUCED_MASTER_PRESSURE_DIFF_BINS)),
        )

    def get_state_dims_reduced(self) -> tuple[int, ...]:
        return (
            len(cfg.REDUCED_TRACKING_ERROR_BINS) + 1,
            len(cfg.REDUCED_VELOCITY_ERROR_BINS) + 1,
            len(cfg.REDUCED_SLAVE_PRESSURE_DIFF_BINS) + 1,
            len(cfg.REDUCED_MASTER_PRESSURE_DIFF_BINS) + 1,
        )
