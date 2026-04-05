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
    ParmsOriginal,
    SimuOriginalProfile,
    SimuOriginalState,
    _rk4_step,
    build_saved_simuoriginal_state,
    saved_environment,
    simuoriginal_derivatives,
)


_TWO_PI = 2.0 * math.pi


def _force_waveform_value(phase: float, waveform: str) -> float:
    waveform = str(waveform).strip().lower()
    if waveform == "sine":
        return math.sin(phase)
    if waveform == "cosine":
        return math.cos(phase)
    if waveform == "square":
        return 1.0 if math.sin(phase) >= 0.0 else -1.0
    if waveform == "multisine":
        return 0.75 * math.sin(phase) + 0.25 * math.sin((2.0 * phase) + 0.35)
    raise ValueError(f"Unknown force waveform: {waveform}")


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
        self.profile = profile or SimuOriginalProfile(fixed_step=self.parms.Ts)
        self.episode_duration = float(
            cfg.EPISODE_DURATION if episode_duration is None else episode_duration
        )
        self.env_switch_time = float(
            cfg.ENV_SWITCH_TIME if env_switch_time is None else env_switch_time
        )
        self.max_steps = max(1, int(round(self.episode_duration / cfg.RL_DT)))
        self.terminate_on_error = bool(terminate_on_error)
        self.rl_dt = float(cfg.RL_DT)
        self.internal_dt = float(self.parms.Ts)
        self.sub_steps = max(1, int(round(self.rl_dt / self.internal_dt)))
        self.x_eq = 0.0
        self.default_reset_options = dict(reset_options or {})

        self._action_table = cfg.V_LEVELS.copy()
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
        self.replica_state = build_saved_simuoriginal_state(self.parms).as_array()
        self.state = np.zeros(self.N_STATE, dtype=np.float64)
        self._history: dict[str, list] | None = None
        self.last_u_v = 0.0
        self.current_env_label = "skin"
        self.current_env_id = 0
        self.invalid_state = False
        self.singularity_time: float | None = None
        self.F_h_nominal = 0.0
        self.F_h_noise = 0.0
        self.F_h = 0.0
        self.F_e = 0.0
        self.a_m_signal = 0.0

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
        self.fh_amp = self.force_amp
        self.fh_bias = self.force_bias
        self.fh_freq = self.force_freq
        self.fh_freq_rad = self.force_freq_rad
        self.fh_phase = self.force_phase
        self.fh_waveform = self.force_waveform

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
        Be, Ke = saved_environment(self.t, self.runtime_profile)
        if abs(Ke - self.profile.skin_Ke) < 1e-12 and abs(Be - self.profile.skin_Be) < 1e-12:
            self.current_env_label = "skin"
            self.current_env_id = 0
        else:
            self.current_env_label = "fat"
            self.current_env_id = 1
        self.Be = float(Be)
        self.Ke = float(Ke)

    def _force_input(self, t: float) -> float:
        phase = (self.force_freq_rad * t) + self.force_phase
        return self.force_bias + (self.force_amp * _force_waveform_value(phase, self.force_waveform))

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
        s = SimuOriginalState.from_array(replica_state)
        volumes = (
            self.parms.V_md + self.parms.A_p * s.xm,
            self.parms.V_md + self.parms.A_p * (self.parms.l_cyl - s.xm),
            self.parms.V_md + self.parms.A_p * s.xs + self.parms.tube_compliance,
            self.parms.V_sd + self.parms.A_p * (self.parms.l_cyl - s.xs) + self.parms.tube_compliance,
        )
        return min(volumes) > 0.0 and bool(np.all(np.isfinite(replica_state)))

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
        self.F_h_nominal = self._force_input(self.t)
        self.F_h_noise = 0.0
        self.F_h = float(self.F_h_nominal)
        self.F_e = float((self.Ke * self.state[self.IX_XS]) + (self.Be * self.state[self.IX_VS]))
        deriv = self._derivative_fn(self.t, self.replica_state)
        self.a_m_signal = float(deriv[2]) if np.all(np.isfinite(deriv)) else 0.0

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

        self.replica_state = build_saved_simuoriginal_state(self.parms).as_array()
        self.state = self._to_env_state(self.replica_state)
        self.t = 0.0
        self.step_count = 0
        self.invalid_state = False
        self.singularity_time = None
        self.last_u_v = 0.0

        self.force_amp = float(cfg.FORCE_INPUT_AMP)
        self.force_bias = 0.0
        self.force_freq = float(cfg.FORCE_INPUT_FREQ)
        self.force_phase = float(cfg.FORCE_INPUT_PHASE)
        self.force_waveform = "sine"
        for key in (
            "force_amp",
            "force_bias",
            "force_freq",
            "force_freq_rad",
            "force_phase",
            "force_waveform",
            "fh_amp",
            "fh_bias",
            "fh_freq",
            "fh_freq_rad",
            "fh_phase",
            "fh_waveform",
        ):
            if key in options:
                setattr(self, key, options[key])
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
            "F_h": [],
            "F_h_nominal": [],
            "F_h_noise": [],
            "a_m_signal": [],
            "F_e": [],
            "u_v": [],
            "x_v": [],
            "env_id": [],
            "env_label": [],
            "pos_error": [],
            "transparency_error": [],
            "reward_track": [],
            "reward_effort": [],
            "reward_transparency": [],
            "reward": [],
            "invalid_state": [],
        }

        return self._get_obs(), self._get_info()

    def _log_step(self, reward: float, track_term: float, effort_term: float, transparency_term: float) -> None:
        if self._history is None:
            return
        x_m_centered, x_s_centered = self.get_centered_positions()
        pos_error = float(self.state[self.IX_XM] - self.state[self.IX_XS])
        transparency_error = float((self.F_e * self.state[self.IX_VM]) - (self.F_h * self.state[self.IX_VS]))
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
        self._history["F_h"].append(self.F_h)
        self._history["F_h_nominal"].append(self.F_h_nominal)
        self._history["F_h_noise"].append(self.F_h_noise)
        self._history["a_m_signal"].append(self.a_m_signal)
        self._history["F_e"].append(self.F_e)
        self._history["u_v"].append(self.last_u_v)
        self._history["x_v"].append(self.state[self.IX_XV])
        self._history["env_id"].append(self.current_env_id)
        self._history["env_label"].append(self.current_env_label)
        self._history["pos_error"].append(pos_error)
        self._history["transparency_error"].append(transparency_error)
        self._history["reward_track"].append(track_term)
        self._history["reward_effort"].append(effort_term)
        self._history["reward_transparency"].append(transparency_term)
        self._history["reward"].append(reward)
        self._history["invalid_state"].append(self.invalid_state)

    def _step_with_voltage(self, u_v: float):
        self.last_u_v = u_v
        self._sync_force_parameters()
        self._update_runtime_profile()

        for _ in range(self.sub_steps):
            if not self._volumes_are_valid(self.replica_state):
                self.invalid_state = True
                self.singularity_time = float(self.t)
                break
            next_state = _rk4_step(self._derivative_fn, self.t, self.replica_state, self.internal_dt)
            self.t += self.internal_dt
            if not self._volumes_are_valid(next_state):
                self.invalid_state = True
                self.singularity_time = float(self.t)
                break
            self.replica_state = next_state

        self.step_count += 1
        self._update_signals()

        pos_error = float(self.state[self.IX_XM] - self.state[self.IX_XS])
        norm_pos_error = float(
            np.clip(pos_error / cfg.MAX_POSITION_ERROR, -cfg.POS_ERR_NORM_CLIP, cfg.POS_ERR_NORM_CLIP)
        )
        transparency_error = float((self.F_e * self.state[self.IX_VM]) - (self.F_h * self.state[self.IX_VS]))
        norm_transparency_error = transparency_error / cfg.MAX_POWER_ERROR
        track_term = cfg.ALPHA_TRACKING * norm_pos_error ** 2
        effort_term = cfg.GAMMA_EFFORT * u_v ** 2
        transparency_term = cfg.BETA_TRANSPARENCY * norm_transparency_error ** 2
        reward = -(track_term + effort_term + transparency_term)

        self._log_step(reward, track_term, effort_term, transparency_term)

        terminated = bool(self.invalid_state)
        if self.terminate_on_error and abs(pos_error) >= cfg.POS_ERROR_FAIL_THRESHOLD:
            terminated = True
        truncated = self.step_count >= self.max_steps
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
        return {
            "time": self.t,
            "u_v": self.last_u_v,
            "F_h": self.F_h,
            "F_h_nominal": self.F_h_nominal,
            "F_h_noise": self.F_h_noise,
            "a_m_signal": self.a_m_signal,
            "F_e": self.F_e,
            "env_id": self.current_env_id,
            "env_label": self.current_env_label,
            "x_m": self.state[self.IX_XM],
            "x_s": self.state[self.IX_XS],
            "x_eq": self.x_eq,
            "x_m_centered": x_m_centered,
            "x_s_centered": x_s_centered,
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
            "invalid_state": self.invalid_state,
            "singularity_time": self.singularity_time,
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
