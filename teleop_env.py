"""
Bilateral Pneumatic Teleoperation — Gymnasium Environment
=========================================================
Full nonlinear dynamics from:
  "Enhanced MRAC Design Using Filtered Regressors for
   Bilateral Pneumatic Teleoperation Control"
  Abed & Daher, IC2AI 2025

Architecture
------------
  Master ←→ Slave via passive pneumatic tubes  (force transparency)
  Servo-valve at slave side provides controlled air  (tracking)

Internal state (12-D, continuous):
  [x_m, v_m, x_s, v_s, P_m1, P_m2, P_s1, P_s2,
   ṁ_L1, ṁ_L2, x_v, v_v]

Observation (10-D, normalised):
  [slave_pos_from_eq, master_pos_from_eq, v_s, v_m,
   P_s1, P_s2, P_m1, P_m2, mdot_L1, mdot_L2]

Discrete RL state (tabular Q-learning):
  same feature set as the observation above

Action:
  Continuous servo-valve voltage u_v ∈ [-5, 5] V.
  (For backward compatibility, integer action indices into V_LEVELS are also accepted.)

Dynamics equations (numbers refer to paper):
  (2)  Master EOM       m_p·ẍ_m = (P_m1 − P_m2)·A_p − β·ẋ_m  − F_h
  (3)  Slave EOM        m_p·ẍ_s = (P_s1 − P_s2)·A_p − (β+B_e)·ẋ_s − K_e·δx_s
  (5)  Pressure ch 1    Ṗ_1 = (RT/V_1)·ṁ_1 − (P_1·A_p/V_1)·ẋ
  (6)  Pressure ch 2    Ṗ_2 = (RT/V_2)·ṁ_2 + (P_2·A_p/V_2)·ẋ
  (7)  Valve flow       ṁ_v = x_v·ρ₀·P_u·c·√(T₀/T)·f(P_d/P_u)   (ISO 6358)
  (8)  Valve spool      ẍ_v + 2·ζ_v·ω·ẋ_v + ω²·x_v = K_v·ω²·u_v
  (10) Tube inertance   dṁ_L/dt = (A_t/L)·(P_up − P_down − ΔP_friction)
  (11) Tube friction    ΔP_friction = 32·μ·ṁ_L·L / (ρ·A_t·D_t²)

Master input semantics supported by this implementation:
  - Reference mode: x_m(t)=r(t) is prescribed and F_h is reconstructed from Eq. (2).
  - Force mode: F_h(t) is prescribed and x_m(t) evolves from the master dynamics.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from . import config as cfg
except ImportError:  # pragma: no cover - direct script execution
    import config as cfg

# Pre-computed constants  (plain Python floats — no numpy overhead)
_RT: float            = cfg.R_GAS * cfg.T_AIR
_SQRT_T_RATIO: float  = math.sqrt(cfg.T0_REF / cfg.T_AIR)
_VALVE_BASE: float    = cfg.RHO0 * cfg.C_SONIC * _SQRT_T_RATIO
_TUBE_INERTANCE: float = cfg.AT / cfg.L_TUBE
_FRICTION_COEFF: float = (32.0 * cfg.MU * cfg.L_TUBE
                          / (cfg.RHO0 * cfg.AT * cfg.DT_TUBE ** 2))
_X_EQ: float          = cfg.L_CYL / 2.0
_OMEGA2: float        = cfg.OMEGA_V ** 2
_2ZETA_OMEGA: float   = 2.0 * cfg.ZETA_V * cfg.OMEGA_V
_KV_OMEGA2: float     = cfg.KV * _OMEGA2
_XMIN: float          = 1e-4
_XMAX: float          = cfg.L_CYL - 1e-4
_PMIN: float          = 1e3
_DT: float            = cfg.DT
_AP: float             = float(cfg.AP)
_MP: float             = float(cfg.MP)
_BETA: float           = float(cfg.BETA)
_VMD: float            = float(cfg.VMD)
_LCYL: float           = float(cfg.L_CYL)
_P_SUPPLY: float       = float(cfg.P_SUPPLY)
_P_ATM: float          = float(cfg.P_ATM)
_B_CRIT: float         = float(cfg.B_CRIT)
_SUB_STEPS: int        = int(cfg.SUB_STEPS)
_TWO_PI: float         = 2.0 * math.pi


# ================================================================== #
#  Fast pure-Python physics kernel  (avoids numpy per-call overhead)  #
# ================================================================== #

def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _iso6358(opening: float, P_u: float, P_d: float) -> float:
    """ISO-6358 mass-flow (Eq 7).  Returns positive flow; caller handles sign."""
    if opening <= 0.0 or P_u <= 0.0:
        return 0.0
    ratio = P_d / P_u
    if ratio < 0.0:
        ratio = 0.0
    elif ratio > 1.0:
        ratio = 1.0
    base = opening * _VALVE_BASE * P_u
    if ratio > _B_CRIT:
        arg = 1.0 - ((ratio - _B_CRIT) / (1.0 - _B_CRIT)) ** 2
        if arg < 0.0:
            arg = 0.0
        return base * math.sqrt(arg)
    return base  # choked


def _bounded_ref_amp(ref_amp: float) -> float:
    max_amp = max(0.0, (_LCYL * 0.5) - 1e-4)
    return _clamp(abs(ref_amp), 0.0, max_amp)


def _master_reference(t: float, ref_amp: float, ref_freq: float, ref_phase: float) -> tuple[float, float, float]:
    amp = _bounded_ref_amp(ref_amp)
    omega = _TWO_PI * ref_freq
    phase = omega * t + ref_phase
    s = math.sin(phase)
    c = math.cos(phase)
    x_ref = _clamp(_X_EQ + amp * s, _XMIN, _XMAX)
    v_ref = amp * omega * c
    a_ref = -amp * omega * omega * s
    return x_ref, v_ref, a_ref


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


def _build_force_noise_components(noise_seed: int | None, n_components: int = 4) -> tuple[tuple[float, float, float], ...]:
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
    base_freq: float,
    noise_std: float,
    noise_components: tuple[tuple[float, float, float], ...] | tuple[()] = (),
) -> float:
    if noise_std <= 0.0 or not noise_components:
        return 0.0

    freq = abs(float(base_freq))
    if freq <= 1e-9:
        freq = float(cfg.FORCE_INPUT_FREQ)

    total = 0.0
    for coeff, freq_mul, phase in noise_components:
        total += coeff * math.sin((_TWO_PI * freq * freq_mul * t) + phase)
    return float(noise_std) * total


def _master_force_components(
    t: float,
    force_amp: float,
    force_bias: float,
    force_freq: float,
    force_phase: float,
    force_waveform: str = "sine",
    force_noise_std: float = 0.0,
    force_noise_components: tuple[tuple[float, float, float], ...] | tuple[()] = (),
) -> tuple[float, float, float]:
    amp = abs(float(force_amp))
    bias = float(force_bias)
    omega = _TWO_PI * force_freq
    phase = (omega * t) + force_phase
    nominal = bias + (amp * _force_waveform_value(phase, force_waveform))
    noise = _force_noise_signal(t, force_freq, force_noise_std, force_noise_components)
    return nominal, noise, nominal + noise


def _master_force_signal(
    t: float,
    force_amp: float,
    force_bias: float,
    force_freq: float,
    force_phase: float,
    force_waveform: str = "sine",
    force_noise_std: float = 0.0,
    force_noise_components: tuple[tuple[float, float, float], ...] | tuple[()] = (),
) -> float:
    _, _, total = _master_force_components(
        t,
        force_amp,
        force_bias,
        force_freq,
        force_phase,
        force_waveform,
        force_noise_std,
        force_noise_components,
    )
    return total


def _derivatives(s: list[float], t: float, u_v: float,
                 master_input_mode: str,
                 ref_amp: float, ref_freq: float, ref_phase: float,
                 force_amp: float, force_bias: float, force_freq: float, force_phase: float, force_waveform: str,
                 force_noise_std: float, force_noise_components: tuple[tuple[float, float, float], ...] | tuple[()],
                 Be: float, Ke: float) -> list[float]:
    """12-D derivative vector — pure Python scalars, no numpy."""
    if master_input_mode == cfg.MASTER_INPUT_REFERENCE:
        x_m, v_m, _ = _master_reference(t, ref_amp, ref_freq, ref_phase)
        dx_m = 0.0
        a_m = 0.0
    elif master_input_mode == cfg.MASTER_INPUT_FORCE:
        x_m = _clamp(s[0], _XMIN, _XMAX)
        v_m = s[1]
        F_h = _master_force_signal(
            t,
            force_amp,
            force_bias,
            force_freq,
            force_phase,
            force_waveform,
            force_noise_std,
            force_noise_components,
        )
        dx_m = v_m
        a_m = ((s[4] - s[5]) * _AP - _BETA * v_m - F_h) / _MP
    else:
        raise ValueError(f"Unknown master_input_mode: {master_input_mode}")

    x_s  = _clamp(s[2], _XMIN, _XMAX)
    v_s  = s[3]
    P_m1 = s[4] if s[4] > _PMIN else _PMIN
    P_m2 = s[5] if s[5] > _PMIN else _PMIN
    P_s1 = s[6] if s[6] > _PMIN else _PMIN
    P_s2 = s[7] if s[7] > _PMIN else _PMIN
    mdot_L1 = s[8]
    mdot_L2 = s[9]
    x_v  = s[10]
    v_v  = s[11]

    # Chamber volumes
    V_m1 = _VMD + x_m * _AP
    V_m2 = _VMD + (_LCYL - x_m) * _AP
    V_s1 = _VMD + x_s * _AP
    V_s2 = _VMD + (_LCYL - x_s) * _AP

    # (A) Equations of motion  (Eq 3 for slave; master is prescribed reference)
    delta_xs = x_s - _X_EQ
    a_s = ((P_s1 - P_s2) * _AP - (_BETA + Be) * v_s - Ke * delta_xs) / _MP

    # (B) Master pressure  (Eqs 5, 6)
    dP_m1 = (_RT / V_m1) * (-mdot_L1) - (P_m1 * _AP / V_m1) * v_m
    dP_m2 = (_RT / V_m2) * (-mdot_L2) + (P_m2 * _AP / V_m2) * v_m

    # (C) Servo-valve spool  (Eq 8)
    dxv = v_v
    dvv = _KV_OMEGA2 * u_v - _2ZETA_OMEGA * v_v - _OMEGA2 * x_v

    # Valve flow (Eq 7) — 4/3 proportional
    if x_v > 0.0:
        mdot_v_s1 =  _iso6358( x_v, _P_SUPPLY, P_s1)
        mdot_v_s2 = -_iso6358( x_v, P_s2, _P_ATM)
    elif x_v < 0.0:
        mdot_v_s1 = -_iso6358(-x_v, P_s1, _P_ATM)
        mdot_v_s2 =  _iso6358(-x_v, _P_SUPPLY, P_s2)
    else:
        mdot_v_s1 = 0.0
        mdot_v_s2 = 0.0

    # Slave pressure  (tubes + valve)
    mdot_s1 = mdot_L2 + mdot_v_s1
    mdot_s2 = mdot_L1 + mdot_v_s2
    dP_s1 = (_RT / V_s1) * mdot_s1 - (P_s1 * _AP / V_s1) * v_s
    dP_s2 = (_RT / V_s2) * mdot_s2 + (P_s2 * _AP / V_s2) * v_s

    # (D) Tube inertance + friction  (Eqs 10, 11)
    d_mdot_L1 = _TUBE_INERTANCE * (P_m1 - P_s2 - _FRICTION_COEFF * mdot_L1)
    d_mdot_L2 = _TUBE_INERTANCE * (P_m2 - P_s1 - _FRICTION_COEFF * mdot_L2)

    return [dx_m, a_m, v_s, a_s,
            dP_m1, dP_m2, dP_s1, dP_s2,
            d_mdot_L1, d_mdot_L2, dxv, dvv]


def _rk4_substeps(state: np.ndarray, t0: float, u_v: float,
                   master_input_mode: str,
                   ref_amp: float, ref_freq: float, ref_phase: float,
                   force_amp: float, force_bias: float, force_freq: float, force_phase: float, force_waveform: str,
                   force_noise_std: float, force_noise_components: tuple[tuple[float, float, float], ...] | tuple[()],
                   Be: float, Ke: float) -> tuple[np.ndarray, float]:
    """Run SUB_STEPS RK4 integration steps.  Returns (new_state, new_t)."""
    # Convert to plain-Python list for speed
    s = state.tolist()
    dt = _DT
    args = (
        u_v, master_input_mode,
        ref_amp, ref_freq, ref_phase,
        force_amp, force_bias, force_freq, force_phase, force_waveform, force_noise_std, force_noise_components,
        Be, Ke,
    )
    t = t0
    if master_input_mode == cfg.MASTER_INPUT_REFERENCE:
        x_ref, v_ref, _ = _master_reference(t, ref_amp, ref_freq, ref_phase)
        s[0] = x_ref
        s[1] = v_ref

    for _ in range(_SUB_STEPS):
        k1 = _derivatives(s, t, *args)

        s2 = [s[i] + 0.5 * dt * k1[i] for i in range(12)]
        k2 = _derivatives(s2, t + 0.5 * dt, *args)

        s3 = [s[i] + 0.5 * dt * k2[i] for i in range(12)]
        k3 = _derivatives(s3, t + 0.5 * dt, *args)

        s4 = [s[i] + dt * k3[i] for i in range(12)]
        k4 = _derivatives(s4, t + dt, *args)

        s = [s[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
             for i in range(12)]

        # Physical constraints
        s[0] = _clamp(s[0], _XMIN, _XMAX)   # x_m
        s[2] = _clamp(s[2], _XMIN, _XMAX)   # x_s
        for j in (4, 5, 6, 7):               # pressures
            if s[j] < _PMIN:
                s[j] = _PMIN
        s[10] = _clamp(s[10], -1.0, 1.0)    # x_v

        # Velocity damping at stroke limits
        if s[0] <= _XMIN and s[1] < 0.0:
            s[1] = 0.0
        if s[0] >= _XMAX and s[1] > 0.0:
            s[1] = 0.0
        if s[2] <= _XMIN and s[3] < 0.0:
            s[3] = 0.0
        if s[2] >= _XMAX and s[3] > 0.0:
            s[3] = 0.0

        t += dt
        if master_input_mode == cfg.MASTER_INPUT_FORCE:
            if s[0] <= _XMIN and s[1] < 0.0:
                s[1] = 0.0
            if s[0] >= _XMAX and s[1] > 0.0:
                s[1] = 0.0
        else:
            x_ref, v_ref, _ = _master_reference(t, ref_amp, ref_freq, ref_phase)
            s[0] = x_ref
            s[1] = v_ref

    return np.array(s, dtype=np.float64), t


class TeleopEnv(gym.Env):
    """Gymnasium env for bilateral pneumatic teleoperation with RL valve control."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # -------------------------------------------------------------- #
    # State-vector indices                                            #
    # -------------------------------------------------------------- #
    IX_XM, IX_VM   = 0, 1       # master position, velocity
    IX_XS, IX_VS   = 2, 3       # slave  position, velocity
    IX_PM1, IX_PM2 = 4, 5       # master chamber pressures
    IX_PS1, IX_PS2 = 6, 7       # slave  chamber pressures
    IX_ML1, IX_ML2 = 8, 9       # tube mass-flow rates
    IX_XV, IX_VV   = 10, 11     # valve spool position, velocity
    N_STATE = 12

    # -------------------------------------------------------------- #
    #  __init__                                                       #
    # -------------------------------------------------------------- #
    def __init__(
        self,
        render_mode: str | None = None,
        env_mode: str | None = None,
        master_input_mode: str | None = None,
        episode_duration: float | None = None,
        env_switch_time: float | None = None,
        terminate_on_error: bool = True,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.env_mode = env_mode or cfg.ENV_MODE_CONSTANT
        self.master_input_mode = master_input_mode or cfg.DEFAULT_MASTER_INPUT_MODE
        self.episode_duration = float(
            cfg.EPISODE_DURATION if episode_duration is None else episode_duration
        )
        self.env_switch_time = float(
            cfg.ENV_SWITCH_TIME if env_switch_time is None else env_switch_time
        )
        self.max_steps = max(1, int(self.episode_duration / cfg.RL_DT))
        self.terminate_on_error = bool(terminate_on_error)

        self._action_table = cfg.V_LEVELS.copy()
        self._u_min = float(self._action_table.min())
        self._u_max = float(self._action_table.max())
        self.action_space = spaces.Box(
            low=np.array([self._u_min], dtype=np.float32),
            high=np.array([self._u_max], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation bounds after normalisation (approximate [-1, 1] range).
        low  = -np.ones(10, dtype=np.float32) * 2.0
        high =  np.ones(10, dtype=np.float32) * 2.0
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self._history: dict[str, list] | None = None
        self.last_u_v: float = 0.0
        self.current_env_label: str = "skin"
        self.current_env_id: int = 0
        self.F_h_nominal: float = 0.0
        self.F_h_noise: float = 0.0

    def _sync_master_input_parameters(self) -> None:
        if self.master_input_mode == cfg.MASTER_INPUT_REFERENCE:
            self.ref_amp = _bounded_ref_amp(float(getattr(self, "fh_amp", self.ref_amp)))
            self.ref_freq = float(getattr(self, "fh_freq", self.ref_freq))
            self.ref_phase = float(getattr(self, "fh_phase", self.ref_phase))
            self.fh_amp = self.ref_amp
            self.fh_freq = self.ref_freq
            self.fh_phase = self.ref_phase
            return

        if self.master_input_mode == cfg.MASTER_INPUT_FORCE:
            self.force_amp = abs(float(getattr(self, "force_amp", getattr(self, "fh_amp", self.force_amp))))
            self.force_bias = float(getattr(self, "force_bias", getattr(self, "fh_bias", self.force_bias)))
            self.force_freq = float(getattr(self, "force_freq", getattr(self, "fh_freq", self.force_freq)))
            self.force_phase = float(getattr(self, "force_phase", getattr(self, "fh_phase", self.force_phase)))
            self.force_waveform = str(getattr(self, "force_waveform", getattr(self, "fh_waveform", "sine"))).strip().lower()
            self.force_noise_std = abs(float(getattr(self, "force_noise_std", getattr(self, "fh_noise_std", 0.0))))
            self.force_noise_seed = int(getattr(self, "force_noise_seed", getattr(self, "fh_noise_seed", 0)))
            self.force_noise_components = (
                _build_force_noise_components(self.force_noise_seed) if self.force_noise_std > 0.0 else ()
            )
            self.fh_amp = self.force_amp
            self.fh_bias = self.force_bias
            self.fh_freq = self.force_freq
            self.fh_phase = self.force_phase
            self.fh_waveform = self.force_waveform
            self.fh_noise_std = self.force_noise_std
            self.fh_noise_seed = self.force_noise_seed
            return

        raise ValueError(f"Unknown master_input_mode: {self.master_input_mode}")

    def _set_environment(self, env_label: str) -> None:
        if env_label == "skin":
            self.Be, self.Ke = cfg.SKIN_BE, cfg.SKIN_KE
            self.current_env_label = "skin"
            self.current_env_id = 0
            return
        if env_label == "fat":
            self.Be, self.Ke = cfg.FAT_BE, cfg.FAT_KE
            self.current_env_label = "fat"
            self.current_env_id = 1
            return
        raise ValueError(f"Unknown environment label: {env_label}")

    def _update_environment_mode(self) -> None:
        if self.env_mode == cfg.ENV_MODE_CONSTANT:
            self._set_environment("skin")
            return
        if self.env_mode == cfg.ENV_MODE_CHANGING:
            if self.t < self.env_switch_time:
                self._set_environment("skin")
            else:
                self._set_environment("fat")
            return
        raise ValueError(f"Unknown env_mode: {self.env_mode}")

    def get_equilibrium_position(self) -> float:
        """Return the cylinder midpoint used as the displacement reference."""
        return _X_EQ

    def get_centered_positions(self) -> tuple[float, float]:
        """Return master/slave displacement from the equilibrium midpoint."""
        return (
            float(self.state[self.IX_XM] - _X_EQ),
            float(self.state[self.IX_XS] - _X_EQ),
        )

    # -------------------------------------------------------------- #
    #  reset                                                          #
    # -------------------------------------------------------------- #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = dict(options or {})

        # --- Internal state: 12 variables ---------------------------
        self.state = np.zeros(self.N_STATE, dtype=np.float64)
        self.state[self.IX_XM]  = _X_EQ          # master at mid-stroke
        self.state[self.IX_XS]  = _X_EQ          # slave  at mid-stroke
        self.state[self.IX_PM1] = cfg.P_ATM       # all chambers at 1 atm
        self.state[self.IX_PM2] = cfg.P_ATM
        self.state[self.IX_PS1] = cfg.P_ATM
        self.state[self.IX_PS2] = cfg.P_ATM

        # --- Time ---------------------------------------------------
        self.t = 0.0
        self.step_count = 0

        # --- Master input parameters -------------------------------------
        self.ref_amp   = float(cfg.REF_POS_AMP)
        self.ref_freq  = float(cfg.REF_POS_FREQ)
        self.ref_phase = float(cfg.REF_POS_PHASE)
        self.force_amp = float(cfg.FORCE_INPUT_AMP)
        self.force_bias = 0.0
        self.force_freq = float(cfg.FORCE_INPUT_FREQ)
        self.force_phase = float(cfg.FORCE_INPUT_PHASE)
        self.force_waveform = "sine"
        self.force_noise_std = 0.0
        self.force_noise_seed = 0
        self.force_noise_components: tuple[tuple[float, float, float], ...] | tuple[()] = ()
        for key in (
            "fh_amp",
            "fh_bias",
            "fh_freq",
            "fh_phase",
            "fh_waveform",
            "fh_noise_std",
            "fh_noise_seed",
            "force_amp",
            "force_bias",
            "force_freq",
            "force_phase",
            "force_waveform",
            "force_noise_std",
            "force_noise_seed",
            "ref_amp",
            "ref_freq",
            "ref_phase",
        ):
            if key in options:
                setattr(self, key, options[key])
        self._sync_master_input_parameters()

        # --- Environment mode ----------------------------------------
        self._update_environment_mode()

        # --- Initialize master state --------------------------------
        if self.master_input_mode == cfg.MASTER_INPUT_REFERENCE:
            x_ref0, v_ref0, _ = _master_reference(self.t, self.ref_amp, self.ref_freq, self.ref_phase)
            self.state[self.IX_XM] = x_ref0
            self.state[self.IX_VM] = v_ref0
        else:
            self.state[self.IX_XM] = _X_EQ
            self.state[self.IX_VM] = 0.0

        # --- Forces (persisted for observation / logging) -----------
        if self.master_input_mode == cfg.MASTER_INPUT_FORCE:
            self.F_h_nominal, self.F_h_noise, self.F_h = _master_force_components(
                self.t,
                self.force_amp,
                self.force_bias,
                self.force_freq,
                self.force_phase,
                self.force_waveform,
                self.force_noise_std,
                self.force_noise_components,
            )
        else:
            self.F_h_nominal = 0.0
            self.F_h_noise = 0.0
            self.F_h = 0.0
        self.a_m_signal = 0.0
        self.F_e = 0.0
        self.last_u_v = 0.0

        # --- Episode history ----------------------------------------
        self._history = {
            "time": [], "x_m": [], "x_s": [],
            "x_m_centered": [], "x_s_centered": [],
            "v_m": [], "v_s": [],
            "P_m1": [], "P_m2": [], "P_s1": [], "P_s2": [],
            "F_h": [], "F_h_nominal": [], "F_h_noise": [], "a_m_signal": [], "F_e": [], "u_v": [], "x_v": [],
            "env_id": [], "env_label": [],
            "pos_error": [], "transparency_error": [],
            "reward_track": [], "reward_effort": [], "reward_transparency": [],
            "reward": [],
        }

        return self._get_obs(), self._get_info()

    # ============================================================== #
    #  DYNAMICS  — delegates to module-level pure-Python functions     #
    # ============================================================== #

    # -------------------------------------------------------------- #
    #  Gym step                                                       #
    # -------------------------------------------------------------- #
    def _step_with_voltage(self, u_v: float):
        self.last_u_v = u_v
        self._update_environment_mode()
        self._sync_master_input_parameters()

        # ── Physics integration (pure-Python fast kernel) ────────
        self.state, self.t = _rk4_substeps(
            self.state, self.t, u_v,
            self.master_input_mode,
            self.ref_amp, self.ref_freq, self.ref_phase,
            self.force_amp, self.force_bias, self.force_freq, self.force_phase, self.force_waveform,
            self.force_noise_std, self.force_noise_components,
            self.Be, self.Ke,
        )

        self.step_count += 1

        if self.master_input_mode == cfg.MASTER_INPUT_REFERENCE:
            # Enforce exact master reference values at RL sample time.
            x_ref, v_ref, a_ref = _master_reference(self.t, self.ref_amp, self.ref_freq, self.ref_phase)
            self.state[self.IX_XM] = x_ref
            self.state[self.IX_VM] = v_ref

        # --- Extract quantities ----------------------------------
        x_m, v_m = self.state[self.IX_XM], self.state[self.IX_VM]
        x_s, v_s = self.state[self.IX_XS], self.state[self.IX_VS]
        x_m_centered, x_s_centered = self.get_centered_positions()
        pos_error = x_m - x_s

        # Update stored signal/forces using the active master-input mode.
        if self.master_input_mode == cfg.MASTER_INPUT_REFERENCE:
            self.a_m_signal = float(a_ref)
            self.F_h_nominal = 0.0
            self.F_h_noise = 0.0
            self.F_h = float(
                (self.state[self.IX_PM1] - self.state[self.IX_PM2]) * _AP
                - _MP * self.a_m_signal
                - _BETA * v_m
            )
        else:
            self.F_h_nominal, self.F_h_noise, self.F_h = _master_force_components(
                self.t,
                self.force_amp,
                self.force_bias,
                self.force_freq,
                self.force_phase,
                self.force_waveform,
                self.force_noise_std,
                self.force_noise_components,
            )
            self.F_h = float(self.F_h)
            self.a_m_signal = float(((self.state[self.IX_PM1] - self.state[self.IX_PM2]) * _AP - _BETA * v_m - self.F_h) / _MP)
        delta_xs = x_s - _X_EQ
        self.F_e = self.Ke * delta_xs + self.Be * v_s

        # --- Reward -----------------------------------------------
        # --- Tracking error (normalized) ---
        pos_error = (x_m - x_s)
        norm_pos_error = pos_error / cfg.MAX_POSITION_ERROR
        norm_pos_error = float(np.clip(
            norm_pos_error, -cfg.POS_ERR_NORM_CLIP, cfg.POS_ERR_NORM_CLIP
        ))

        # --- Transparency error (power mismatch) ---
        transparency_error = self.F_e * v_m - self.F_h * v_s
        norm_transparency_error = transparency_error / cfg.MAX_POWER_ERROR

        track_term = cfg.ALPHA_TRACKING * norm_pos_error**2
        effort_term = cfg.GAMMA_EFFORT * u_v**2
        transparency_term = cfg.BETA_TRANSPARENCY * norm_transparency_error**2

        # --- Reward ---
        reward = -(track_term + effort_term + transparency_term)


        # --- History (one entry per RL step) ----------------------
        if self._history is not None:
            self._history["time"].append(self.t)
            self._history["x_m"].append(x_m)
            self._history["x_s"].append(x_s)
            self._history["x_m_centered"].append(x_m_centered)
            self._history["x_s_centered"].append(x_s_centered)
            self._history["v_m"].append(v_m)
            self._history["v_s"].append(v_s)
            self._history["P_m1"].append(self.state[self.IX_PM1])
            self._history["P_m2"].append(self.state[self.IX_PM2])
            self._history["P_s1"].append(self.state[self.IX_PS1])
            self._history["P_s2"].append(self.state[self.IX_PS2])
            self._history["F_h"].append(self.F_h)
            self._history["F_h_nominal"].append(self.F_h_nominal)
            self._history["F_h_noise"].append(self.F_h_noise)
            self._history["a_m_signal"].append(self.a_m_signal)
            self._history["F_e"].append(self.F_e)
            self._history["u_v"].append(u_v)
            self._history["x_v"].append(self.state[self.IX_XV])
            self._history["env_id"].append(self.current_env_id)
            self._history["env_label"].append(self.current_env_label)
            self._history["pos_error"].append(pos_error)
            self._history["transparency_error"].append(transparency_error)
            self._history["reward_track"].append(track_term)
            self._history["reward_effort"].append(effort_term)
            self._history["reward_transparency"].append(transparency_term)
            self._history["reward"].append(reward)

        terminated = bool(self.terminate_on_error and (abs(pos_error) >= cfg.POS_ERROR_FAIL_THRESHOLD))
        truncated  = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _action_to_voltage(self, action: int | float | np.ndarray) -> float:
        """
        Accept either:
          - integer action index into V_LEVELS (legacy tabular RL), or
          - scalar/shape-(1,) continuous voltage in [u_min, u_max].
        """
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
        u_v = self._action_to_voltage(action)
        return self._step_with_voltage(u_v)

    def step_voltage(self, u_v: float):
        u_v = float(np.clip(u_v, self._u_min, self._u_max))
        return self._step_with_voltage(u_v)

    # -------------------------------------------------------------- #
    #  render                                                         #
    # -------------------------------------------------------------- #
    def render(self):
        """Return episode history dict (for external plotting)."""
        return self._history

    # -------------------------------------------------------------- #
    #  Helpers                                                        #
    # -------------------------------------------------------------- #
    def _get_obs(self) -> np.ndarray:
        """
        10-D normalised observation:
        [slave_pos_from_eq, master_pos_from_eq, v_s, v_m,
         P_s1, P_s2, P_m1, P_m2, mdot_L1, mdot_L2]

        Each feature is divided by its physical scale so the vector
        lives in approximately [-1, 1].
        """
        return np.array([
            (self.state[self.IX_XS] - _X_EQ)  / cfg.OBS_SCALE_POS,
            (self.state[self.IX_XM] - _X_EQ)  / cfg.OBS_SCALE_POS,
            self.state[self.IX_VS]             / cfg.OBS_SCALE_VEL,
            self.state[self.IX_VM]             / cfg.OBS_SCALE_VEL,
            self.state[self.IX_PS1]            / cfg.OBS_SCALE_PRESSURE,
            self.state[self.IX_PS2]            / cfg.OBS_SCALE_PRESSURE,
            self.state[self.IX_PM1]            / cfg.OBS_SCALE_PRESSURE,
            self.state[self.IX_PM2]            / cfg.OBS_SCALE_PRESSURE,
            self.state[self.IX_ML1]            / cfg.OBS_SCALE_FLOW,
            self.state[self.IX_ML2]            / cfg.OBS_SCALE_FLOW,
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        x_m_centered, x_s_centered = self.get_centered_positions()
        return {
            "time": self.t,
            "u_v":  self.last_u_v,
            "F_h":  self.F_h,
            "F_h_nominal": self.F_h_nominal,
            "F_h_noise": self.F_h_noise,
            "a_m_signal": self.a_m_signal,
            "F_e":  self.F_e,
            "env_id": self.current_env_id,
            "env_label": self.current_env_label,
            "x_m":  self.state[self.IX_XM],
            "x_s":  self.state[self.IX_XS],
            "x_eq": _X_EQ,
            "x_m_centered": x_m_centered,
            "x_s_centered": x_s_centered,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "episode_duration": self.episode_duration,
            "env_switch_time": self.env_switch_time,
            "terminate_on_error": self.terminate_on_error,
            "master_input_mode": self.master_input_mode,
            "force_bias": getattr(self, "force_bias", None),
            "force_freq": getattr(self, "force_freq", None),
            "force_waveform": getattr(self, "force_waveform", None),
            "force_noise_std": getattr(self, "force_noise_std", None),
            "force_noise_seed": getattr(self, "force_noise_seed", None),
        }

    # ---------- Q-table helpers ----------

    def discretise_obs(self, obs: np.ndarray) -> tuple[int, ...]:
        """Map continuous/plant state to discrete tabular RL state."""
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
        """Number of discrete bins per dimension (for Q-table shape)."""
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

    # ---------- Reduced 4-D Q-table helpers ----------

    def discretise_obs_reduced(self, obs: np.ndarray) -> tuple[int, ...]:
        """
        Map the 10-D normalised observation to a reduced 4-D discrete state:
          (tracking_error, velocity_error, slave_pdiff, master_pdiff)
        Un-normalises the relevant features before digitising.
        """
        slave_pos  = obs[0] * cfg.OBS_SCALE_POS
        master_pos = obs[1] * cfg.OBS_SCALE_POS
        v_s        = obs[2] * cfg.OBS_SCALE_VEL
        v_m        = obs[3] * cfg.OBS_SCALE_VEL
        P_s1       = obs[4] * cfg.OBS_SCALE_PRESSURE
        P_s2       = obs[5] * cfg.OBS_SCALE_PRESSURE
        P_m1       = obs[6] * cfg.OBS_SCALE_PRESSURE
        P_m2       = obs[7] * cfg.OBS_SCALE_PRESSURE

        tracking_error = master_pos - slave_pos
        velocity_error = v_m - v_s
        slave_pdiff    = P_s1 - P_s2
        master_pdiff   = P_m1 - P_m2

        return (
            int(np.digitize(tracking_error, cfg.REDUCED_TRACKING_ERROR_BINS)),
            int(np.digitize(velocity_error, cfg.REDUCED_VELOCITY_ERROR_BINS)),
            int(np.digitize(slave_pdiff,    cfg.REDUCED_SLAVE_PRESSURE_DIFF_BINS)),
            int(np.digitize(master_pdiff,   cfg.REDUCED_MASTER_PRESSURE_DIFF_BINS)),
        )

    def get_state_dims_reduced(self) -> tuple[int, ...]:
        """Number of discrete bins per dimension for the reduced 4-D state."""
        return (
            len(cfg.REDUCED_TRACKING_ERROR_BINS) + 1,
            len(cfg.REDUCED_VELOCITY_ERROR_BINS) + 1,
            len(cfg.REDUCED_SLAVE_PRESSURE_DIFF_BINS) + 1,
            len(cfg.REDUCED_MASTER_PRESSURE_DIFF_BINS) + 1,
        )
