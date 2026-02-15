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

Observation (4-D, for plotting/diagnostics):
  [pos_error, vel_error, F_h, F_e]

Discrete RL state (tabular Q-learning):
  [pos_error, vel_error, (P_m1-P_m2), (P_s1-P_s2), mdot_L1, mdot_L2, x_v]

Action:
  Discrete index → servo-valve voltage u_v ∈ V_LEVELS.

Dynamics equations (numbers refer to paper):
  (2)  Master EOM       m_p·ẍ_m = (P_m1 − P_m2)·A_p − β·ẋ_m  − F_h
  (3)  Slave EOM        m_p·ẍ_s = (P_s1 − P_s2)·A_p − (β+B_e)·ẋ_s − K_e·δx_s
  (5)  Pressure ch 1    Ṗ_1 = (RT/V_1)·ṁ_1 − (P_1·A_p/V_1)·ẋ
  (6)  Pressure ch 2    Ṗ_2 = (RT/V_2)·ṁ_2 + (P_2·A_p/V_2)·ẋ
  (7)  Valve flow       ṁ_v = x_v·ρ₀·P_u·c·√(T₀/T)·f(P_d/P_u)   (ISO 6358)
  (8)  Valve spool      ẍ_v + 2·ζ_v·ω·ẋ_v + ω²·x_v = K_v·ω²·u_v
  (10) Tube inertance   dṁ_L/dt = (A_t/L)·(P_up − P_down − ΔP_friction)
  (11) Tube friction    ΔP_friction = 32·μ·ṁ_L·L / (ρ·A_t·D_t²)
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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


def _derivatives(s: list[float], t: float, u_v: float,
                 fh_amp: float, fh_freq: float, fh_phase: float,
                 Be: float, Ke: float) -> list[float]:
    """12-D derivative vector — pure Python scalars, no numpy."""
    x_m  = _clamp(s[0], _XMIN, _XMAX)
    v_m  = s[1]
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

    # Human force
    F_h = fh_amp * math.sin(_TWO_PI * fh_freq * t + fh_phase)

    # (A) Equations of motion  (Eqs 2, 3)
    a_m = ((P_m1 - P_m2) * _AP - _BETA * v_m - F_h) / _MP
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

    return [v_m, a_m, v_s, a_s,
            dP_m1, dP_m2, dP_s1, dP_s2,
            d_mdot_L1, d_mdot_L2, dxv, dvv]


def _rk4_substeps(state: np.ndarray, t0: float, u_v: float,
                   fh_amp: float, fh_freq: float, fh_phase: float,
                   Be: float, Ke: float) -> tuple[np.ndarray, float]:
    """Run SUB_STEPS RK4 integration steps.  Returns (new_state, new_t)."""
    # Convert to plain-Python list for speed
    s = state.tolist()
    dt = _DT
    args = (u_v, fh_amp, fh_freq, fh_phase, Be, Ke)
    t = t0

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
    def __init__(self, render_mode: str | None = None, env_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.env_mode = env_mode or cfg.ENV_MODE_CONSTANT

        self.action_space = spaces.Discrete(cfg.N_ACTIONS)

        low  = np.array([-cfg.L_CYL, -1.0, -20.0, -50.0], dtype=np.float32)
        high = np.array([ cfg.L_CYL,  1.0,  20.0,  50.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self._action_table = cfg.V_LEVELS.copy()
        self._history: dict[str, list] | None = None
        self.last_u_v: float = 0.0
        self.current_env_label: str = "skin"
        self.current_env_id: int = 0

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
            if self.t < cfg.ENV_SWITCH_TIME:
                self._set_environment("skin")
            else:
                self._set_environment("fat")
            return
        raise ValueError(f"Unknown env_mode: {self.env_mode}")

    # -------------------------------------------------------------- #
    #  reset                                                          #
    # -------------------------------------------------------------- #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

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

        # --- Constant force profile (deterministic training setup) ---
        self.fh_amp   = cfg.FH_AMP
        self.fh_freq  = cfg.FH_FREQ
        self.fh_phase = 0.0

        # --- Environment mode ----------------------------------------
        self._update_environment_mode()

        # --- Forces (persisted for observation / logging) -----------
        self.F_h = 0.0
        self.F_e = 0.0
        self.last_u_v = 0.0

        # --- Episode history ----------------------------------------
        self._history = {
            "time": [], "x_m": [], "x_s": [],
            "v_m": [], "v_s": [],
            "P_m1": [], "P_m2": [], "P_s1": [], "P_s2": [],
            "F_h": [], "F_e": [], "u_v": [], "x_v": [],
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
    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        u_v = float(self._action_table[action])
        self.last_u_v = u_v
        self._update_environment_mode()

        # ── Physics integration (pure-Python fast kernel) ────────
        self.state, self.t = _rk4_substeps(
            self.state, self.t, u_v,
            self.fh_amp, self.fh_freq, self.fh_phase,
            self.Be, self.Ke,
        )

        self.step_count += 1

        # --- Extract quantities ----------------------------------
        x_m, v_m = self.state[self.IX_XM], self.state[self.IX_VM]
        x_s, v_s = self.state[self.IX_XS], self.state[self.IX_VS]
        pos_error = x_m - x_s

        # Update stored forces for observation
        self.F_h = self.fh_amp * np.sin(
            2 * np.pi * self.fh_freq * self.t + self.fh_phase)
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

        # Optional clipping
        reward = float(np.clip(reward, -cfg.REWARD_CLIP, cfg.REWARD_CLIP))


        # --- History (one entry per RL step) ----------------------
        if self._history is not None:
            self._history["time"].append(self.t)
            self._history["x_m"].append(x_m)
            self._history["x_s"].append(x_s)
            self._history["v_m"].append(v_m)
            self._history["v_s"].append(v_s)
            self._history["P_m1"].append(self.state[self.IX_PM1])
            self._history["P_m2"].append(self.state[self.IX_PM2])
            self._history["P_s1"].append(self.state[self.IX_PS1])
            self._history["P_s2"].append(self.state[self.IX_PS2])
            self._history["F_h"].append(self.F_h)
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

        terminated = abs(pos_error) >= cfg.POS_ERROR_FAIL_THRESHOLD
        truncated  = self.step_count >= cfg.MAX_STEPS

        return self._get_obs(), reward, terminated, truncated, self._get_info()

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
        """4-D continuous observation: [pos_error, vel_error, F_h, F_e]."""
        return np.array([
            self.state[self.IX_XM] - self.state[self.IX_XS],
            self.state[self.IX_VM] - self.state[self.IX_VS],
            self.F_h,
            self.F_e,
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "time": self.t,
            "u_v":  self.last_u_v,
            "F_h":  self.F_h,
            "F_e":  self.F_e,
            "env_id": self.current_env_id,
            "env_label": self.current_env_label,
            "x_m":  self.state[self.IX_XM],
            "x_s":  self.state[self.IX_XS],
            "step_count": self.step_count,
        }

    # ---------- Q-table helpers ----------

    def discretise_obs(self, obs: np.ndarray) -> tuple[int, ...]:
        """Map continuous/plant state to discrete tabular RL state."""
        pm_diff = self.state[self.IX_PM1] - self.state[self.IX_PM2]
        ps_diff = self.state[self.IX_PS1] - self.state[self.IX_PS2]
        mdot_l1 = self.state[self.IX_ML1]
        mdot_l2 = self.state[self.IX_ML2]
        spool_x = self.state[self.IX_XV]

        return (
            int(np.digitize(obs[0], cfg.POS_ERROR_BINS)),
            int(np.digitize(obs[1], cfg.VEL_ERROR_BINS)),
            int(np.digitize(pm_diff, cfg.PM_DIFF_BINS)),
            int(np.digitize(ps_diff, cfg.PS_DIFF_BINS)),
            int(np.digitize(mdot_l1, cfg.FLOW_BINS)),
            int(np.digitize(mdot_l2, cfg.FLOW_BINS)),
            int(np.digitize(spool_x, cfg.SPOOL_POS_BINS)),
        )

    def get_state_dims(self) -> tuple[int, ...]:
        """Number of discrete bins per dimension (for Q-table shape)."""
        return (
            len(cfg.POS_ERROR_BINS) + 1,
            len(cfg.VEL_ERROR_BINS) + 1,
            len(cfg.PM_DIFF_BINS)   + 1,
            len(cfg.PS_DIFF_BINS)   + 1,
            len(cfg.FLOW_BINS)      + 1,
            len(cfg.FLOW_BINS)      + 1,
            len(cfg.SPOOL_POS_BINS) + 1,
        )
