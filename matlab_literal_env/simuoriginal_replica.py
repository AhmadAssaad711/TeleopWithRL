from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import csv
import math

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class ParmsOriginal:
    # Hand
    m_h: float = 0.0 * 11.6
    B_h: float = 0.0 * 17.26
    k_h: float = 0.0 * 243.2

    # General
    R: float = 287.0
    rho0: float = 1.204
    T: float = 273.15 + 20.0
    P_s: float = 3e5
    P_atm: float = 1.013e5

    # Tube
    D_t: float = 4e-3
    L_t: float = 10.0
    mui: float = 1.813e-5

    # Cylinder
    m_p: float = 0.25
    beta: float = 0.33 * 35.0
    l_cyl: float = 0.275
    A_p: float = 4.2072e-4
    V_md: float = 2.4797e-5
    V_sd: float = 2.4797e-5

    # Valve
    C_v: float = 4.5e-9
    b_v: float = 0.21
    omega_v: float = 150.0
    K_v: float = 1.0 / 5.0
    zeta_v: float = 0.7

    # Solver
    Ts: float = 0.001

    @property
    def A_t(self) -> float:
        return math.pi * self.D_t**2 / 4.0

    @property
    def P_md(self) -> float:
        return self.R * self.T * self.rho0

    @property
    def tube_damping(self) -> float:
        return 32.0 * self.mui / (self.rho0 * self.D_t**2)

    @property
    def tube_compliance(self) -> float:
        return self.A_t * self.L_t


@dataclass(frozen=True)
class SimuOriginalProfile:
    stop_time: float = 180.0
    solver_name: str = "ode4"
    fixed_step: float = 0.001

    # Saved top-level switch selections from SimuOriginal.slx
    force_source: str = "sine"
    control_source: str = "constant_zero"

    # Saved Sine Wave block settings.
    force_amplitude: float = 10.0
    force_bias: float = 5.0
    force_frequency_rad: float = 0.5
    force_phase_rad: float = 0.0

    # Plant-side environment profile saved inside the Plant subsystem.
    env_switch_time: float = 30.0
    skin_Ke: float = 331.0
    skin_Be: float = 3e-3
    delta_Ke_after_switch: float = -248.0
    delta_Be_after_switch: float = -0.002

    # Saved valve manual switch positions inside the Valve subsystem.
    # In the SLX these switches currently select P_atm instead of Pm1/Pm2.
    valve_branch_1_pressure_source: str = "atm"
    valve_branch_2_pressure_source: str = "atm"


@dataclass(frozen=True)
class SimuOriginalState:
    Pm1: float
    Pm2: float
    xm_dot: float
    xm: float
    Ps1: float
    Ps2: float
    xs_dot: float
    xs: float
    mL1_dot: float
    mL2_dot: float
    x_v: float
    x_v_dot: float

    def as_array(self) -> ArrayLike:
        return np.array(
            [
                self.Pm1,
                self.Pm2,
                self.xm_dot,
                self.xm,
                self.Ps1,
                self.Ps2,
                self.xs_dot,
                self.xs,
                self.mL1_dot,
                self.mL2_dot,
                self.x_v,
                self.x_v_dot,
            ],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: ArrayLike) -> "SimuOriginalState":
        return cls(*[float(v) for v in values])


@dataclass(frozen=True)
class SimuOriginalResult:
    time: ArrayLike
    state: ArrayLike
    F_h: ArrayLike
    u: ArrayLike
    K_e: ArrayLike
    B_e: ArrayLike
    x_m: ArrayLike
    xm_dot: ArrayLike
    x_s: ArrayLike
    xs_dot: ArrayLike
    Fe: ArrayLike
    Pm1: ArrayLike
    Pm2: ArrayLike
    Ps1: ArrayLike
    Ps2: ArrayLike
    mL1_dot: ArrayLike
    mL2_dot: ArrayLike
    mm1_dot: ArrayLike
    mm2_dot: ArrayLike
    mv1_dot: ArrayLike
    mv2_dot: ArrayLike
    x_v: ArrayLike
    x_v_dot: ArrayLike
    valid_steps: int
    singularity_time: Optional[float]


def build_saved_simuoriginal_state(parms: Optional[ParmsOriginal] = None) -> SimuOriginalState:
    parms = parms or ParmsOriginal()
    return SimuOriginalState(
        Pm1=parms.P_md,
        Pm2=parms.P_md,
        xm_dot=0.0,
        xm=0.0,
        Ps1=parms.P_md,
        Ps2=parms.P_md,
        xs_dot=0.0,
        xs=0.0,
        mL1_dot=0.0,
        mL2_dot=0.0,
        x_v=0.0,
        x_v_dot=0.0,
    )


def saved_force_input(t: float, profile: Optional[SimuOriginalProfile] = None) -> float:
    profile = profile or SimuOriginalProfile()
    if profile.force_source != "sine":
        raise ValueError(f"Unsupported saved force source: {profile.force_source}")
    return profile.force_bias + profile.force_amplitude * math.sin(
        profile.force_frequency_rad * t + profile.force_phase_rad
    )


def saved_control_input(_t: float, profile: Optional[SimuOriginalProfile] = None) -> float:
    profile = profile or SimuOriginalProfile()
    if profile.control_source != "constant_zero":
        raise ValueError(f"Unsupported saved control source: {profile.control_source}")
    return 0.0


def saved_environment(t: float, profile: Optional[SimuOriginalProfile] = None) -> tuple[float, float]:
    profile = profile or SimuOriginalProfile()
    Ke = profile.skin_Ke
    Be = profile.skin_Be
    if t >= profile.env_switch_time:
        Ke += profile.delta_Ke_after_switch
        Be += profile.delta_Be_after_switch
    return Ke, Be


def _valve_reference_pressure(
    source: str,
    chamber_pressure: float,
    parms: ParmsOriginal,
) -> float:
    if source == "atm":
        return parms.P_atm
    if source == "chamber":
        return chamber_pressure
    raise ValueError(f"Unsupported valve pressure source: {source}")


def _valve_branch_constant(reference_pressure: float, parms: ParmsOriginal) -> float:
    pressure_ratio = reference_pressure / parms.P_s
    normalized = (pressure_ratio - parms.b_v) / (1.0 - parms.b_v)
    radical = max(0.0, 1.0 - normalized**2)
    subsonic_gain = parms.C_v * parms.rho0 * parms.P_s * math.sqrt(radical)
    choked_gain = parms.C_v * parms.rho0 * parms.P_s
    return subsonic_gain if pressure_ratio > parms.b_v else choked_gain


def _top_level_observables(
    t: float,
    y: ArrayLike,
    parms: ParmsOriginal,
    profile: SimuOriginalProfile,
    F_h_fn: Callable[[float], float],
    u_fn: Callable[[float], float],
) -> dict[str, float]:
    state = SimuOriginalState.from_array(y)
    F_h = F_h_fn(t)
    u = u_fn(t)
    K_e, B_e = saved_environment(t, profile)

    ref_p1 = _valve_reference_pressure(
        profile.valve_branch_1_pressure_source, state.Pm1, parms
    )
    ref_p2 = _valve_reference_pressure(
        profile.valve_branch_2_pressure_source, state.Pm2, parms
    )
    valve_constant_1 = _valve_branch_constant(ref_p1, parms)
    valve_constant_2 = _valve_branch_constant(ref_p2, parms)

    mv1_dot = valve_constant_1 * state.x_v
    mv2_dot = -valve_constant_2 * state.x_v
    mm1_dot = mv1_dot - state.mL1_dot
    mm2_dot = mv2_dot - state.mL2_dot
    Fe = K_e * state.xs + B_e * state.xs_dot

    return {
        "F_h": F_h,
        "u": u,
        "K_e": K_e,
        "B_e": B_e,
        "Fe": Fe,
        "mv1_dot": mv1_dot,
        "mv2_dot": mv2_dot,
        "mm1_dot": mm1_dot,
        "mm2_dot": mm2_dot,
    }


def simuoriginal_derivatives(
    t: float,
    y: ArrayLike,
    parms: Optional[ParmsOriginal] = None,
    profile: Optional[SimuOriginalProfile] = None,
    F_h_fn: Optional[Callable[[float], float]] = None,
    u_fn: Optional[Callable[[float], float]] = None,
) -> ArrayLike:
    parms = parms or ParmsOriginal()
    profile = profile or SimuOriginalProfile()
    F_h_fn = F_h_fn or (lambda tau: saved_force_input(tau, profile))
    u_fn = u_fn or (lambda tau: saved_control_input(tau, profile))

    s = SimuOriginalState.from_array(y)
    obs = _top_level_observables(t, y, parms, profile, F_h_fn, u_fn)
    F_h = obs["F_h"]
    u = obs["u"]
    K_e = obs["K_e"]
    B_e = obs["B_e"]
    mm1_dot = obs["mm1_dot"]
    mm2_dot = obs["mm2_dot"]

    # Valve second-order transfer function: Xv/U = K_v * omega_v^2 / (s^2 + 2*zeta_v*omega_v*s + omega_v^2)
    x_v_ddot = (
        parms.K_v * parms.omega_v**2 * u
        - 2.0 * parms.zeta_v * parms.omega_v * s.x_v_dot
        - parms.omega_v**2 * s.x_v
    )

    # Tube dynamics.
    mL1_ddot = (
        (parms.A_t / parms.L_t) * (s.Pm1 - s.Ps2)
        - parms.tube_damping * s.mL1_dot
    )
    mL2_ddot = (
        (parms.A_t / parms.L_t) * (s.Pm2 - s.Ps1)
        - parms.tube_damping * s.mL2_dot
    )

    # Master cylinder.
    V_m1 = parms.V_md + parms.A_p * s.xm
    V_m2 = parms.V_md + parms.A_p * (parms.l_cyl - s.xm)
    Pm1_dot = (parms.R * parms.T * mm1_dot - parms.A_p * s.Pm1 * s.xm_dot) / V_m1
    Pm2_dot = (parms.R * parms.T * mm2_dot + parms.A_p * s.Pm2 * s.xm_dot) / V_m2
    xm_ddot = (
        -parms.k_h * s.xm
        + parms.A_p * (s.Pm1 - s.Pm2)
        + F_h
        - (parms.B_h + parms.beta) * s.xm_dot
    ) / parms.m_p

    # Slave cylinder. The Simulink wiring makes the tube-compliance term appear
    # on both pressure-derivative paths, so we solve the equivalent explicit form.
    V_s1 = parms.V_md + parms.A_p * s.xs + parms.tube_compliance
    V_s2 = parms.V_sd + parms.A_p * (parms.l_cyl - s.xs) + parms.tube_compliance
    Ps1_dot = (parms.R * parms.T * s.mL2_dot - parms.A_p * s.Ps1 * s.xs_dot) / V_s1
    Ps2_dot = (parms.R * parms.T * s.mL1_dot + parms.A_p * s.Ps2 * s.xs_dot) / V_s2
    xs_ddot = (
        -K_e * s.xs
        + parms.A_p * (s.Ps1 - s.Ps2)
        - (B_e + parms.beta) * s.xs_dot
    ) / parms.m_p

    return np.array(
        [
            Pm1_dot,
            Pm2_dot,
            xm_ddot,
            s.xm_dot,
            Ps1_dot,
            Ps2_dot,
            xs_ddot,
            s.xs_dot,
            mL1_ddot,
            mL2_ddot,
            s.x_v_dot,
            x_v_ddot,
        ],
        dtype=float,
    )


def _rk4_step(
    f: Callable[[float, ArrayLike], ArrayLike],
    t: float,
    y: ArrayLike,
    dt: float,
) -> ArrayLike:
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_simuoriginal_replica(
    duration: Optional[float] = None,
    parms: Optional[ParmsOriginal] = None,
    profile: Optional[SimuOriginalProfile] = None,
    initial_state: Optional[SimuOriginalState] = None,
    F_h_fn: Optional[Callable[[float], float]] = None,
    u_fn: Optional[Callable[[float], float]] = None,
) -> SimuOriginalResult:
    parms = parms or ParmsOriginal()
    profile = profile or SimuOriginalProfile(fixed_step=parms.Ts)
    initial_state = initial_state or build_saved_simuoriginal_state(parms)
    F_h_fn = F_h_fn or (lambda t: saved_force_input(t, profile))
    u_fn = u_fn or (lambda t: saved_control_input(t, profile))

    dt = profile.fixed_step
    stop_time = profile.stop_time if duration is None else duration
    time = np.arange(0.0, stop_time + 0.5 * dt, dt, dtype=float)
    state = np.zeros((time.size, 12), dtype=float)
    state[0] = initial_state.as_array()

    singularity_time: Optional[float] = None
    valid_steps = time.size

    derivative_fn = lambda tau, yy: simuoriginal_derivatives(
        tau,
        yy,
        parms=parms,
        profile=profile,
        F_h_fn=F_h_fn,
        u_fn=u_fn,
    )

    for i in range(time.size - 1):
        y_i = state[i]
        s = SimuOriginalState.from_array(y_i)
        volumes = (
            parms.V_md + parms.A_p * s.xm,
            parms.V_md + parms.A_p * (parms.l_cyl - s.xm),
            parms.V_md + parms.A_p * s.xs + parms.tube_compliance,
            parms.V_sd + parms.A_p * (parms.l_cyl - s.xs) + parms.tube_compliance,
        )
        if min(volumes) <= 0.0 or not np.all(np.isfinite(y_i)):
            singularity_time = float(time[i])
            valid_steps = i + 1
            state = state[: valid_steps]
            time = time[: valid_steps]
            break

        y_next = _rk4_step(derivative_fn, float(time[i]), y_i, dt)
        if not np.all(np.isfinite(y_next)):
            singularity_time = float(time[i + 1])
            valid_steps = i + 1
            state = state[: valid_steps]
            time = time[: valid_steps]
            break
        state[i + 1] = y_next

    n = time.size
    F_h = np.zeros(n, dtype=float)
    u = np.zeros(n, dtype=float)
    K_e = np.zeros(n, dtype=float)
    B_e = np.zeros(n, dtype=float)
    Fe = np.zeros(n, dtype=float)
    mv1_dot = np.zeros(n, dtype=float)
    mv2_dot = np.zeros(n, dtype=float)
    mm1_dot = np.zeros(n, dtype=float)
    mm2_dot = np.zeros(n, dtype=float)

    for i, t in enumerate(time):
        obs = _top_level_observables(float(t), state[i], parms, profile, F_h_fn, u_fn)
        F_h[i] = obs["F_h"]
        u[i] = obs["u"]
        K_e[i] = obs["K_e"]
        B_e[i] = obs["B_e"]
        Fe[i] = obs["Fe"]
        mv1_dot[i] = obs["mv1_dot"]
        mv2_dot[i] = obs["mv2_dot"]
        mm1_dot[i] = obs["mm1_dot"]
        mm2_dot[i] = obs["mm2_dot"]

    return SimuOriginalResult(
        time=time,
        state=state,
        F_h=F_h,
        u=u,
        K_e=K_e,
        B_e=B_e,
        x_m=state[:, 3],
        xm_dot=state[:, 2],
        x_s=state[:, 7],
        xs_dot=state[:, 6],
        Fe=Fe,
        Pm1=state[:, 0],
        Pm2=state[:, 1],
        Ps1=state[:, 4],
        Ps2=state[:, 5],
        mL1_dot=state[:, 8],
        mL2_dot=state[:, 9],
        mm1_dot=mm1_dot,
        mm2_dot=mm2_dot,
        mv1_dot=mv1_dot,
        mv2_dot=mv2_dot,
        x_v=state[:, 10],
        x_v_dot=state[:, 11],
        valid_steps=valid_steps,
        singularity_time=singularity_time,
    )


def write_simuoriginal_result(
    result: SimuOriginalResult,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "simuoriginal_replica.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "F_h_N",
                "u_V",
                "K_e_N_per_m",
                "B_e_Ns_per_m",
                "x_m_m",
                "xm_dot_m_per_s",
                "x_s_m",
                "xs_dot_m_per_s",
                "Fe_N",
                "Pm1_Pa",
                "Pm2_Pa",
                "Ps1_Pa",
                "Ps2_Pa",
                "mL1_dot",
                "mL2_dot",
                "mm1_dot",
                "mm2_dot",
                "mv1_dot",
                "mv2_dot",
                "x_v",
                "x_v_dot",
            ]
        )
        for row in zip(
            result.time,
            result.F_h,
            result.u,
            result.K_e,
            result.B_e,
            result.x_m,
            result.xm_dot,
            result.x_s,
            result.xs_dot,
            result.Fe,
            result.Pm1,
            result.Pm2,
            result.Ps1,
            result.Ps2,
            result.mL1_dot,
            result.mL2_dot,
            result.mm1_dot,
            result.mm2_dot,
            result.mv1_dot,
            result.mv2_dot,
            result.x_v,
            result.x_v_dot,
        ):
            writer.writerow(row)

    summary_path = out_dir / "simuoriginal_replica_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("SimuOriginal nonlinear Python replica\n")
        handle.write(f"duration_exported_s: {result.time[-1]:.6f}\n")
        handle.write(f"valid_steps: {result.valid_steps}\n")
        handle.write(f"singularity_time_s: {result.singularity_time}\n")
        handle.write(f"x_m_min_m: {np.min(result.x_m):.6g}\n")
        handle.write(f"x_m_max_m: {np.max(result.x_m):.6g}\n")
        handle.write(f"x_s_min_m: {np.min(result.x_s):.6g}\n")
        handle.write(f"x_s_max_m: {np.max(result.x_s):.6g}\n")
        handle.write(f"Fe_min_N: {np.min(result.Fe):.6g}\n")
        handle.write(f"Fe_max_N: {np.max(result.Fe):.6g}\n")
        handle.write(f"Pm1_min_Pa: {np.min(result.Pm1):.6g}\n")
        handle.write(f"Pm1_max_Pa: {np.max(result.Pm1):.6g}\n")
        handle.write(f"Pm2_min_Pa: {np.min(result.Pm2):.6g}\n")
        handle.write(f"Pm2_max_Pa: {np.max(result.Pm2):.6g}\n")
        handle.write(f"Ps1_min_Pa: {np.min(result.Ps1):.6g}\n")
        handle.write(f"Ps1_max_Pa: {np.max(result.Ps1):.6g}\n")
        handle.write(f"Ps2_min_Pa: {np.min(result.Ps2):.6g}\n")
        handle.write(f"Ps2_max_Pa: {np.max(result.Ps2):.6g}\n")
