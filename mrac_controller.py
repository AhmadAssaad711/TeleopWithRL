"""
Filtered-regressor MRAC controller aligned with the IC2AI 2025 paper.

Paper equations implemented here (notation kept close to the manuscript):
  phi(t) = 1/P [u, s*y, y, -s*u_c, -u_c]
  Phi(t) = P1[phi(t)] = 1/P2 [u, s*y, y, -s*u_c, -u_c], with P = P1*P2
  u(t)   = -theta^T(t) Phi(t)                 (realizable law, Eq. 24)
  eta(t) = -(1/P1 u + phi^T theta)            (Eq. 26)
  eps    = e_f + G1*eta, and with Q=P => e_f = e = y - y_m
  theta_dot = -gamma1*phi*eps - gamma2*int(phi*eps)   (Eq. 27)

We choose:
  P1 = Am = s^2 + 2*zeta*wn*s + wn^2
  P2 = A0 = s + a0
so P = A0*Am exactly as in Section IV of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config as cfg


@dataclass
class _FirstOrderFilter:
    a0: float
    x: float = 0.0
    xdot: float = 0.0

    def reset(self) -> None:
        self.x = 0.0
        self.xdot = 0.0

    def step(self, u: float, dt: float) -> float:
        self.xdot = -self.a0 * self.x + u
        self.x += dt * self.xdot
        return self.x


@dataclass
class _SecondOrderFilter:
    a1: float
    a0: float
    y: float = 0.0
    yd: float = 0.0
    ydd: float = 0.0

    def reset(self) -> None:
        self.y = 0.0
        self.yd = 0.0
        self.ydd = 0.0

    def step(self, u: float, dt: float) -> float:
        self.ydd = -self.a1 * self.yd - self.a0 * self.y + u
        self.yd += dt * self.ydd
        self.y += dt * self.yd
        return self.y


class _CascadedFilter:
    """Implements 1 / (A0 * Am) using first-order then second-order filters."""

    def __init__(self, a0: float, wn: float, zeta: float):
        self.f1 = _FirstOrderFilter(a0=a0)
        self.f2 = _SecondOrderFilter(a1=2.0 * zeta * wn, a0=wn * wn)

    def reset(self) -> None:
        self.f1.reset()
        self.f2.reset()

    def step(self, u: float, dt: float) -> float:
        y1 = self.f1.step(u, dt)
        return self.f2.step(y1, dt)

    def step_full(self, u: float, dt: float) -> tuple[float, float]:
        y1 = self.f1.step(u, dt)
        y = self.f2.step(y1, dt)
        return y, self.f2.yd


class _RegressorSignalFilter:
    """
    Dual filter bank for one signal:
      - phi branch: 1/(P1*P2) = 1/(A0*Am)
      - Phi branch: 1/P2      = 1/A0
    """

    def __init__(self, a0: float, wn: float, zeta: float):
        self.p_filter = _CascadedFilter(a0=a0, wn=wn, zeta=zeta)
        self.p2_filter = _FirstOrderFilter(a0=a0)

    def reset(self) -> None:
        self.p_filter.reset()
        self.p2_filter.reset()

    def step(self, u: float, dt: float) -> tuple[float, float, float, float]:
        phi, dphi = self.p_filter.step_full(u, dt)
        phi_p1 = self.p2_filter.step(u, dt)
        dphi_p1 = self.p2_filter.xdot
        return phi, phi_p1, dphi, dphi_p1


class FilteredMRACController:
    """
    Paper-inspired MRAC with filtered regressors and PI adaptation law.

    Controller regressor vectors:
      phi = [1/P u, 1/P (s y), 1/P y, -1/P (s uc), -1/P uc]
      Phi = P1[phi] = [1/P2 u, 1/P2 (s y), 1/P2 y, -1/P2 (s uc), -1/P2 uc]
    Control law:
      u = -theta^T Phi
    Adaptation law:
      theta_dot = -gamma1 * phi * eps - gamma2 * integral(phi * eps)
    """

    def __init__(self):
        self.wn = float(cfg.MRAC_WN)
        self.zeta = float(cfg.MRAC_ZETA)
        self.a0 = float(cfg.MRAC_A0)
        self.gamma1 = float(cfg.MRAC_GAMMA1)
        self.gamma2 = float(cfg.MRAC_GAMMA2)
        self.u_clip = float(cfg.MRAC_U_CLIP)
        self.dt = float(cfg.RL_DT)
        self.g1_gain = float(getattr(cfg, "MRAC_G1_GAIN", 1.0))

        # Adaptive parameters: [r1', s0, s1, t0, t1]
        self.theta = np.array(cfg.MRAC_THETA0, dtype=np.float64).copy()
        self.int_phi_eps = np.zeros_like(self.theta)

        # Reference model state: y_m'' + 2*zeta*wn*y_m' + wn^2*y_m = wn^2*u_c
        self.y_m = 0.0
        self.y_md = 0.0

        # Filter banks for regressor terms (phi and P1[phi] branches)
        self.f_u = _RegressorSignalFilter(self.a0, self.wn, self.zeta)
        self.f_y = _RegressorSignalFilter(self.a0, self.wn, self.zeta)
        self.f_uc = _RegressorSignalFilter(self.a0, self.wn, self.zeta)

        # 1/P1(u), with P1 = Am
        self.f_p1_u = _SecondOrderFilter(a1=2.0 * self.zeta * self.wn, a0=self.wn * self.wn)

        # Last-step diagnostics (for plotting/verification)
        self.last_phi = np.zeros_like(self.theta)
        self.last_phi_p1 = np.zeros_like(self.theta)
        self.last_eps = 0.0
        self.last_eta = 0.0
        self.last_e = 0.0
        self.last_theta_dot = np.zeros_like(self.theta)

        self.u_prev = 0.0

    def reset(self) -> None:
        self.theta = np.array(cfg.MRAC_THETA0, dtype=np.float64).copy()
        self.int_phi_eps[:] = 0.0

        self.y_m = 0.0
        self.y_md = 0.0

        self.f_u.reset()
        self.f_y.reset()
        self.f_uc.reset()
        self.f_p1_u.reset()

        self.last_phi[:] = 0.0
        self.last_phi_p1[:] = 0.0
        self.last_eps = 0.0
        self.last_eta = 0.0
        self.last_e = 0.0
        self.last_theta_dot[:] = 0.0

        self.u_prev = 0.0

    def _update_reference_model(self, u_c: float) -> None:
        y_mdd = -2.0 * self.zeta * self.wn * self.y_md - (self.wn * self.wn) * self.y_m + (self.wn * self.wn) * u_c
        self.y_md += self.dt * y_mdd
        self.y_m += self.dt * self.y_md

    def step_voltage(self, pos_error: float, u_c: float) -> float:
        y = float(pos_error)
        uc = float(u_c)

        # Build filtered regressors.
        phi_u, phi_p1_u, _, _ = self.f_u.step(self.u_prev, self.dt)
        phi_y, phi_p1_y, phi_sy, phi_p1_sy = self.f_y.step(y, self.dt)
        phi_uc, phi_p1_uc, phi_suc, phi_p1_suc = self.f_uc.step(uc, self.dt)
        phi = np.array([phi_u, phi_sy, phi_y, -phi_suc, -phi_uc], dtype=np.float64)
        phi_p1 = np.array([phi_p1_u, phi_p1_sy, phi_p1_y, -phi_p1_suc, -phi_p1_uc], dtype=np.float64)

        # Reference model tracking error: e_f = e since Q = P in paper design.
        self._update_reference_model(uc)
        e = y - self.y_m

        # Realizable control law from paper Eq. (24): u = -theta^T P1[phi].
        u_raw = -float(np.dot(self.theta, phi_p1))
        u = float(np.clip(u_raw, -self.u_clip, self.u_clip))

        # Augmented error epsilon = e + G1*eta with normalized G1 gain.
        one_over_p1_u = self.f_p1_u.step(self.u_prev, self.dt)
        eta = -(one_over_p1_u + float(np.dot(phi, self.theta)))
        eps = e + self.g1_gain * eta

        # PI adaptation law.
        self.int_phi_eps += phi * eps * self.dt
        theta_dot = -self.gamma1 * phi * eps - self.gamma2 * self.int_phi_eps
        self.theta += theta_dot * self.dt

        self.last_phi[:] = phi
        self.last_phi_p1[:] = phi_p1
        self.last_eps = eps
        self.last_eta = eta
        self.last_e = e
        self.last_theta_dot[:] = theta_dot

        self.u_prev = u
        return u

    def diagnostics(self) -> dict[str, np.ndarray | float]:
        return {
            "theta": self.theta.copy(),
            "theta_dot": self.last_theta_dot.copy(),
            "phi": self.last_phi.copy(),
            "phi_p1": self.last_phi_p1.copy(),
            "eps": float(self.last_eps),
            "eta": float(self.last_eta),
            "e": float(self.last_e),
            "y_m": float(self.y_m),
            "u_prev": float(self.u_prev),
        }

    @staticmethod
    def voltage_to_action(u_v: float, action_table: np.ndarray) -> int:
        return int(np.argmin(np.abs(action_table - u_v)))

    def step_action(self, pos_error: float, u_c: float, action_table: np.ndarray) -> tuple[int, float]:
        u_v = self.step_voltage(pos_error=pos_error, u_c=u_c)
        action = self.voltage_to_action(u_v, action_table)
        return action, u_v
