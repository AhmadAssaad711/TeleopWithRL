"""
Filtered-regressor MRAC controller (paper-inspired implementation).

This module follows the controller/filter structure described in:
  "Enhanced MRAC Design Using Filtered Regressors for Bilateral Pneumatic
   Teleoperation Control" (IC2AI 2025).

Notes:
- The paper provides controller structure and adaptation law, but does not expose
  all linearized plant coefficients needed for a numerically identical replica.
- This implementation keeps the same architecture (filtered regressors + PI
  adaptation law) and uses available simulation signals in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config as cfg


@dataclass
class _FirstOrderFilter:
    a0: float
    x: float = 0.0

    def reset(self) -> None:
        self.x = 0.0

    def step(self, u: float, dt: float) -> float:
        self.x += dt * (-self.a0 * self.x + u)
        return self.x


@dataclass
class _SecondOrderFilter:
    a1: float
    a0: float
    y: float = 0.0
    yd: float = 0.0

    def reset(self) -> None:
        self.y = 0.0
        self.yd = 0.0

    def step(self, u: float, dt: float) -> float:
        ydd = -self.a1 * self.yd - self.a0 * self.y + u
        self.yd += dt * ydd
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


class FilteredMRACController:
    """
    Paper-inspired MRAC with filtered regressors and PI adaptation law.

    Controller regressor shape:
      phi = [1/P u, 1/P (s y), 1/P y, -1/P (s uc), -1/P uc]
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

        # Adaptive parameters: [r1', s0, s1, t0, t1]
        self.theta = np.array(cfg.MRAC_THETA0, dtype=np.float64).copy()
        self.int_phi_eps = np.zeros_like(self.theta)

        # Reference model state: y_m'' + 2*zeta*wn*y_m' + wn^2*y_m = wn^2*u_c
        self.y_m = 0.0
        self.y_md = 0.0

        # Filter banks for regressor terms
        self.f_u = _CascadedFilter(self.a0, self.wn, self.zeta)
        self.f_sy = _CascadedFilter(self.a0, self.wn, self.zeta)
        self.f_y = _CascadedFilter(self.a0, self.wn, self.zeta)
        self.f_suc = _CascadedFilter(self.a0, self.wn, self.zeta)
        self.f_uc = _CascadedFilter(self.a0, self.wn, self.zeta)

        # 1/P1(u), with P1 = Am
        self.f_p1_u = _SecondOrderFilter(a1=2.0 * self.zeta * self.wn, a0=self.wn * self.wn)

        self.prev_y = 0.0
        self.prev_uc = 0.0
        self.u_prev = 0.0
        self._initialized = False

    def reset(self) -> None:
        self.theta = np.array(cfg.MRAC_THETA0, dtype=np.float64).copy()
        self.int_phi_eps[:] = 0.0

        self.y_m = 0.0
        self.y_md = 0.0

        self.f_u.reset()
        self.f_sy.reset()
        self.f_y.reset()
        self.f_suc.reset()
        self.f_uc.reset()
        self.f_p1_u.reset()

        self.prev_y = 0.0
        self.prev_uc = 0.0
        self.u_prev = 0.0
        self._initialized = False

    def _update_reference_model(self, u_c: float) -> None:
        y_mdd = -2.0 * self.zeta * self.wn * self.y_md - (self.wn * self.wn) * self.y_m + (self.wn * self.wn) * u_c
        self.y_md += self.dt * y_mdd
        self.y_m += self.dt * self.y_md

    def step_voltage(self, pos_error: float, u_c: float) -> float:
        if not self._initialized:
            self.prev_y = float(pos_error)
            self.prev_uc = float(u_c)
            self._initialized = True

        y = float(pos_error)
        uc = float(u_c)

        dy = (y - self.prev_y) / self.dt
        duc = (uc - self.prev_uc) / self.dt
        self.prev_y = y
        self.prev_uc = uc

        # Build filtered regressor terms using previous control command.
        phi_u = self.f_u.step(self.u_prev, self.dt)
        phi_sy = self.f_sy.step(dy, self.dt)
        phi_y = self.f_y.step(y, self.dt)
        phi_suc = self.f_suc.step(duc, self.dt)
        phi_uc = self.f_uc.step(uc, self.dt)
        phi = np.array([phi_u, phi_sy, phi_y, -phi_suc, -phi_uc], dtype=np.float64)

        # Reference model tracking error.
        self._update_reference_model(uc)
        e = y - self.y_m

        # Realizable control law.
        u_raw = -float(np.dot(self.theta, phi))
        u = float(np.clip(u_raw, -self.u_clip, self.u_clip))

        # Augmented error with normalized G1 gain.
        one_over_p1_u = self.f_p1_u.step(self.u_prev, self.dt)
        eta = -(one_over_p1_u + float(np.dot(phi, self.theta)))
        eps = e + eta

        # PI adaptation law.
        self.int_phi_eps += phi * eps * self.dt
        theta_dot = -self.gamma1 * phi * eps - self.gamma2 * self.int_phi_eps
        self.theta += theta_dot * self.dt

        self.u_prev = u
        return u

    @staticmethod
    def voltage_to_action(u_v: float, action_table: np.ndarray) -> int:
        return int(np.argmin(np.abs(action_table - u_v)))

    def step_action(self, pos_error: float, u_c: float, action_table: np.ndarray) -> tuple[int, float]:
        u_v = self.step_voltage(pos_error=pos_error, u_c=u_c)
        action = self.voltage_to_action(u_v, action_table)
        return action, u_v
