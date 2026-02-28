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
import os
from typing import Any

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

def _mk_outdirs() -> dict[str, str]:
    base = os.path.join(
        os.path.dirname(__file__),
        cfg.RESULTS_ROOT_DIR,
        cfg.MRAC_RESULTS_DIR,
        "controller_env_run",
    )
    paths = {
        "base": base,
        "logs": os.path.join(base, "logs"),
        "plots": os.path.join(base, "plots"),
        "episodes": os.path.join(base, "episodes"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def _as_array(values: list[Any]) -> np.ndarray:
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return np.asarray(values, dtype=object)


def _save_response_plot(t: np.ndarray, x_m: np.ndarray, x_s: np.ndarray, pe: np.ndarray, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, x_m * 1000.0, label="Master position", lw=1.8, color="tab:blue")
    axes[0].plot(t, x_s * 1000.0, label="Slave position", lw=1.8, color="tab:orange")
    axes[0].axvline(cfg.PAPER_ENV_SWITCH_TIME, color="gray", lw=1.0, ls="--")
    axes[0].set_ylabel("Position [mm]")
    axes[0].set_title("MRAC on TeleopEnv: Master/Slave Response")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, pe * 1000.0, lw=1.8, color="tab:red", label="x_m - x_s")
    axes[1].axhline(0.0, color="gray", lw=0.8)
    axes[1].axvline(cfg.PAPER_ENV_SWITCH_TIME, color="gray", lw=1.0, ls="--")
    axes[1].set_ylabel("Tracking error [mm]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_theta_plot(t: np.ndarray, theta: np.ndarray, out_path: str) -> None:
    import matplotlib.pyplot as plt

    labels = ("r1_prime", "s0", "s1", "t0", "t1")
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown")
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (lbl, c) in enumerate(zip(labels, colors)):
        ax.plot(t, theta[:, i], lw=1.6, color=c, label=lbl)
    ax.axvline(cfg.PAPER_ENV_SWITCH_TIME, color="gray", lw=1.0, ls="--")
    ax.set_title("MRAC on TeleopEnv: Adaptive Parameters")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Parameter value")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=5, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_control_plot(t: np.ndarray, u: np.ndarray, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t, u, lw=1.8, color="tab:purple")
    ax.axvline(cfg.PAPER_ENV_SWITCH_TIME, color="gray", lw=1.0, ls="--")
    ax.set_title("MRAC on TeleopEnv: Control Input")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("u_v [V]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_regressor_plot(t: np.ndarray, phi: np.ndarray, out_path: str) -> None:
    import matplotlib.pyplot as plt

    labels = ("1/P u", "1/P (s y)", "1/P y", "-1/P (s u_c)", "-1/P u_c")
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    for i in range(5):
        axes[i].plot(t, phi[:, i], lw=1.4)
        axes[i].axvline(cfg.PAPER_ENV_SWITCH_TIME, color="gray", lw=1.0, ls="--")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("MRAC on TeleopEnv: Filtered Regressor Components", y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_mrac_on_teleop_env(seed: int = 0) -> dict[str, str]:
    import matplotlib
    from teleop_env import TeleopEnv

    matplotlib.use("Agg", force=True)

    paths = _mk_outdirs()
    env = TeleopEnv(
        env_mode=cfg.ENV_MODE_CHANGING,
        episode_duration=cfg.PAPER_EPISODE_DURATION,
        env_switch_time=cfg.PAPER_ENV_SWITCH_TIME,
        terminate_on_error=False,
    )
    ctrl = FilteredMRACController()
    ctrl.reset()

    _, info = env.reset(seed=seed)
    env.fh_amp = float(cfg.PAPER_FORCE_AMP)
    env.fh_freq = float(cfg.PAPER_FORCE_FREQ)
    env.fh_phase = float(cfg.PAPER_FORCE_PHASE)

    ctrl_history: dict[str, list] = {
        "time": [],
        "u_v": [],
        "theta": [],
        "theta_dot": [],
        "phi": [],
        "phi_p1": [],
        "eps": [],
        "eta": [],
        "e": [],
        "y_m": [],
    }

    done = False
    while not done:
        u_v = ctrl.step_voltage(
            pos_error=float(info["x_m"] - info["x_s"]),
            u_c=float(info["x_m"]),
        )
        d = ctrl.diagnostics()
        _, _, terminated, truncated, info = env.step_voltage(u_v)
        done = terminated or truncated

        ctrl_history["time"].append(float(info["time"]))
        ctrl_history["u_v"].append(float(u_v))
        ctrl_history["theta"].append(np.array(d["theta"], dtype=np.float64))
        ctrl_history["theta_dot"].append(np.array(d["theta_dot"], dtype=np.float64))
        ctrl_history["phi"].append(np.array(d["phi"], dtype=np.float64))
        ctrl_history["phi_p1"].append(np.array(d["phi_p1"], dtype=np.float64))
        ctrl_history["eps"].append(float(d["eps"]))
        ctrl_history["eta"].append(float(d["eta"]))
        ctrl_history["e"].append(float(d["e"]))
        ctrl_history["y_m"].append(float(d["y_m"]))

    env_history = env.render() or {}
    t = np.asarray(env_history.get("time", []), dtype=np.float64)
    x_m = np.asarray(env_history.get("x_m", []), dtype=np.float64)
    x_s = np.asarray(env_history.get("x_s", []), dtype=np.float64)
    pe = np.asarray(env_history.get("pos_error", []), dtype=np.float64)
    tr = np.asarray(env_history.get("transparency_error", []), dtype=np.float64)

    t_ctrl = np.asarray(ctrl_history["time"], dtype=np.float64)
    theta = np.asarray(ctrl_history["theta"], dtype=np.float64)
    phi = np.asarray(ctrl_history["phi"], dtype=np.float64)
    u_v = np.asarray(ctrl_history["u_v"], dtype=np.float64)

    episode_npz = os.path.join(paths["episodes"], "mrac_controller_episode.npz")
    payload: dict[str, np.ndarray] = {}
    for k, v in env_history.items():
        payload[f"env_{k}"] = _as_array(v)
    for k, v in ctrl_history.items():
        payload[f"ctrl_{k}"] = _as_array(v)
    np.savez(episode_npz, **payload)

    if t.size and theta.ndim == 2 and theta.shape[0] == t_ctrl.size:
        _save_response_plot(
            t=t,
            x_m=x_m,
            x_s=x_s,
            pe=pe,
            out_path=os.path.join(paths["plots"], "mrac_response.png"),
        )
        _save_theta_plot(
            t=t_ctrl,
            theta=theta,
            out_path=os.path.join(paths["plots"], "mrac_theta.png"),
        )
        _save_control_plot(
            t=t_ctrl,
            u=u_v,
            out_path=os.path.join(paths["plots"], "mrac_control_input.png"),
        )
        _save_regressor_plot(
            t=t_ctrl,
            phi=phi,
            out_path=os.path.join(paths["plots"], "mrac_regressors.png"),
        )

    rmse_tracking = float(np.sqrt(np.mean(pe ** 2))) if pe.size else float("nan")
    rmse_transparency = float(np.sqrt(np.mean(tr ** 2))) if tr.size else float("nan")
    max_abs_error = float(np.max(np.abs(pe))) if pe.size else float("nan")

    summary_path = os.path.join(paths["logs"], "mrac_controller_metrics.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MRAC on TeleopEnv metrics\n")
        f.write(f"duration_s={cfg.PAPER_EPISODE_DURATION:.2f}\n")
        f.write(f"switch_time_s={cfg.PAPER_ENV_SWITCH_TIME:.2f}\n")
        f.write(f"steps={len(t)}\n")
        f.write(f"tracking_rmse_m={rmse_tracking:.8f}\n")
        f.write(f"transparency_rmse_w={rmse_transparency:.8f}\n")
        f.write(f"max_abs_tracking_error_m={max_abs_error:.8f}\n")
        if theta.ndim == 2 and theta.size:
            f.write("final_theta=" + ",".join(f"{v:.8f}" for v in theta[-1]) + "\n")

    print("MRAC run on TeleopEnv complete.")
    print(f"Results: {paths['base']}")
    print(f"Episode file: {episode_npz}")
    print(f"Summary: {summary_path}")
    return paths


if __name__ == "__main__":
    run_mrac_on_teleop_env(seed=0)
