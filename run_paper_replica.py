"""Run a paper-faithful MRAC simulation on the nonlinear teleoperation model."""

from __future__ import annotations

import os
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from mrac_controller import FilteredMRACController
from teleop_env import TeleopEnv


def _mk_outdirs() -> dict[str, str]:
    base = os.path.join(os.path.dirname(__file__), cfg.RESULTS_ROOT_DIR, cfg.PAPER_RESULTS_DIR)
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
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, x_m * 1000.0, label="Master position", lw=1.8, color="tab:blue")
    axes[0].plot(t, x_s * 1000.0, label="Slave position", lw=1.8, color="tab:orange")
    axes[0].axvline(cfg.PAPER_ENV_SWITCH_TIME, color="gray", lw=1.0, ls="--")
    axes[0].set_ylabel("Position [mm]")
    axes[0].set_title("Paper Replica: Master/Slave Response")
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
    labels = ("r1_prime", "s0", "s1", "t0", "t1")
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown")
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (lbl, c) in enumerate(zip(labels, colors)):
        ax.plot(t, theta[:, i], lw=1.6, color=c, label=lbl)
    ax.axvline(cfg.PAPER_ENV_SWITCH_TIME, color="gray", lw=1.0, ls="--")
    ax.set_title("Paper Replica: Adaptive Parameters")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Parameter value")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=5, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_control_plot(t: np.ndarray, u: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t, u, lw=1.8, color="tab:purple")
    ax.axvline(cfg.PAPER_ENV_SWITCH_TIME, color="gray", lw=1.0, ls="--")
    ax.set_title("Paper Replica: Control Input")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("u_v [V]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_regressor_plot(t: np.ndarray, phi: np.ndarray, out_path: str) -> None:
    labels = ("1/P u", "1/P (s y)", "1/P y", "-1/P (s u_c)", "-1/P u_c")
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    for i in range(5):
        axes[i].plot(t, phi[:, i], lw=1.4)
        axes[i].axvline(cfg.PAPER_ENV_SWITCH_TIME, color="gray", lw=1.0, ls="--")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Paper Replica: Filtered Regressor Components", y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_paper_replica(seed: int = 0) -> None:
    paths = _mk_outdirs()
    env = TeleopEnv(
        env_mode=cfg.ENV_MODE_CHANGING,
        episode_duration=cfg.PAPER_EPISODE_DURATION,
        env_switch_time=cfg.PAPER_ENV_SWITCH_TIME,
        terminate_on_error=False,
    )
    ctrl = FilteredMRACController()
    ctrl.reset()

    obs, info = env.reset(seed=seed)
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
        obs, _, terminated, truncated, info = env.step_voltage(u_v)
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

    episode_npz = os.path.join(paths["episodes"], "paper_replica_episode.npz")
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
            out_path=os.path.join(paths["plots"], "paper_fig4_response.png"),
        )
        _save_theta_plot(
            t=t_ctrl,
            theta=theta,
            out_path=os.path.join(paths["plots"], "paper_fig5_theta.png"),
        )
        _save_control_plot(
            t=t_ctrl,
            u=u_v,
            out_path=os.path.join(paths["plots"], "paper_fig6_control_input.png"),
        )
        _save_regressor_plot(
            t=t_ctrl,
            phi=phi,
            out_path=os.path.join(paths["plots"], "paper_fig7_regressors.png"),
        )

    rmse_tracking = float(np.sqrt(np.mean(pe ** 2))) if pe.size else float("nan")
    rmse_transparency = float(np.sqrt(np.mean(tr ** 2))) if tr.size else float("nan")
    max_abs_error = float(np.max(np.abs(pe))) if pe.size else float("nan")

    summary_path = os.path.join(paths["logs"], "paper_replica_metrics.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Paper-replica simulation metrics\n")
        f.write(f"duration_s={cfg.PAPER_EPISODE_DURATION:.2f}\n")
        f.write(f"switch_time_s={cfg.PAPER_ENV_SWITCH_TIME:.2f}\n")
        f.write(f"steps={len(t)}\n")
        f.write(f"tracking_rmse_m={rmse_tracking:.8f}\n")
        f.write(f"transparency_rmse_w={rmse_transparency:.8f}\n")
        f.write(f"max_abs_tracking_error_m={max_abs_error:.8f}\n")
        if theta.ndim == 2 and theta.size:
            f.write("final_theta=" + ",".join(f"{v:.8f}" for v in theta[-1]) + "\n")

    print("Paper-replica run complete.")
    print(f"Results: {paths['base']}")
    print(f"Episode file: {episode_npz}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    run_paper_replica(seed=0)
