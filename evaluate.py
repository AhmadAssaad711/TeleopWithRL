"""
Evaluation & Visualisation — trained Q-learning valve controller
for bilateral pneumatic teleoperation.

Usage:
    python evaluate.py              # loads q_table.npy + training_log.npz
    python evaluate.py --no-train   # skip training-curve plots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from teleop_env import TeleopEnv
from q_learning_agent import QLearningAgent


# ====================================================================== #
#  Run one greedy episode                                                 #
# ====================================================================== #
def run_episode(env: TeleopEnv, agent: QLearningAgent) -> dict:
    obs, _ = env.reset()
    state  = env.discretise_obs(obs)
    done   = False

    while not done:
        action = agent.select_action(state)
        obs, _, terminated, truncated, _ = env.step(action)
        done  = terminated or truncated
        state = env.discretise_obs(obs)

    return env.render()


# ====================================================================== #
#  Run a "no-control" baseline (V_s = 0)                                 #
# ====================================================================== #
def run_baseline(env: TeleopEnv) -> dict:
    """Episode with zero valve voltage (pure open-loop passive pneumatics)."""
    env.reset()
    done = False
    zero_action = int(np.argmin(np.abs(env._action_table)))

    while not done:
        _, _, terminated, truncated, _ = env.step(zero_action)
        done = terminated or truncated

    return env.render()


# ====================================================================== #
#  Plotting                                                               #
# ====================================================================== #
def plot_episode(history: dict, title: str = "RL Valve Controller"):
    """5-panel plot: positions, pressures, forces, valve, errors."""
    t = np.array(history["time"])

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"Bilateral Pneumatic Teleoperation — {title}",
                 fontsize=14, fontweight="bold")

    # 1 ── Position tracking ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, np.array(history["x_m"]) * 1000, "b", lw=1.2, label="x_m (master)")
    ax.plot(t, np.array(history["x_s"]) * 1000, "r", lw=1.2, label="x_s (slave)")
    ax.axhline(cfg.L_CYL / 2.0 * 1000, color="gray", ls=":", lw=0.8, label="mid-stroke")
    ax.set_ylabel("Position [mm]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2 ── Chamber pressures ──────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, np.array(history["P_s1"]) / 1000, "b", lw=1, label="P_s1")
    ax.plot(t, np.array(history["P_s2"]) / 1000, "r", lw=1, label="P_s2")
    ax.plot(t, np.array(history["P_m1"]) / 1000, "b--", lw=0.7, alpha=0.6, label="P_m1")
    ax.plot(t, np.array(history["P_m2"]) / 1000, "r--", lw=0.7, alpha=0.6, label="P_m2")
    ax.set_ylabel("Pressure [kPa]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3 ── Forces ─────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, history["F_h"], "b",  lw=1, label="F_h (human)")
    ax.plot(t, history["F_e"], "r",  lw=1, label="F_e (environment)")
    ax.set_ylabel("Force [N]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4 ── Valve voltage & spool ──────────────────────────────────────
    ax = axes[3]
    ax.plot(t, history["u_v"], "g",  lw=1,   label="u_v (RL output) [V]")
    ax2 = ax.twinx()
    ax2.plot(t, history["x_v"], "m", lw=0.8, alpha=0.7, label="x_v (spool)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Valve voltage [V]")
    ax2.set_ylabel("Spool position [-]")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5 ── Tracking Error ─────────────────────────────────────────────
    ax = axes[4]
    pe = np.array(history["pos_error"]) * 1000          # → mm
    ax.plot(t, pe, "b", lw=1, label="pos error (x_m − x_s)")
    ax.set_ylabel("Tracking error [mm]")
    ax.set_xlabel("Time [s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_dir = os.path.join(os.path.dirname(__file__), "results", "plots")
    os.makedirs(_save_dir, exist_ok=True)
    plt.savefig(os.path.join(_save_dir, "evaluation_plot.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(log_path: str = "training_log.npz"):
    """Plot reward curve + evaluation metrics over training."""
    data = np.load(log_path)
    ep_rewards = data["episode_rewards"]

    # ── 1. Episode reward (with moving average) ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep_rewards, alpha=0.15, lw=0.5, color="steelblue")
    window = 200
    if len(ep_rewards) >= window:
        ma = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(ep_rewards)), ma,
                color="navy", lw=2, label=f"MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_dir = os.path.join(os.path.dirname(__file__), "results", "plots")
    os.makedirs(_save_dir, exist_ok=True)
    plt.savefig(os.path.join(_save_dir, "training_reward.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # ── 2. Evaluation curves ─────────────────────────────────────────
    if "eval_rewards" not in data:
        return

    eval_r  = data["eval_rewards"]
    eval_te = data["eval_tracking_rmse"]  * 1000   # → mm
    x_eval  = np.arange(1, len(eval_r) + 1) * cfg.EVAL_EVERY

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x_eval, eval_r, "bo-", lw=1.5)
    axes[0].set_title("Eval: Mean Reward")
    axes[0].set_xlabel("Episode");  axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_eval, eval_te, "ro-", lw=1.5)
    axes[1].set_title("Eval: Tracking RMSE")
    axes[1].set_xlabel("Episode");  axes[1].set_ylabel("RMSE [mm]")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Evaluation Metrics Over Training", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_dir = os.path.join(os.path.dirname(__file__), "results", "plots")
    os.makedirs(_save_dir, exist_ok=True)
    plt.savefig(os.path.join(_save_dir, "eval_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()


# ====================================================================== #
#  Main                                                                   #
# ====================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-train", action="store_true",
                        help="Skip training-curve plots")
    args = parser.parse_args()

    q_path = Path(os.path.join(os.path.dirname(__file__), "results", "models", "q_table.npy"))
    if not q_path.exists():
        # Fallback to root for backward compat
        q_path = Path("q_table.npy")
    if not q_path.exists():
        print(f"ERROR: q_table.npy not found. Run train.py first.")
        sys.exit(1)

    # ── Load agent ───────────────────────────────────────────────────
    env        = TeleopEnv()
    state_dims = env.get_state_dims()
    agent      = QLearningAgent(state_dims, cfg.N_ACTIONS)
    agent.load(str(q_path))
    agent.epsilon = 0.0            # fully greedy
    print(f"Loaded: {agent}")

    # ── Run RL episode ───────────────────────────────────────────────
    print("\nRunning RL evaluation episode ...")
    h_rl = run_episode(env, agent)

    pe_rl = np.array(h_rl["pos_error"])
    rmse_track = np.sqrt(np.mean(pe_rl ** 2)) * 1000   # mm
    total_r    = np.sum(h_rl["reward"])

    # ── Run baseline (zero voltage) ──────────────────────────────────
    print("Running baseline (V_s=0) episode ...")
    h_bl = run_baseline(env)

    pe_bl = np.array(h_bl["pos_error"])
    rmse_track_bl = np.sqrt(np.mean(pe_bl ** 2)) * 1000
    total_r_bl    = np.sum(h_bl["reward"])

    # ── Print comparison ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  {'Metric':<30} {'RL':>10} {'Baseline':>10}")
    print("-" * 60)
    print(f"  {'Tracking RMSE [mm]':<30} {rmse_track:>10.2f} {rmse_track_bl:>10.2f}")
    print(f"  {'Total Reward':<30} {total_r:>10.2f} {total_r_bl:>10.2f}")
    print("=" * 60)

    # ── Plots ────────────────────────────────────────────────────────
    plot_episode(h_rl, title="RL Voltage Controller")
    plot_episode(h_bl, title="Baseline (V = 0)")

    log_npz = Path(os.path.join(os.path.dirname(__file__), "results", "logs", "training_log.npz"))
    if not log_npz.exists():
        log_npz = Path("training_log.npz")
    if not args.no_train and log_npz.exists():
        plot_training_curves(str(log_npz))

    print("\nPlots saved to results/plots/")


if __name__ == "__main__":
    main()
