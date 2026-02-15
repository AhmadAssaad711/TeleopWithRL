"""Generate plots from current training data."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WD = os.path.dirname(os.path.abspath(__file__))
os.chdir(WD)


def _parse_eval_line(line: str):
    """Parse one train.py evaluation print line into metrics."""
    if "Ep " not in line or "TE" not in line:
        return None

    normalized = line.replace("│", "|").replace("â”‚", "|")
    parts = [p.strip() for p in normalized.split("|") if p.strip()]
    if len(parts) < 6:
        return None

    try:
        ep = int(parts[0].split()[-1])
        epsilon = float(parts[1].split()[-1])
        reward = float(parts[2].split()[-1])
        te_mm = float(parts[3].split()[-2])
        coverage = float(parts[5].split()[0].replace("%", ""))
    except (ValueError, IndexError):
        return None

    return ep, epsilon, reward, te_mm, coverage


def generate_plots() -> None:
    # 1) Parse evaluation checkpoints from log
    log_path = os.path.join(WD, "results", "logs", "training_output.log")
    if not os.path.exists(log_path):
        log_path = os.path.join(WD, "training_output.log")

    eps, epsilons, rewards, rmses, coverages = [], [], [], [], []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parsed = _parse_eval_line(line)
                if parsed is None:
                    continue
                ep, epsilon, reward, te_mm, cov = parsed
                eps.append(ep)
                epsilons.append(epsilon)
                rewards.append(reward)
                rmses.append(te_mm)
                coverages.append(cov)

    print(f"Found {len(eps)} evaluation checkpoints (up to ep {eps[-1] if eps else '?'})")

    # 2) Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Q-Learning Training Progress - Pneumatic Teleoperation", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(eps, rmses, "ro-", lw=2, markersize=8)
    ax.axhline(22.2, color="gray", ls="--", lw=1.5, label="Baseline (no valve) = 22.2 mm")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Tracking RMSE [mm]")
    ax.set_title("Tracking Error (lower = better)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(eps, rewards, "bo-", lw=2, markersize=8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Evaluation Reward (higher = better)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(eps, epsilons, "gs-", lw=2, markersize=8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(eps, coverages, "ms-", lw=2, markersize=8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coverage [%]")
    ax.set_title("Q-table State-Action Coverage")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(WD, "results", "plots", "training_progress.png")
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close()

    # 3) Run one RL episode + baseline and make comparison plots
    import config as cfg
    from q_learning_agent import QLearningAgent
    from teleop_env import TeleopEnv

    q_file = os.path.join(WD, "results", "models", "q_table.npy")
    if not os.path.exists(q_file):
        q_file = os.path.join(WD, "q_table.npy")
    if not os.path.exists(q_file):
        print("No q_table.npy found - skipping episode plots")
        print("\nDone!")
        return

    env = TeleopEnv()
    state_dims = env.get_state_dims()
    agent = QLearningAgent(state_dims, cfg.N_ACTIONS)
    agent.load(q_file)
    agent.epsilon = 0.0
    print("Loaded Q-table successfully.")

    # RL episode
    obs, _ = env.reset()
    env.fh_amp = 10.0
    env.fh_freq = 0.5
    env.fh_phase = 0.0
    env.Be, env.Ke = cfg.SKIN_BE, cfg.SKIN_KE
    state = env.discretise_obs(obs)
    zero_act = int(np.argmin(np.abs(cfg.V_LEVELS)))
    done = False
    while not done:
        q_values = agent.q_values(state)
        max_q = q_values.max()
        best = np.flatnonzero(q_values == max_q)
        action = zero_act if zero_act in best else int(best[0])
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = env.discretise_obs(obs)
    h_rl = env.render()

    # Baseline episode
    env.reset()
    env.fh_amp = 10.0
    env.fh_freq = 0.5
    env.fh_phase = 0.0
    env.Be, env.Ke = cfg.SKIN_BE, cfg.SKIN_KE
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(zero_act)
        done = terminated or truncated
    h_bl = env.render()

    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
    fig.suptitle("RL Controller vs Baseline (no valve) - Skin Environment", fontsize=14, fontweight="bold")

    t_rl = np.array(h_rl["time"])
    t_bl = np.array(h_bl["time"])

    ax = axes[0]
    ax.plot(t_rl, np.array(h_rl["x_m"]) * 1000, "b-", lw=1.5, label="x_m (master)")
    ax.plot(t_rl, np.array(h_rl["x_s"]) * 1000, "r-", lw=1.5, label="x_s (slave, RL)")
    ax.plot(t_bl, np.array(h_bl["x_s"]) * 1000, "r--", lw=1, alpha=0.6, label="x_s (slave, baseline)")
    ax.set_ylabel("Position [mm]")
    ax.set_title("Position Tracking")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_rl, np.array(h_rl["P_s1"]) / 1000, "b-", lw=1, label="P_s1 (RL)")
    ax.plot(t_rl, np.array(h_rl["P_s2"]) / 1000, "r-", lw=1, label="P_s2 (RL)")
    ax.plot(t_rl, np.array(h_rl["P_m1"]) / 1000, "b--", lw=0.7, alpha=0.5, label="P_m1")
    ax.plot(t_rl, np.array(h_rl["P_m2"]) / 1000, "r--", lw=0.7, alpha=0.5, label="P_m2")
    ax.set_ylabel("Pressure [kPa]")
    ax.set_title("Chamber Pressures (RL)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t_rl, h_rl["F_h"], "b-", lw=1, label="F_h (human)")
    ax.plot(t_rl, h_rl["F_e"], "r-", lw=1, label="F_e (env, RL)")
    ax.plot(t_bl, h_bl["F_e"], "r--", lw=0.8, alpha=0.5, label="F_e (env, baseline)")
    ax.set_ylabel("Force [N]")
    ax.set_title("Forces")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(t_rl, h_rl["u_v"], "g-", lw=1, label="u_v [V]")
    ax2 = ax.twinx()
    ax2.plot(t_rl, h_rl["x_v"], "m-", lw=0.8, alpha=0.7, label="x_v (spool)")
    ax.set_ylabel("Valve Voltage [V]")
    ax2.set_ylabel("Spool Position")
    ax.set_title("Valve Control Signal (RL)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    pe_rl = np.array(h_rl["pos_error"]) * 1000
    pe_bl = np.array(h_bl["pos_error"]) * 1000
    ax.plot(t_rl, pe_rl, "b-", lw=1.5, label=f"RL (RMSE={np.sqrt(np.mean(pe_rl**2)):.1f} mm)")
    ax.plot(t_bl, pe_bl, "r--", lw=1, alpha=0.7, label=f"Baseline (RMSE={np.sqrt(np.mean(pe_bl**2)):.1f} mm)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Tracking Error [mm]")
    ax.set_title("Tracking Error (x_m - x_s)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[5]
    power_err_rl = np.array(h_rl["F_e"]) * np.array(h_rl["v_m"]) - np.array(h_rl["F_h"]) * np.array(h_rl["v_s"])
    power_err_bl = np.array(h_bl["F_e"]) * np.array(h_bl["v_m"]) - np.array(h_bl["F_h"]) * np.array(h_bl["v_s"])
    ax.plot(t_rl, power_err_rl, "b-", lw=1.5, label=f"RL (RMSE={np.sqrt(np.mean(power_err_rl**2)):.2f} W)")
    ax.plot(t_bl, power_err_bl, "r--", lw=1, alpha=0.7, label=f"Baseline (RMSE={np.sqrt(np.mean(power_err_bl**2)):.2f} W)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Power Error [W]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Transparency Error (F_e*v_m - F_h*v_s)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(WD, "results", "plots", "episode_comparison.png")
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()

    rmse_rl = np.sqrt(np.mean(pe_rl**2))
    rmse_bl = np.sqrt(np.mean(pe_bl**2))
    print(f"\nRL RMSE:       {rmse_rl:.2f} mm")
    print(f"Baseline RMSE: {rmse_bl:.2f} mm")
    print(f"Improvement:   {(1 - rmse_rl / rmse_bl) * 100:.1f}%")
    print("\nDone!")


if __name__ == "__main__":
    generate_plots()
