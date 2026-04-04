"""Train/evaluate DQN, Q-learning, and MRAC agents — 3-way comparison."""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import config as cfg
    from .dqn_agent import DQNAgent
    from .mrac_controller import FilteredMRACController, baseline_mrac_inputs
    from .q_learning_agent import QLearningAgent
    from .teleop_env import TeleopEnv
except ImportError:  # pragma: no cover - direct script execution
    import config as cfg
    from dqn_agent import DQNAgent
    from mrac_controller import FilteredMRACController, baseline_mrac_inputs
    from q_learning_agent import QLearningAgent
    from teleop_env import TeleopEnv


@dataclass
class EvalResult:
    mean_reward: float
    tracking_rmse_m: float
    transparency_rmse: float
    history: dict


def _mk_agent_dirs(agent_name: str) -> dict[str, str]:
    base = os.path.join(os.path.dirname(__file__), cfg.RESULTS_ROOT_DIR, agent_name)
    paths = {k: os.path.join(base, k) for k in ("models", "logs", "plots", "episodes")}
    paths["base"] = base
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def _mk_compare_dir() -> str:
    p = os.path.join(os.path.dirname(__file__), cfg.RESULTS_ROOT_DIR, cfg.COMPARE_RESULTS_DIR)
    os.makedirs(p, exist_ok=True)
    return p


def _greedy_action(q_values: np.ndarray, zero_action: int) -> int:
    max_q = q_values.max()
    best = np.flatnonzero(q_values == max_q)
    return zero_action if zero_action in best else int(best[0])


# ------------------------------------------------------------------ #
#  Evaluation helpers                                                  #
# ------------------------------------------------------------------ #

def _eval_episode_stats(env: TeleopEnv) -> tuple[float, float, float, dict]:
    """Extract reward/tracking/transparency from a completed episode."""
    h = env.render() or {}
    pe = np.asarray(h.get("pos_error", []), dtype=np.float64)
    te = np.asarray(h.get("transparency_error", []), dtype=np.float64)
    rw = np.asarray(h.get("reward", []), dtype=np.float64)
    return (
        float(rw.sum()) if rw.size else 0.0,
        float(np.sqrt(np.mean(pe**2))) if pe.size else 0.0,
        float(np.sqrt(np.mean(te**2))) if te.size else 0.0,
        h,
    )


def _evaluate_dqn(agent: DQNAgent, env_mode: str,
                  master_input_mode: str,
                  n_episodes: int = 1) -> EvalResult:
    rewards, trk, trn = [], [], []
    rep_hist: dict = {}
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    for ep in range(n_episodes):
        env = TeleopEnv(env_mode=env_mode, master_input_mode=master_input_mode)
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        r, t, tr, h = _eval_episode_stats(env)
        rewards.append(r); trk.append(t); trn.append(tr)
        if ep == 0:
            rep_hist = h
    agent.epsilon = old_eps
    return EvalResult(float(np.mean(rewards)), float(np.mean(trk)),
                      float(np.mean(trn)), rep_hist)


def _evaluate_ql(agent: QLearningAgent, env_mode: str,
                 master_input_mode: str,
                 n_episodes: int = 1) -> EvalResult:
    rewards, trk, trn = [], [], []
    rep_hist: dict = {}
    zero_action = int(np.argmin(np.abs(cfg.V_LEVELS)))
    for ep in range(n_episodes):
        env = TeleopEnv(env_mode=env_mode, master_input_mode=master_input_mode)
        obs, _ = env.reset(seed=ep)
        state = env.discretise_obs_reduced(obs)
        done = False
        while not done:
            action = _greedy_action(agent.q_values(state), zero_action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = env.discretise_obs_reduced(obs)
        r, t, tr, h = _eval_episode_stats(env)
        rewards.append(r); trk.append(t); trn.append(tr)
        if ep == 0:
            rep_hist = h
    return EvalResult(float(np.mean(rewards)), float(np.mean(trk)),
                      float(np.mean(trn)), rep_hist)


def _evaluate_mrac(env_mode: str, master_input_mode: str,
                   n_episodes: int = 1) -> EvalResult:
    rewards, trk, trn = [], [], []
    rep_hist: dict = {}
    for ep in range(n_episodes):
        env = TeleopEnv(env_mode=env_mode, master_input_mode=master_input_mode)
        ctrl = FilteredMRACController()
        ctrl.reset()
        obs, info = env.reset(seed=ep)
        done = False
        while not done:
            y, u_c = baseline_mrac_inputs(info)
            u_v = ctrl.step_voltage(pos_error=y, u_c=u_c)
            obs, _, terminated, truncated, info = env.step_voltage(u_v)
            done = terminated or truncated
        r, t, tr, h = _eval_episode_stats(env)
        rewards.append(r); trk.append(t); trn.append(tr)
        if ep == 0:
            rep_hist = h
    return EvalResult(float(np.mean(rewards)), float(np.mean(trk)),
                      float(np.mean(trn)), rep_hist)


# ------------------------------------------------------------------ #
#  Training wrappers                                                   #
# ------------------------------------------------------------------ #

def _train_dqn(env_mode: str, out_name: str,
               master_input_mode: str,
               num_episodes: int = cfg.DQN_NUM_EPISODES) -> tuple[DQNAgent, EvalResult]:
    dirs = _mk_agent_dirs(out_name)
    env = TeleopEnv(env_mode=env_mode, master_input_mode=master_input_mode)
    agent = DQNAgent(obs_dim=10, n_actions=cfg.N_ACTIONS, seed=42)

    ep_rewards = np.zeros(num_episodes, dtype=np.float64)
    print(f"Training DQN ({out_name}) for {num_episodes} episodes …")

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done, ep_ret = False, 0.0
        while not done:
            a = agent.select_action(obs)
            nobs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.store_transition(obs, a, r, nobs, done)
            agent.train_step()
            obs = nobs
            ep_ret += r
        agent.decay_epsilon()
        ep_rewards[ep] = ep_ret

        if (ep + 1) % 100 == 0 or (ep + 1) == num_episodes:
            w = min(ep + 1, 200)
            print(f"  [{out_name}] ep {ep+1:>5} | avgR({w}) {np.mean(ep_rewards[max(0,ep+1-w):ep+1]):+.2f} | eps {agent.epsilon:.4f}")

    agent.save(os.path.join(dirs["models"], "dqn_model.pt"))
    np.savez(os.path.join(dirs["logs"], "training_log.npz"), episode_rewards=ep_rewards)
    _save_training_curve(ep_rewards, f"DQN ({out_name})",
                         os.path.join(dirs["plots"], "training_curve.png"))

    # -- Post-training test: 100 greedy episodes --
    print(f"  [{out_name}] Running 100-episode greedy test …")
    test_ev = _evaluate_dqn(agent, env_mode, master_input_mode, n_episodes=100)
    print(f"  [{out_name}] TEST  avgR={test_ev.mean_reward:+.2f} | "
          f"track={test_ev.tracking_rmse_m*1000:.3f} mm | "
          f"transp={test_ev.transparency_rmse:.5f} W")
    with open(os.path.join(dirs["logs"], "test_100ep.txt"), "w") as f:
        f.write(f"master_input_mode={master_input_mode}\n")
        f.write(f"mean_reward={test_ev.mean_reward:.6f}\n")
        f.write(f"tracking_rmse_mm={test_ev.tracking_rmse_m*1000:.6f}\n")
        f.write(f"transparency_rmse_w={test_ev.transparency_rmse:.6f}\n")

    ev = _evaluate_dqn(agent, env_mode, master_input_mode, n_episodes=cfg.DQN_EVAL_EPISODES)
    np.savez(os.path.join(dirs["episodes"], "eval_episode.npz"),
             **{k: np.array(v, dtype=object) for k, v in ev.history.items()})
    return agent, ev


def _train_ql(env_mode: str, out_name: str,
              master_input_mode: str,
              num_episodes: int = cfg.NUM_EPISODES) -> tuple[QLearningAgent, EvalResult]:
    dirs = _mk_agent_dirs(out_name)
    env = TeleopEnv(env_mode=env_mode, master_input_mode=master_input_mode)
    state_dims = env.get_state_dims_reduced()
    agent = QLearningAgent(state_dims, cfg.N_ACTIONS, seed=42)

    ep_rewards = np.zeros(num_episodes, dtype=np.float64)
    print(f"Training Q-learning ({out_name}) for {num_episodes} episodes …")

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        state = env.discretise_obs_reduced(obs)
        done, ep_ret = False, 0.0
        while not done:
            a = agent.select_action(state)
            nobs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ns = env.discretise_obs_reduced(nobs)
            agent.update(state, a, r, ns, done)
            state = ns
            ep_ret += r
        agent.decay_epsilon()
        ep_rewards[ep] = ep_ret

        if (ep + 1) % 100 == 0 or (ep + 1) == num_episodes:
            w = min(ep + 1, 200)
            print(f"  [{out_name}] ep {ep+1:>5} | avgR({w}) {np.mean(ep_rewards[max(0,ep+1-w):ep+1]):+.2f} | eps {agent.epsilon:.4f} | states {agent.discovered_states()}")

    agent.save(os.path.join(dirs["models"], "q_table.npy"))
    np.savez(os.path.join(dirs["logs"], "training_log.npz"), episode_rewards=ep_rewards)
    _save_training_curve(ep_rewards, f"Q-learning ({out_name})",
                         os.path.join(dirs["plots"], "training_curve.png"))

    # -- Post-training test: 100 greedy episodes --
    print(f"  [{out_name}] Running 100-episode greedy test …")
    test_ev = _evaluate_ql(agent, env_mode, master_input_mode, n_episodes=100)
    print(f"  [{out_name}] TEST  avgR={test_ev.mean_reward:+.2f} | "
          f"track={test_ev.tracking_rmse_m*1000:.3f} mm | "
          f"transp={test_ev.transparency_rmse:.5f} W")
    with open(os.path.join(dirs["logs"], "test_100ep.txt"), "w") as f:
        f.write(f"master_input_mode={master_input_mode}\n")
        f.write(f"mean_reward={test_ev.mean_reward:.6f}\n")
        f.write(f"tracking_rmse_mm={test_ev.tracking_rmse_m*1000:.6f}\n")
        f.write(f"transparency_rmse_w={test_ev.transparency_rmse:.6f}\n")

    ev = _evaluate_ql(agent, env_mode, master_input_mode, n_episodes=cfg.EVAL_EPISODES)
    np.savez(os.path.join(dirs["episodes"], "eval_episode.npz"),
             **{k: np.array(v, dtype=object) for k, v in ev.history.items()})
    return agent, ev


# ------------------------------------------------------------------ #
#  Training curve plot                                                 #
# ------------------------------------------------------------------ #

def _save_training_curve(ep_rewards: np.ndarray, agent_label: str,
                         out_path: str, window: int = 100) -> None:
    """Save a smoothed reward-per-episode training curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(1, len(ep_rewards) + 1)
    ax.plot(episodes, ep_rewards, alpha=0.25, color="tab:blue", lw=0.6, label="Episode reward")
    if len(ep_rewards) >= window:
        smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1:], smoothed, color="tab:blue", lw=1.8,
                label=f"{window}-ep moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(f"{agent_label} — Training Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ #
#  Comparison plots                                                    #
# ------------------------------------------------------------------ #

def _save_comparison(results: dict[str, EvalResult], title: str,
                     out_path: str) -> None:
    """Plot tracking and transparency error for multiple agents."""
    colors = {"DQN": "tab:blue", "Q-learning": "tab:green", "MRAC": "tab:orange"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for name, ev in results.items():
        t = np.asarray(ev.history.get("time", []), dtype=np.float64)
        pe = np.asarray(ev.history.get("pos_error", []), dtype=np.float64) * 1000
        tr = np.asarray(ev.history.get("transparency_error", []), dtype=np.float64)
        if t.size == 0:
            continue
        c = colors.get(name, "gray")
        rmse_pe = float(np.sqrt(np.mean(pe**2)))
        rmse_tr = float(np.sqrt(np.mean(tr**2)))
        axes[0].plot(t, pe, label=f"{name} ({rmse_pe:.2f} mm)", color=c, lw=1.4, alpha=0.85)
        axes[1].plot(t, tr, label=f"{name} ({rmse_tr:.3f} W)",  color=c, lw=1.4, alpha=0.85)

    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_ylabel("Tracking Error [mm]"); axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[1].axhline(0, color="gray", lw=0.6)
    axes[1].set_ylabel("Transparency Error [W]"); axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ #
#  Main benchmark                                                      #
# ------------------------------------------------------------------ #

def run_full_benchmark() -> None:
    master_input_mode = cfg.DEFAULT_MASTER_INPUT_MODE
    print(f"Running benchmark with master_input_mode={master_input_mode} | results_root={cfg.RESULTS_ROOT_DIR}")

    # ---- 1. Train DQN ----
    _, dqn_const = _train_dqn(cfg.ENV_MODE_CONSTANT, cfg.DQN_CONSTANT_DIR, master_input_mode)
    _, dqn_chg   = _train_dqn(cfg.ENV_MODE_CHANGING,  cfg.DQN_CHANGING_DIR, master_input_mode)

    # ---- 2. Train Q-learning ----
    _, ql_const = _train_ql(cfg.ENV_MODE_CONSTANT, cfg.Q_LEARNING_CONSTANT_DIR, master_input_mode)
    _, ql_chg   = _train_ql(cfg.ENV_MODE_CHANGING,  cfg.Q_LEARNING_CHANGING_DIR, master_input_mode)

    # ---- 3. MRAC baselines ----
    mrac_dirs = _mk_agent_dirs(cfg.MRAC_RESULTS_DIR)
    mrac_const = _evaluate_mrac(cfg.ENV_MODE_CONSTANT, master_input_mode, n_episodes=cfg.EVAL_EPISODES)
    mrac_chg   = _evaluate_mrac(cfg.ENV_MODE_CHANGING,  master_input_mode, n_episodes=cfg.EVAL_EPISODES)

    np.savez(os.path.join(mrac_dirs["episodes"], "constant_eval.npz"),
             **{k: np.array(v, dtype=object) for k, v in mrac_const.history.items()})
    np.savez(os.path.join(mrac_dirs["episodes"], "changing_eval.npz"),
             **{k: np.array(v, dtype=object) for k, v in mrac_chg.history.items()})
    with open(os.path.join(mrac_dirs["logs"], "mrac_metrics.txt"), "w") as f:
        f.write(f"master_input_mode={master_input_mode}\n")
        for label, ev in [("constant", mrac_const), ("changing", mrac_chg)]:
            f.write(f"MRAC {label}: R={ev.mean_reward:.3f}, "
                    f"TE={ev.tracking_rmse_m*1000:.3f} mm, "
                    f"TrE={ev.transparency_rmse:.5f} W\n")

    # ---- 4. Comparison plots ----
    cmp = _mk_compare_dir()
    _save_comparison(
        {"DQN": dqn_const, "Q-learning": ql_const, "MRAC": mrac_const},
        "Constant Environment (F_h input): DQN vs Q-learning vs MRAC",
        os.path.join(cmp, "comparison_constant.png"),
    )
    _save_comparison(
        {"DQN": dqn_chg, "Q-learning": ql_chg, "MRAC": mrac_chg},
        "Changing Environment (F_h input): DQN vs Q-learning vs MRAC",
        os.path.join(cmp, "comparison_changing.png"),
    )

    # ---- 5. Summary CSV ----
    with open(os.path.join(cmp, "summary.csv"), "w") as f:
        f.write("master_input_mode,case,agent,mean_reward,tracking_rmse_mm,transparency_rmse_w\n")
        for label, evs in [("constant", {"DQN": dqn_const, "Q-learning": ql_const, "MRAC": mrac_const}),
                           ("changing", {"DQN": dqn_chg,   "Q-learning": ql_chg,   "MRAC": mrac_chg})]:
            for name, ev in evs.items():
                f.write(f"{master_input_mode},{label},{name},{ev.mean_reward:.6f},"
                        f"{ev.tracking_rmse_m*1000:.6f},{ev.transparency_rmse:.6f}\n")

    print("\nBenchmark complete.")
    print(f"Comparison plots → {os.path.join(cfg.RESULTS_ROOT_DIR, cfg.COMPARE_RESULTS_DIR)}")


if __name__ == "__main__":
    run_full_benchmark()
