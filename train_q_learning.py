"""Train a tabular Q-learning agent with reduced 4-D state space."""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import config as cfg
    from .q_learning_agent import QLearningAgent
    from .teleop_env import TeleopEnv
except ImportError:  # pragma: no cover - direct script execution
    import config as cfg
    from q_learning_agent import QLearningAgent
    from teleop_env import TeleopEnv

MOVING_AVG_WINDOW = 200
PRINT_EVERY       = 100


def _out_dir(env_mode: str) -> str:
    if env_mode == cfg.ENV_MODE_CHANGING:
        return cfg.Q_LEARNING_CHANGING_DIR
    return cfg.Q_LEARNING_CONSTANT_DIR


def _mk_dirs(out_name: str) -> dict[str, str]:
    base = os.path.join(os.path.dirname(__file__), cfg.RESULTS_ROOT_DIR, out_name)
    paths = {k: os.path.join(base, k) for k in ("models", "logs", "plots", "episodes")}
    paths["base"] = base
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, min(w, x.size))
    return np.convolve(x, np.ones(w) / w, mode="same")


def _greedy_action(q_values: np.ndarray, zero_action: int) -> int:
    max_q = q_values.max()
    best = np.flatnonzero(q_values == max_q)
    return zero_action if zero_action in best else int(best[0])


# ------------------------------------------------------------------ #

def train_q_learning(
    total_episodes: int = cfg.NUM_EPISODES,
    env_mode: str = cfg.ENV_MODE_CHANGING,
    master_input_mode: str = cfg.DEFAULT_MASTER_INPUT_MODE,
    env_cls=None,
    env_kwargs: dict | None = None,
    results_dir_name: str | None = None,
) -> None:
    out_name = results_dir_name or _out_dir(env_mode)
    paths = _mk_dirs(out_name)
    env_cls = TeleopEnv if env_cls is None else env_cls
    env_kwargs = dict(env_kwargs or {})

    env = env_cls(env_mode=env_mode, master_input_mode=master_input_mode, **env_kwargs)
    state_dims = env.get_state_dims_reduced()
    agent = QLearningAgent(state_dims=state_dims, n_actions=cfg.N_ACTIONS, seed=42)

    print(
        f"Q-learning training (reduced 4-D) | episodes={total_episodes} | "
        f"env_mode={env_mode} | master_input_mode={master_input_mode} | state_dims={state_dims}"
    )
    print(f"Output -> {paths['base']}")

    ep_returns = np.zeros(total_episodes, dtype=np.float64)
    ep_track_rmse = np.zeros(total_episodes, dtype=np.float64)
    ep_transp_rmse = np.zeros(total_episodes, dtype=np.float64)
    total_steps = 0

    for ep in range(total_episodes):
        obs, _ = env.reset(seed=ep)
        state = env.discretise_obs_reduced(obs)
        done = False
        ep_ret = 0.0

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = env.discretise_obs_reduced(next_obs)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_ret += reward
            total_steps += 1

        agent.decay_epsilon()

        h = env.render() or {}
        pe = np.asarray(h.get("pos_error", []), dtype=np.float64)
        te = np.asarray(h.get("transparency_error", []), dtype=np.float64)
        ep_returns[ep] = ep_ret
        ep_track_rmse[ep] = float(np.sqrt(np.mean(pe ** 2))) if pe.size else 0.0
        ep_transp_rmse[ep] = float(np.sqrt(np.mean(te ** 2))) if te.size else 0.0

        if ep == 0 or (ep + 1) % PRINT_EVERY == 0 or (ep + 1) == total_episodes:
            w = min(ep + 1, 100)
            s = max(0, ep + 1 - w)
            print(
                f"[Q-LR] ep {ep+1:>5}/{total_episodes} | "
                f"eps {agent.epsilon:.4f} | "
                f"avgR({w}) {np.mean(ep_returns[s:ep+1]):+9.2f} | "
                f"TE {np.mean(ep_track_rmse[s:ep+1]) * 1000:7.2f} mm | "
                f"TrE {np.mean(ep_transp_rmse[s:ep+1]):7.4f} W | "
                f"states {agent.discovered_states()} | "
                f"cov {agent.coverage():.1%}"
            )

    agent.save(os.path.join(paths["models"], "q_table.npy"))
    np.savez(
        os.path.join(paths["logs"], "training_log.npz"),
        episode_returns=ep_returns,
        episode_tracking_rmse=ep_track_rmse,
        episode_transparency_rmse=ep_transp_rmse,
    )

    eval_hist = _evaluate_greedy(
        agent,
        env_mode,
        master_input_mode,
        env_cls=env_cls,
        env_kwargs=env_kwargs,
    )
    np.savez(
        os.path.join(paths["episodes"], "greedy_eval_episode.npz"),
        **{k: np.array(v, dtype=object) for k, v in eval_hist.items()},
    )

    _save_plots(ep_returns, ep_track_rmse, ep_transp_rmse, paths["plots"])

    pe_ev = np.asarray(eval_hist.get("pos_error", []), dtype=np.float64)
    te_ev = np.asarray(eval_hist.get("transparency_error", []), dtype=np.float64)
    with open(os.path.join(paths["logs"], "summary.txt"), "w") as f:
        f.write(f"env_mode={env_mode}\n")
        f.write(f"master_input_mode={master_input_mode}\n")
        f.write(f"results_root={cfg.RESULTS_ROOT_DIR}\n")
        f.write(f"total_episodes={total_episodes}\n")
        f.write(f"total_env_steps={total_steps}\n")
        f.write(f"state_dims={state_dims}\n")
        f.write(f"discovered_states={agent.discovered_states()}\n")
        f.write(f"action_coverage={agent.coverage():.6f}\n")
        f.write(f"final_epsilon={agent.epsilon:.6f}\n")
        f.write(f"eval_tracking_rmse_m={float(np.sqrt(np.mean(pe_ev ** 2))) if pe_ev.size else float('nan'):.8f}\n")
        f.write(f"eval_transparency_rmse_w={float(np.sqrt(np.mean(te_ev ** 2))) if te_ev.size else float('nan'):.8f}\n")

    print("Training complete.")


# ------------------------------------------------------------------ #

def _evaluate_greedy(agent: QLearningAgent,
                     env_mode: str, master_input_mode: str,
                     env_cls=None,
                     env_kwargs: dict | None = None) -> dict:
    env_cls = TeleopEnv if env_cls is None else env_cls
    env_kwargs = dict(env_kwargs or {})
    env = env_cls(env_mode=env_mode, master_input_mode=master_input_mode, **env_kwargs)
    zero_action = int(np.argmin(np.abs(cfg.V_LEVELS)))
    obs, _ = env.reset(seed=123)
    state = env.discretise_obs_reduced(obs)
    done = False
    while not done:
        q_vals = agent.q_values(state)
        action = _greedy_action(q_vals, zero_action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = env.discretise_obs_reduced(obs)
    return env.render() or {}


def _save_plots(returns, track, transp, plot_dir):
    eps = np.arange(1, len(returns) + 1)
    ma_r = _moving_avg(returns, MOVING_AVG_WINDOW)
    ma_t = _moving_avg(track * 1000, MOVING_AVG_WINDOW)
    ma_tr = _moving_avg(transp, MOVING_AVG_WINDOW)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(eps, returns, lw=0.6, alpha=0.3, color="tab:blue")
    axes[0].plot(eps, ma_r, lw=1.8, color="tab:red", label=f"MA({MOVING_AVG_WINDOW})")
    axes[0].set_ylabel("Return"); axes[0].set_title("Q-learning: Reward"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps, track * 1000, lw=0.6, alpha=0.3, color="tab:green")
    axes[1].plot(eps, ma_t, lw=1.8, color="tab:olive", label=f"MA({MOVING_AVG_WINDOW})")
    axes[1].set_ylabel("mm"); axes[1].set_title("Q-learning: Tracking RMSE"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(eps, transp, lw=0.6, alpha=0.3, color="tab:purple")
    axes[2].plot(eps, ma_tr, lw=1.8, color="tab:pink", label=f"MA({MOVING_AVG_WINDOW})")
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("W"); axes[2].set_title("Q-learning: Transparency RMSE"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-learning (reduced 4-D state).")
    parser.add_argument("--env-mode",
                        choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING],
                        default=cfg.ENV_MODE_CHANGING)
    parser.add_argument("--master-input-mode",
                        choices=[cfg.MASTER_INPUT_REFERENCE, cfg.MASTER_INPUT_FORCE],
                        default=cfg.DEFAULT_MASTER_INPUT_MODE)
    parser.add_argument("--episodes", type=int, default=cfg.NUM_EPISODES)
    args = parser.parse_args()
    train_q_learning(
        total_episodes=args.episodes,
        env_mode=args.env_mode,
        master_input_mode=args.master_input_mode,
    )
