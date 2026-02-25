"""Train Q-learning on the new 6-D observation state for 10k episodes and plot results."""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from q_learning_agent import QLearningAgent
from teleop_env import TeleopEnv


TOTAL_TRAIN_EPISODES = 10_000
ENV_MODE = cfg.ENV_MODE_CHANGING
OUT_DIR_NAME = os.path.join(cfg.RL_CHANGING_DIR, "rl_new_obs_10k_episodes")
MOVING_AVG_WINDOW = 200
TRACKER_WINDOW = 100
PRINT_EVERY = 100


def _mk_dirs() -> dict[str, str]:
    base = os.path.join(os.path.dirname(__file__), cfg.RESULTS_ROOT_DIR, OUT_DIR_NAME)
    paths = {
        "base": base,
        "models": os.path.join(base, "models"),
        "logs": os.path.join(base, "logs"),
        "plots": os.path.join(base, "plots"),
        "episodes": os.path.join(base, "episodes"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x
    window = int(max(1, min(window, x.size)))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def _greedy_action(q_values: np.ndarray, zero_action: int) -> int:
    max_q = q_values.max()
    best = np.flatnonzero(q_values == max_q)
    return zero_action if zero_action in best else int(best[0])


def _print_tracker(
    ep_idx: int,
    total_episodes: int,
    total_steps: int,
    agent: QLearningAgent,
    episode_returns: np.ndarray,
    episode_tracking_rmse: np.ndarray,
    episode_transparency_rmse: np.ndarray,
) -> None:
    end = ep_idx + 1
    start = max(0, end - TRACKER_WINDOW)
    mean_return = float(np.mean(episode_returns[start:end]))
    mean_track_mm = float(np.mean(episode_tracking_rmse[start:end]) * 1000.0)
    mean_transparency = float(np.mean(episode_transparency_rmse[start:end]))
    print(
        f"[train] ep {end:>5}/{total_episodes} | steps {total_steps:>8} | "
        f"eps {agent.epsilon:.4f} | avgR({end-start}) {mean_return:+9.3f} | "
        f"avgTE {mean_track_mm:7.3f} mm | avgTrE {mean_transparency:7.4f} W | "
        f"states {agent.discovered_states():>6} | cov {agent.coverage():6.2%}"
    )


def _train_for_episodes(
    total_episodes: int,
    env_mode: str,
    print_every: int,
) -> tuple[QLearningAgent, dict[str, np.ndarray]]:
    env = TeleopEnv(env_mode=env_mode)
    state_dims = env.get_state_dims()
    agent = QLearningAgent(state_dims=state_dims, n_actions=cfg.N_ACTIONS, seed=42)

    episode_returns = np.zeros(total_episodes, dtype=np.float64)
    episode_steps = np.zeros(total_episodes, dtype=np.int64)
    episode_tracking_rmse = np.zeros(total_episodes, dtype=np.float64)
    episode_transparency_rmse = np.zeros(total_episodes, dtype=np.float64)

    total_steps = 0
    for ep in range(total_episodes):
        obs, _ = env.reset(seed=ep)
        state = env.discretise_obs(obs)
        done = False
        ep_return = 0.0
        ep_step_count = 0

        while not done:
            action = agent.select_action(state)
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = env.discretise_obs(obs_next)

            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_return += reward
            ep_step_count += 1
            total_steps += 1

        h = env.render() or {}
        pe = np.array(h.get("pos_error", []), dtype=np.float64)
        te = np.array(h.get("transparency_error", []), dtype=np.float64)
        episode_returns[ep] = ep_return
        episode_steps[ep] = ep_step_count
        episode_tracking_rmse[ep] = float(np.sqrt(np.mean(pe ** 2))) if pe.size else np.nan
        episode_transparency_rmse[ep] = float(np.sqrt(np.mean(te ** 2))) if te.size else np.nan

        agent.decay_epsilon()

        should_print = (
            ep == 0
            or (ep + 1) % max(1, print_every) == 0
            or (ep + 1) == total_episodes
        )
        if should_print:
            _print_tracker(
                ep_idx=ep,
                total_episodes=total_episodes,
                total_steps=total_steps,
                agent=agent,
                episode_returns=episode_returns,
                episode_tracking_rmse=episode_tracking_rmse,
                episode_transparency_rmse=episode_transparency_rmse,
            )

    train_log = {
        "episode_returns": episode_returns,
        "episode_steps": episode_steps,
        "episode_tracking_rmse": episode_tracking_rmse,
        "episode_transparency_rmse": episode_transparency_rmse,
        "total_steps": np.array([total_steps], dtype=np.int64),
    }
    return agent, train_log


def _evaluate_greedy_episode(agent: QLearningAgent, env_mode: str) -> dict:
    env = TeleopEnv(env_mode=env_mode)
    zero_action = int(np.argmin(np.abs(cfg.V_LEVELS)))

    obs, _ = env.reset(seed=123)
    state = env.discretise_obs(obs)
    done = False

    while not done:
        q_values = agent.q_values(state)
        action = _greedy_action(q_values, zero_action=zero_action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = env.discretise_obs(obs)

    return env.render() or {}


def _build_policy_map(agent: QLearningAgent) -> np.ndarray:
    n_slave, n_master, *_ = agent.state_dims

    zero_action = int(np.argmin(np.abs(cfg.V_LEVELS)))
    z_ps1 = int(np.digitize(cfg.P_ATM, cfg.SLAVE_P1_BINS))
    z_ps2 = int(np.digitize(cfg.P_ATM, cfg.SLAVE_P2_BINS))
    z_pm1 = int(np.digitize(cfg.P_ATM, cfg.MASTER_P1_BINS))
    z_pm2 = int(np.digitize(cfg.P_ATM, cfg.MASTER_P2_BINS))
    z_f1 = int(np.digitize(0.0, cfg.MASS_FLOW1_BINS))
    z_f2 = int(np.digitize(0.0, cfg.MASS_FLOW2_BINS))

    policy_map = np.zeros((n_slave, n_master), dtype=np.int64)
    for i in range(n_slave):
        for j in range(n_master):
            state = (i, j, z_ps1, z_ps2, z_pm1, z_pm2, z_f1, z_f2)
            q_values = agent.q_values(state)
            policy_map[i, j] = _greedy_action(q_values, zero_action=zero_action)
    return policy_map


def _save_reward_plot(train_log: dict[str, np.ndarray], out_path: str) -> None:
    rewards = train_log["episode_returns"]
    r_ma = _moving_average(rewards, MOVING_AVG_WINDOW)
    episodes = np.arange(1, rewards.size + 1, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(episodes, rewards, lw=0.8, alpha=0.35, color="tab:blue", label="Episode return")
    ax.plot(episodes, r_ma, lw=1.8, color="tab:red", label=f"Moving average ({MOVING_AVG_WINDOW})")
    ax.set_title("Reward During 10k-Episode Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_tracking_plot(train_log: dict[str, np.ndarray], out_path: str) -> None:
    trk = train_log["episode_tracking_rmse"] * 1000.0
    trk_ma = _moving_average(trk, MOVING_AVG_WINDOW)
    episodes = np.arange(1, trk.size + 1, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(episodes, trk, lw=0.8, alpha=0.35, color="tab:green", label="Episode tracking RMSE")
    ax.plot(episodes, trk_ma, lw=1.8, color="tab:olive", label=f"Moving average ({MOVING_AVG_WINDOW})")
    ax.set_title("Tracking Error During Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Tracking RMSE [mm]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_transparency_plot(train_log: dict[str, np.ndarray], out_path: str) -> None:
    tr = train_log["episode_transparency_rmse"]
    tr_ma = _moving_average(tr, MOVING_AVG_WINDOW)
    episodes = np.arange(1, tr.size + 1, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(episodes, tr, lw=0.8, alpha=0.35, color="tab:purple", label="Episode transparency RMSE")
    ax.plot(episodes, tr_ma, lw=1.8, color="tab:pink", label=f"Moving average ({MOVING_AVG_WINDOW})")
    ax.set_title("Transparency Error During Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Transparency RMSE [W]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_policy_plot(policy_map: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(
        policy_map,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=cfg.N_ACTIONS - 1,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Greedy action index")
    cbar.set_ticks(np.arange(cfg.N_ACTIONS))
    ax.set_title("Learned Policy Map (fixed pressure/flow at nominal bins)")
    ax.set_xlabel("Master position-error bin")
    ax.set_ylabel("Slave position-error bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_q_learning_new_obs_10k(total_episodes: int = TOTAL_TRAIN_EPISODES) -> None:
    print(
        f"Starting Q-learning training with new observation space | "
        f"episodes={total_episodes} | env_mode={ENV_MODE}"
    )
    paths = _mk_dirs()
    agent, train_log = _train_for_episodes(
        total_episodes=total_episodes,
        env_mode=ENV_MODE,
        print_every=PRINT_EVERY,
    )
    eval_history = _evaluate_greedy_episode(agent, env_mode=ENV_MODE)
    policy_map = _build_policy_map(agent)

    model_path = os.path.join(paths["models"], "q_table_10k_episodes.npy")
    agent.save(model_path)

    np.savez(
        os.path.join(paths["logs"], "training_log_10k_episodes.npz"),
        **train_log,
    )
    np.savez(
        os.path.join(paths["episodes"], "greedy_eval_episode.npz"),
        **{k: np.array(v, dtype=object) for k, v in eval_history.items()},
    )
    np.save(
        os.path.join(paths["logs"], "policy_map_action_idx.npy"),
        policy_map,
        allow_pickle=False,
    )

    _save_tracking_plot(
        train_log=train_log,
        out_path=os.path.join(paths["plots"], "tracking_plot.png"),
    )
    _save_reward_plot(
        train_log=train_log,
        out_path=os.path.join(paths["plots"], "reward_plot.png"),
    )
    _save_transparency_plot(
        train_log=train_log,
        out_path=os.path.join(paths["plots"], "transparency_plot.png"),
    )
    _save_policy_plot(
        policy_map=policy_map,
        out_path=os.path.join(paths["plots"], "learned_policy_plot.png"),
    )

    pe_eval = np.array(eval_history.get("pos_error", []), dtype=np.float64)
    tr_eval = np.array(eval_history.get("transparency_error", []), dtype=np.float64)
    tracking_rmse_eval = float(np.sqrt(np.mean(pe_eval ** 2))) if pe_eval.size else float("nan")
    transparency_rmse_eval = float(np.sqrt(np.mean(tr_eval ** 2))) if tr_eval.size else float("nan")
    mean_ep_reward = float(np.mean(train_log["episode_returns"]))
    mean_last_reward = float(np.mean(train_log["episode_returns"][-TRACKER_WINDOW:]))

    summary_path = os.path.join(paths["logs"], "summary_10k_episodes.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Q-learning with new observation space (10k training episodes)\n")
        f.write(f"env_mode={ENV_MODE}\n")
        f.write(f"total_train_episodes={total_episodes}\n")
        f.write(f"total_env_steps={int(train_log['total_steps'][0])}\n")
        f.write(f"state_dims={agent.state_dims}\n")
        f.write(f"n_actions={agent.n_actions}\n")
        f.write(f"discovered_states={agent.discovered_states()}\n")
        f.write(f"action_coverage={agent.coverage():.6f}\n")
        f.write(f"final_epsilon={agent.epsilon:.6f}\n")
        f.write(f"mean_episode_return={mean_ep_reward:.8f}\n")
        f.write(f"mean_episode_return_last_{TRACKER_WINDOW}={mean_last_reward:.8f}\n")
        f.write(f"eval_tracking_rmse_m={tracking_rmse_eval:.8f}\n")
        f.write(f"eval_transparency_rmse_w={transparency_rmse_eval:.8f}\n")

    print("Training complete.")
    print(f"Results: {paths['base']}")
    print(f"Plots: {paths['plots']}")
    print(f"Model: {model_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    run_q_learning_new_obs_10k()
