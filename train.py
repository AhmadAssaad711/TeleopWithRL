"""
Training script - Q-learning for bilateral pneumatic teleoperation.

The RL agent outputs the servo-valve voltage u_v to make the slave
piston track the master through passive pneumatic tube coupling.

Usage:
    python train.py
"""

from __future__ import annotations

import os
import time

import numpy as np
try:
    from tqdm import trange
except ImportError:  # fallback when tqdm is not installed
    def trange(n: int, desc: str | None = None):
        return range(n)

import config as cfg
from teleop_env import TeleopEnv
from q_learning_agent import QLearningAgent


# ====================================================================== #
#  Evaluation helper                                                      #
# ====================================================================== #
def evaluate_agent(
    env: TeleopEnv,
    agent: QLearningAgent,
    n_episodes: int = 10,
) -> dict[str, float]:
    """Run n_episodes with greedy policy and return averaged metrics."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0  # fully greedy

    rewards, tracking_rmses = [], []
    zero_action = int(np.argmin(np.abs(cfg.V_LEVELS)))

    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = env.discretise_obs(obs)
        ep_reward = 0.0
        done = False

        while not done:
            q_values = agent.q_values(state)
            max_q = q_values.max()
            best = np.flatnonzero(q_values == max_q)
            action = zero_action if zero_action in best else int(best[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = env.discretise_obs(obs)
            ep_reward += reward

        rewards.append(ep_reward)

        h = env.render()
        if h and h["pos_error"]:
            pe = np.array(h["pos_error"])
            tracking_rmses.append(np.sqrt(np.mean(pe ** 2)))

    agent.epsilon = saved_eps

    return {
        "mean_reward": float(np.mean(rewards)),
        "tracking_rmse_m": float(np.mean(tracking_rmses)) if tracking_rmses else 0.0,
    }


# ====================================================================== #
#  Main training loop                                                     #
# ====================================================================== #
def train() -> QLearningAgent:
    env = TeleopEnv()
    state_dims = env.get_state_dims()
    agent = QLearningAgent(state_dims, cfg.N_ACTIONS, seed=42)

    n_states = int(np.prod(state_dims))
    print("=" * 60)
    print("  Bilateral Teleoperation - Q-Learning (Direct Voltage)")
    print("=" * 60)
    print(f"  State dims :  {state_dims}  ->  {n_states:,} theoretical states")
    print(f"  Actions    :  {cfg.N_ACTIONS}")
    print(f"  Q-table    :  sparse (full size {n_states * cfg.N_ACTIONS:,} entries)")
    print(f"  Episodes   :  {cfg.NUM_EPISODES:,}")
    print(f"  RL freq    :  {1.0 / (cfg.SUB_STEPS * cfg.DT):.0f} Hz")
    print(f"  RL steps/ep:  {cfg.MAX_STEPS}")
    print("=" * 60)
    print()

    # Logging buffers
    episode_rewards = np.zeros(cfg.NUM_EPISODES, dtype=np.float64)
    eval_log_rewards = []
    eval_log_tracking = []

    t_start = time.time()

    for ep in trange(cfg.NUM_EPISODES, desc="Training"):
        obs, _ = env.reset()
        state = env.discretise_obs(obs)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = env.discretise_obs(obs)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward

        agent.decay_epsilon()
        episode_rewards[ep] = ep_reward

        # Periodic evaluation
        if (ep + 1) % cfg.EVAL_EVERY == 0:
            metrics = evaluate_agent(env, agent, cfg.EVAL_EPISODES)
            eval_log_rewards.append(metrics["mean_reward"])
            eval_log_tracking.append(metrics["tracking_rmse_m"])

            elapsed = time.time() - t_start
            print(
                f"\n  Ep {ep+1:>6} | eps {agent.epsilon:.4f} | "
                f"R {metrics['mean_reward']:+8.2f} | "
                f"TE {metrics['tracking_rmse_m']*1000:6.2f} mm | "
                f"{elapsed:.0f}s | {agent.coverage():.1%} coverage | "
                f"S {agent.discovered_states():,}"
            )

    # Save artefacts
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)

    agent.save(os.path.join(results_dir, "models", "q_table.npy"))
    np.savez(
        os.path.join(results_dir, "logs", "training_log.npz"),
        episode_rewards=episode_rewards,
        eval_rewards=np.array(eval_log_rewards),
        eval_tracking_rmse=np.array(eval_log_tracking),
    )

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Final epsilon  : {agent.epsilon:.5f}")
    print("  Q-table saved  : results/models/q_table.npy")
    print("  Log saved      : results/logs/training_log.npz")
    print(f"  {agent}")
    print("=" * 60)

    return agent


if __name__ == "__main__":
    train()
