"""Train/evaluate RL agents and MRAC, with organized result folders."""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from adaptive_q_learning_agent import AdaptiveQLearningAgent
from mrac_controller import FilteredMRACController
from q_learning_agent import QLearningAgent
from teleop_env import TeleopEnv

try:
    from tqdm import trange
except ImportError:
    def trange(n: int, desc: str | None = None):
        return range(n)


@dataclass
class EvalResult:
    mean_reward: float
    tracking_rmse_m: float
    transparency_rmse: float
    history: dict


def _mk_agent_dirs(agent_name: str) -> dict[str, str]:
    base = os.path.join(os.path.dirname(__file__), cfg.RESULTS_ROOT_DIR, agent_name)
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


def _mk_compare_dir() -> str:
    p = os.path.join(os.path.dirname(__file__), cfg.RESULTS_ROOT_DIR, cfg.COMPARE_RESULTS_DIR)
    os.makedirs(p, exist_ok=True)
    return p


def _greedy_action(q_values: np.ndarray, zero_action: int) -> int:
    max_q = q_values.max()
    best = np.flatnonzero(q_values == max_q)
    return zero_action if zero_action in best else int(best[0])


def _evaluate_rl(
    agent: QLearningAgent,
    env_mode: str,
    contextual: bool,
    n_episodes: int,
) -> EvalResult:
    rewards, trk_rmse, trn_rmse = [], [], []
    rep_history = None

    zero_action = int(np.argmin(np.abs(cfg.V_LEVELS)))

    for ep in range(n_episodes):
        env = TeleopEnv(env_mode=env_mode)
        obs, info = env.reset(seed=ep)
        base_state = env.discretise_obs(obs)
        state = (
            AdaptiveQLearningAgent.build_state(base_state, info["env_id"])
            if contextual
            else base_state
        )

        done = False
        ep_reward = 0.0

        while not done:
            q_values = agent.q_values(state)
            action = _greedy_action(q_values, zero_action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            base_state = env.discretise_obs(obs)
            state = (
                AdaptiveQLearningAgent.build_state(base_state, info["env_id"])
                if contextual
                else base_state
            )
            ep_reward += reward

        h = env.render()
        if rep_history is None:
            rep_history = h
        rewards.append(ep_reward)
        pe = np.array(h["pos_error"], dtype=np.float64)
        te = np.array(h["transparency_error"], dtype=np.float64)
        trk_rmse.append(float(np.sqrt(np.mean(pe ** 2))))
        trn_rmse.append(float(np.sqrt(np.mean(te ** 2))))

    return EvalResult(
        mean_reward=float(np.mean(rewards)),
        tracking_rmse_m=float(np.mean(trk_rmse)),
        transparency_rmse=float(np.mean(trn_rmse)),
        history=rep_history if rep_history is not None else {},
    )


def _evaluate_mrac(env_mode: str, n_episodes: int) -> EvalResult:
    rewards, trk_rmse, trn_rmse = [], [], []
    rep_history = None

    for ep in range(n_episodes):
        env = TeleopEnv(env_mode=env_mode)
        ctrl = FilteredMRACController()
        ctrl.reset()

        obs, info = env.reset(seed=ep)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = ctrl.step_action(
                pos_error=float(obs[0]),
                u_c=float(info["x_m"]),
                action_table=cfg.V_LEVELS,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        h = env.render()
        if rep_history is None:
            rep_history = h
        rewards.append(ep_reward)
        pe = np.array(h["pos_error"], dtype=np.float64)
        te = np.array(h["transparency_error"], dtype=np.float64)
        trk_rmse.append(float(np.sqrt(np.mean(pe ** 2))))
        trn_rmse.append(float(np.sqrt(np.mean(te ** 2))))

    return EvalResult(
        mean_reward=float(np.mean(rewards)),
        tracking_rmse_m=float(np.mean(trk_rmse)),
        transparency_rmse=float(np.mean(trn_rmse)),
        history=rep_history if rep_history is not None else {},
    )


def _save_training_plot(episode_rewards: np.ndarray, eval_rewards: np.ndarray, eval_track: np.ndarray,
                        eval_transparency: np.ndarray, out_path: str, title: str) -> None:
    x_eval = np.arange(1, len(eval_rewards) + 1) * cfg.EVAL_EVERY
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0, 0].plot(episode_rewards, color="tab:blue", alpha=0.35, lw=0.8)
    axes[0, 0].set_title("Episode Return")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_eval, eval_rewards, "o-", color="tab:green", lw=1.5)
    axes[0, 1].set_title("Eval Mean Return")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Return")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x_eval, eval_track * 1000.0, "o-", color="tab:red", lw=1.5)
    axes[1, 0].set_title("Eval Tracking RMSE")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("mm")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x_eval, eval_transparency, "o-", color="tab:purple", lw=1.5)
    axes[1, 1].set_title("Eval Transparency RMSE")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("W")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_head_to_head(mrac: EvalResult, rl: EvalResult, title: str, out_path: str) -> None:
    t_m = np.array(mrac.history["time"], dtype=np.float64)
    t_r = np.array(rl.history["time"], dtype=np.float64)

    pe_m = np.array(mrac.history["pos_error"], dtype=np.float64) * 1000.0
    pe_r = np.array(rl.history["pos_error"], dtype=np.float64) * 1000.0
    tr_m = np.array(mrac.history["transparency_error"], dtype=np.float64)
    tr_r = np.array(rl.history["transparency_error"], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(t_m, pe_m, label=f"MRAC ({np.sqrt(np.mean(pe_m**2)):.2f} mm RMSE)", color="tab:orange", lw=1.6)
    axes[0].plot(t_r, pe_r, label=f"RL ({np.sqrt(np.mean(pe_r**2)):.2f} mm RMSE)", color="tab:blue", lw=1.4)
    axes[0].axhline(0.0, color="gray", lw=0.6)
    axes[0].set_ylabel("Tracking Error [mm]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t_m, tr_m, label=f"MRAC ({np.sqrt(np.mean(tr_m**2)):.2f} W RMSE)", color="tab:orange", lw=1.6)
    axes[1].plot(t_r, tr_r, label=f"RL ({np.sqrt(np.mean(tr_r**2)):.2f} W RMSE)", color="tab:blue", lw=1.4)
    axes[1].axhline(0.0, color="gray", lw=0.6)
    axes[1].set_ylabel("Transparency Error [W]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _train_rl(env_mode: str, contextual: bool, out_name: str) -> tuple[QLearningAgent, EvalResult]:
    dirs = _mk_agent_dirs(out_name)
    log_path = os.path.join(dirs["logs"], "training_output.log")

    env = TeleopEnv(env_mode=env_mode)
    base_dims = env.get_state_dims()
    state_dims = base_dims + (cfg.N_ENV_CONTEXTS,) if contextual else base_dims
    agent_cls = AdaptiveQLearningAgent if contextual else QLearningAgent
    agent = agent_cls(state_dims, cfg.N_ACTIONS, seed=42)

    ep_rewards = np.zeros(cfg.NUM_EPISODES, dtype=np.float64)
    eval_rewards, eval_track, eval_transparency = [], [], []

    with open(log_path, "w", encoding="utf-8") as logf:
        def _log(msg: str) -> None:
            print(msg)
            logf.write(msg + "\n")
            logf.flush()

        _log("=" * 72)
        _log(f"Training {out_name} | env_mode={env_mode} | contextual={contextual}")
        _log(f"state_dims={state_dims} actions={cfg.N_ACTIONS} episodes={cfg.NUM_EPISODES}")
        _log("=" * 72)

        for ep in trange(cfg.NUM_EPISODES, desc=f"Train-{out_name}"):
            obs, info = env.reset()
            base_state = env.discretise_obs(obs)
            state = (
                AdaptiveQLearningAgent.build_state(base_state, info["env_id"])
                if contextual
                else base_state
            )

            done = False
            ep_reward = 0.0

            while not done:
                action = agent.select_action(state)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_base = env.discretise_obs(obs)
                next_state = (
                    AdaptiveQLearningAgent.build_state(next_base, info["env_id"])
                    if contextual
                    else next_base
                )
                agent.update(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward

            agent.decay_epsilon()
            ep_rewards[ep] = ep_reward

            if (ep + 1) % cfg.EVAL_EVERY == 0:
                ev = _evaluate_rl(agent, env_mode=env_mode, contextual=contextual, n_episodes=cfg.EVAL_EPISODES)
                eval_rewards.append(ev.mean_reward)
                eval_track.append(ev.tracking_rmse_m)
                eval_transparency.append(ev.transparency_rmse)
                _log(
                    f"Ep {ep+1:>6} | eps {agent.epsilon:.4f} | R {ev.mean_reward:+9.2f} | "
                    f"TE {ev.tracking_rmse_m*1000:6.2f} mm | TrE {ev.transparency_rmse:6.3f} W | "
                    f"action coverage {agent.coverage():.1%}"
                )

    agent.save(os.path.join(dirs["models"], "q_table.npy"))
    np.savez(
        os.path.join(dirs["logs"], "training_log.npz"),
        episode_rewards=ep_rewards,
        eval_rewards=np.array(eval_rewards, dtype=np.float64),
        eval_tracking_rmse=np.array(eval_track, dtype=np.float64),
        eval_transparency_rmse=np.array(eval_transparency, dtype=np.float64),
    )
    _save_training_plot(
        ep_rewards,
        np.array(eval_rewards, dtype=np.float64),
        np.array(eval_track, dtype=np.float64),
        np.array(eval_transparency, dtype=np.float64),
        os.path.join(dirs["plots"], "training_progress.png"),
        title=f"{out_name} training curves",
    )

    final_eval = _evaluate_rl(agent, env_mode=env_mode, contextual=contextual, n_episodes=cfg.EVAL_EPISODES)
    np.savez(
        os.path.join(dirs["episodes"], "representative_eval_episode.npz"),
        **{k: np.array(v, dtype=object) for k, v in final_eval.history.items()},
    )
    return agent, final_eval


def run_full_benchmark() -> None:
    # 1) RL constant environment
    rl_const_agent, rl_const_eval = _train_rl(
        env_mode=cfg.ENV_MODE_CONSTANT,
        contextual=False,
        out_name=cfg.RL_CONSTANT_DIR,
    )

    # 2) RL changing environment (adaptive via environment-context state)
    rl_chg_agent, rl_chg_eval = _train_rl(
        env_mode=cfg.ENV_MODE_CHANGING,
        contextual=True,
        out_name=cfg.RL_CHANGING_DIR,
    )

    # 3) MRAC evaluations
    mrac_dirs = _mk_agent_dirs(cfg.MRAC_RESULTS_DIR)
    mrac_const = _evaluate_mrac(env_mode=cfg.ENV_MODE_CONSTANT, n_episodes=cfg.EVAL_EPISODES)
    mrac_chg = _evaluate_mrac(env_mode=cfg.ENV_MODE_CHANGING, n_episodes=cfg.EVAL_EPISODES)

    np.savez(
        os.path.join(mrac_dirs["episodes"], "constant_eval_episode.npz"),
        **{k: np.array(v, dtype=object) for k, v in mrac_const.history.items()},
    )
    np.savez(
        os.path.join(mrac_dirs["episodes"], "changing_eval_episode.npz"),
        **{k: np.array(v, dtype=object) for k, v in mrac_chg.history.items()},
    )
    with open(os.path.join(mrac_dirs["logs"], "mrac_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(
            "MRAC constant: "
            f"R={mrac_const.mean_reward:.3f}, "
            f"TE={mrac_const.tracking_rmse_m*1000:.3f} mm, "
            f"TrE={mrac_const.transparency_rmse:.5f} W\n"
        )
        f.write(
            "MRAC changing: "
            f"R={mrac_chg.mean_reward:.3f}, "
            f"TE={mrac_chg.tracking_rmse_m*1000:.3f} mm, "
            f"TrE={mrac_chg.transparency_rmse:.5f} W\n"
        )

    # 4) Head-to-head comparisons
    cmp_dir = _mk_compare_dir()
    _save_head_to_head(
        mrac=mrac_const,
        rl=rl_const_eval,
        title="Head-to-Head: MRAC vs RL (Constant Environment)",
        out_path=os.path.join(cmp_dir, "mrac_vs_rl_constant.png"),
    )
    _save_head_to_head(
        mrac=mrac_chg,
        rl=rl_chg_eval,
        title="Head-to-Head: MRAC vs RL (Changing Environment)",
        out_path=os.path.join(cmp_dir, "mrac_vs_rl_changing.png"),
    )

    summary_path = os.path.join(cmp_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("case,agent,mean_reward,tracking_rmse_mm,transparency_rmse_w\n")
        f.write(
            f"constant,MRAC,{mrac_const.mean_reward:.6f},"
            f"{mrac_const.tracking_rmse_m*1000:.6f},{mrac_const.transparency_rmse:.6f}\n"
        )
        f.write(
            f"constant,RL,{rl_const_eval.mean_reward:.6f},"
            f"{rl_const_eval.tracking_rmse_m*1000:.6f},{rl_const_eval.transparency_rmse:.6f}\n"
        )
        f.write(
            f"changing,MRAC,{mrac_chg.mean_reward:.6f},"
            f"{mrac_chg.tracking_rmse_m*1000:.6f},{mrac_chg.transparency_rmse:.6f}\n"
        )
        f.write(
            f"changing,RL,{rl_chg_eval.mean_reward:.6f},"
            f"{rl_chg_eval.tracking_rmse_m*1000:.6f},{rl_chg_eval.transparency_rmse:.6f}\n"
        )

    print("Benchmark complete.")
    print(f"RL constant results  : {os.path.join(cfg.RESULTS_ROOT_DIR, cfg.RL_CONSTANT_DIR)}")
    print(f"RL changing results  : {os.path.join(cfg.RESULTS_ROOT_DIR, cfg.RL_CHANGING_DIR)}")
    print(f"MRAC results         : {os.path.join(cfg.RESULTS_ROOT_DIR, cfg.MRAC_RESULTS_DIR)}")
    print(f"Comparison outputs   : {os.path.join(cfg.RESULTS_ROOT_DIR, cfg.COMPARE_RESULTS_DIR)}")


if __name__ == "__main__":
    run_full_benchmark()
