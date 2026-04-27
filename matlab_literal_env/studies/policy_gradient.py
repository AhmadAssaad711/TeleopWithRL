from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
except ImportError as exc:  # pragma: no cover - optional until runtime
    PPO = SAC = TD3 = None  # type: ignore[assignment]
    BaseCallback = object  # type: ignore[assignment]
    NormalActionNoise = None  # type: ignore[assignment]
    DummyVecEnv = VecMonitor = None  # type: ignore[assignment]
    _SB3_IMPORT_ERROR = exc
else:  # pragma: no cover - import side effect only
    _SB3_IMPORT_ERROR = None

from ... import config as cfg
from ..simuoriginal_env import SimuOriginalReplicaEnv
from .common import (
    RunResult,
    aggregate_episode_histories,
    history_array,
    history_with_obs,
    mk_run_dirs,
    plot_action_usage,
    plot_average_core_rollout,
    plot_rollout_dashboard,
    plot_state_trajectory,
    rollout_metrics,
    save_history_npz,
    save_training_plot,
    write_run_summary,
)
from .dqn_state_variants import DQNStateVariant, get_dqn_state_variant
from .rewarding import ReplicaRewardEnv, RewardVariant, reward_variant_from_name


PG_ALGO_PPO_CONTINUOUS = "ppo_continuous"
PG_ALGO_TD3 = "td3"
PG_ALGO_SAC = "sac"
PG_ALGO_PPO_DISCRETE = "ppo_discrete"
PG_ALGO_CHOICES = (
    PG_ALGO_PPO_CONTINUOUS,
    PG_ALGO_TD3,
    PG_ALGO_SAC,
    PG_ALGO_PPO_DISCRETE,
)

_CONTINUOUS_ACTION_ALGOS = {
    PG_ALGO_PPO_CONTINUOUS,
    PG_ALGO_TD3,
    PG_ALGO_SAC,
}


def require_sb3() -> None:
    if _SB3_IMPORT_ERROR is not None:
        raise ImportError(
            "Policy-gradient experiments require 'stable-baselines3'. "
            "Install it with 'pip install stable-baselines3'."
        ) from _SB3_IMPORT_ERROR


def algo_output_dir_name(algo: str) -> str:
    algo = str(algo)
    if algo == PG_ALGO_PPO_CONTINUOUS:
        return "ppo"
    if algo == PG_ALGO_TD3:
        return "td3"
    if algo == PG_ALGO_SAC:
        return "sac"
    if algo == PG_ALGO_PPO_DISCRETE:
        return "ppod"
    raise KeyError(f"Unknown policy-gradient algo: {algo}")


def algo_notebook_tag(algo: str) -> str:
    algo = str(algo)
    if algo == PG_ALGO_PPO_CONTINUOUS:
        return "ppo"
    if algo == PG_ALGO_TD3:
        return "td3"
    if algo == PG_ALGO_SAC:
        return "sac"
    if algo == PG_ALGO_PPO_DISCRETE:
        return "ppod"
    raise KeyError(f"Unknown policy-gradient algo: {algo}")


def algo_display_name(algo: str) -> str:
    algo = str(algo)
    if algo == PG_ALGO_PPO_CONTINUOUS:
        return "PPO Continuous"
    if algo == PG_ALGO_TD3:
        return "TD3"
    if algo == PG_ALGO_SAC:
        return "SAC"
    if algo == PG_ALGO_PPO_DISCRETE:
        return "PPO Discrete"
    raise KeyError(f"Unknown policy-gradient algo: {algo}")


def _resolve_action_levels(action_levels: list[float] | tuple[float, ...] | np.ndarray | None) -> np.ndarray:
    levels = np.asarray(cfg.V_LEVELS if action_levels is None else action_levels, dtype=np.float64).reshape(-1)
    if levels.size == 0:
        raise ValueError("action_levels must contain at least one level")
    return levels.astype(np.float64, copy=True)


class PolicyGradientReplicaEnv(gym.Env):
    """SB3-friendly wrapper around the SimuOriginal replica env."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        *,
        env_mode: str,
        env_kwargs: dict[str, Any],
        reward_variant: RewardVariant,
        state_variant: DQNStateVariant,
        algo: str,
    ):
        super().__init__()
        self.env_mode = str(env_mode)
        self.env_kwargs = dict(env_kwargs)
        self.reward_variant = reward_variant
        self.state_variant = state_variant
        self.algo = str(algo)
        self.base_env = SimuOriginalReplicaEnv(
            env_mode=self.env_mode,
            master_input_mode=cfg.MASTER_INPUT_FORCE,
            **dict(self.env_kwargs),
        )
        self.reward_env = ReplicaRewardEnv(self.base_env, self.reward_variant)
        self.parallel_envs = 1
        self.obs_dim = int(self.state_variant.obs_dim)
        self.action_levels = _resolve_action_levels(getattr(self.base_env, "action_levels", None))
        self.is_discrete = self.algo == PG_ALGO_PPO_DISCRETE

        if self.is_discrete:
            self.action_space = spaces.Discrete(int(self.action_levels.size))
        else:
            self.action_space = self.reward_env.action_space

        low = np.full(self.obs_dim, -np.inf, dtype=np.float32)
        high = np.full(self.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.last_obs = np.zeros(self.obs_dim, dtype=np.float32)

    def _transform(self, obs: np.ndarray, info: dict[str, Any] | None) -> np.ndarray:
        transformed = self.state_variant.extractor(np.asarray(obs, dtype=np.float32), info or {})
        return np.asarray(transformed, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.reward_env.reset(seed=seed, options=options)
        self.last_obs = self._transform(obs, info)
        return self.last_obs.copy(), dict(info)

    def step(self, action):
        mapped_action = int(action) if self.is_discrete else action
        obs, reward, terminated, truncated, info = self.reward_env.step(mapped_action)
        self.last_obs = self._transform(obs, info)
        info = dict(info)
        if terminated or truncated:
            history = self.render() or {}
            info["episode_metrics"] = rollout_metrics(
                history,
                env_switch_time=float(getattr(self.base_env, "env_switch_time", cfg.ENV_SWITCH_TIME)),
            )
        return self.last_obs.copy(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        history = self.reward_env.render() or {}
        merged = dict(history)
        merged["state_variant_name"] = self.state_variant.name
        merged["state_variant_features"] = list(self.state_variant.feature_names)
        merged["algo"] = self.algo
        return merged

    def close(self):
        if hasattr(self.base_env, "close"):
            self.base_env.close()


def build_policy_gradient_env_factory(
    *,
    algo: str,
    env_mode: str,
    env_kwargs: dict[str, Any],
    reward_variant: RewardVariant,
    state_variant: DQNStateVariant,
) -> Callable[[], PolicyGradientReplicaEnv]:
    def _factory() -> PolicyGradientReplicaEnv:
        return PolicyGradientReplicaEnv(
            algo=algo,
            env_mode=env_mode,
            env_kwargs=dict(env_kwargs),
            reward_variant=reward_variant,
            state_variant=state_variant,
        )

    return _factory


def _episode_steps(env_kwargs: dict[str, Any]) -> int:
    duration = float(env_kwargs.get("episode_duration", cfg.EPISODE_DURATION))
    return max(1, int(round(duration / float(cfg.RL_DT))))


def total_timesteps_from_episodes(env_kwargs: dict[str, Any], total_episodes: int) -> int:
    return int(max(1, total_episodes) * _episode_steps(env_kwargs))


class PolicyGradientMetricsCallback(BaseCallback):
    def __init__(
        self,
        *,
        total_episodes: int,
        eval_every_episodes: int,
        eval_episodes: int,
        eval_fn: Callable[[Any, int, int], tuple[dict[str, float], dict[str, Any]]],
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.total_episodes = int(max(1, total_episodes))
        self.eval_every_episodes = int(max(1, eval_every_episodes))
        self.eval_episodes = int(max(1, eval_episodes))
        self.eval_fn = eval_fn

        self.completed_episodes = 0
        self.episode_returns: list[float] = []
        self.episode_tracking_rmse: list[float] = []
        self.episode_transparency_rmse: list[float] = []
        self.episode_pre_tracking_rmse: list[float] = []
        self.episode_post_tracking_rmse: list[float] = []
        self.episode_pre_transparency_rmse: list[float] = []
        self.episode_post_transparency_rmse: list[float] = []
        self.episode_invalid: list[float] = []
        self.eval_steps: list[int] = []
        self.eval_mean_reward: list[float] = []
        self.eval_tracking_rmse: list[float] = []
        self.eval_transparency_rmse: list[float] = []

    def _record_eval(self) -> None:
        eval_metrics, _ = self.eval_fn(self.model, self.eval_episodes, 10_000 + self.completed_episodes)
        self.eval_steps.append(self.completed_episodes)
        self.eval_mean_reward.append(float(eval_metrics["mean_reward"]))
        self.eval_tracking_rmse.append(float(eval_metrics["tracking_rmse_m"]))
        self.eval_transparency_rmse.append(float(eval_metrics["transparency_rmse_w"]))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if infos is None or dones is None:
            return True

        for done, info in zip(list(dones), list(infos)):
            if not bool(done):
                continue
            metrics = dict(info.get("episode_metrics") or {})
            if not metrics:
                continue
            self.completed_episodes += 1
            self.episode_returns.append(float(metrics.get("mean_reward", 0.0)))
            self.episode_tracking_rmse.append(float(metrics.get("tracking_rmse_m", 0.0)))
            self.episode_transparency_rmse.append(float(metrics.get("transparency_rmse_w", 0.0)))
            self.episode_pre_tracking_rmse.append(float(metrics.get("pre_switch_tracking_rmse_m", 0.0)))
            self.episode_post_tracking_rmse.append(float(metrics.get("post_switch_tracking_rmse_m", 0.0)))
            self.episode_pre_transparency_rmse.append(float(metrics.get("pre_switch_transparency_rmse_w", 0.0)))
            self.episode_post_transparency_rmse.append(float(metrics.get("post_switch_transparency_rmse_w", 0.0)))
            self.episode_invalid.append(float(metrics.get("invalid_episode", 0.0)))

            if (
                self.completed_episodes == 1
                or self.completed_episodes % self.eval_every_episodes == 0
                or self.completed_episodes >= self.total_episodes
            ):
                self._record_eval()
        return True


def evaluate_policy_gradient(
    model: Any,
    env_factory: Callable[[], PolicyGradientReplicaEnv],
    *,
    n_episodes: int,
    seed_offset: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    episode_metrics: list[dict[str, float]] = []
    episode_histories: list[dict[str, Any]] = []
    completed_episodes = 0
    terminated_episodes = 0
    truncated_episodes = 0
    stroke_limit_episodes = 0
    tracking_error_fail_episodes = 0
    volume_singularity_episodes = 0
    episode_steps: list[int] = []

    for ep in range(int(max(1, n_episodes))):
        env = env_factory()
        obs, info = env.reset(seed=seed_offset + ep)
        done = False
        obs_trace: list[np.ndarray] = []
        final_info = dict(info)
        final_terminated = False
        final_truncated = False

        while not done:
            obs_trace.append(np.asarray(obs, dtype=np.float32).copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            final_info = dict(info)
            final_terminated = bool(terminated)
            final_truncated = bool(truncated)

        history = history_with_obs(env.render() or {}, obs_trace)
        metrics = rollout_metrics(history, env_switch_time=float(getattr(env.base_env, "env_switch_time", cfg.ENV_SWITCH_TIME)))
        episode_metrics.append(metrics)
        episode_histories.append(history)
        episode_steps.append(int(len(history.get("time", []))))

        if str(final_info.get("termination_reason")) == "max_steps" or final_truncated:
            completed_episodes += 1
        if str(final_info.get("termination_reason")) == "stroke_limit":
            stroke_limit_episodes += 1
        if str(final_info.get("termination_reason")) == "tracking_error_fail":
            tracking_error_fail_episodes += 1
        if str(final_info.get("termination_reason")) == "volume_singularity":
            volume_singularity_episodes += 1
        if final_terminated:
            terminated_episodes += 1
        if final_truncated:
            truncated_episodes += 1
        env.close()

    aggregate = {
        key: float(np.mean([row[key] for row in episode_metrics])) for key in episode_metrics[0].keys()
    }
    aggregate_history = aggregate_episode_histories(episode_histories)
    aggregate.update(
        {
            "evaluation_episodes": float(n_episodes),
            "completed_episodes": float(completed_episodes),
            "completed_episode_rate": float(completed_episodes / max(1, n_episodes)),
            "terminated_episodes": float(terminated_episodes),
            "truncated_episodes": float(truncated_episodes),
            "stroke_limit_episodes": float(stroke_limit_episodes),
            "tracking_error_fail_episodes": float(tracking_error_fail_episodes),
            "volume_singularity_episodes": float(volume_singularity_episodes),
            "mean_episode_steps": float(np.mean(episode_steps)) if episode_steps else 0.0,
            "mean_episode_seconds": float(np.mean(episode_steps) * cfg.RL_DT) if episode_steps else 0.0,
        }
    )
    aggregate_history["evaluation_episodes"] = int(n_episodes)
    aggregate_history["completed_episodes"] = int(completed_episodes)
    aggregate_history["completed_episode_rate"] = float(completed_episodes / max(1, n_episodes))
    aggregate_history["terminated_episodes"] = int(terminated_episodes)
    aggregate_history["truncated_episodes"] = int(truncated_episodes)
    aggregate_history["stroke_limit_episodes"] = int(stroke_limit_episodes)
    aggregate_history["tracking_error_fail_episodes"] = int(tracking_error_fail_episodes)
    aggregate_history["volume_singularity_episodes"] = int(volume_singularity_episodes)
    aggregate_history["mean_episode_steps"] = float(np.mean(episode_steps)) if episode_steps else 0.0
    aggregate_history["mean_episode_seconds"] = float(np.mean(episode_steps) * cfg.RL_DT) if episode_steps else 0.0
    return aggregate, aggregate_history


def _continuous_action_usage(history: dict[str, Any], out_path: str | Path, title: str) -> None:
    actions = history_array(history, "u_v_all" if "u_v_all" in history else "u_v", dtype=np.float64)
    labels = history_array(history, "env_label_all" if "env_label_all" in history else "env_label", dtype=object)
    if actions.size == 0:
        return

    bins = np.linspace(float(np.min(cfg.V_LEVELS)), float(np.max(cfg.V_LEVELS)), 21, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 5))
    for env_name, color in (("skin", "tab:blue"), ("fat", "tab:orange")):
        mask = labels == env_name
        if not np.any(mask):
            continue
        ax.hist(
            actions[mask],
            bins=bins,
            density=True,
            alpha=0.45,
            color=color,
            label=env_name,
        )
    ax.set_xlabel("Voltage action [V]")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}: continuous action usage by environment")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_policy_gradient_visuals(
    history: dict[str, Any],
    plots_dir: str | Path,
    title: str,
    env_switch_time: float,
    *,
    action_mode: str,
    action_levels: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> None:
    plots_dir = Path(plots_dir)
    plot_average_core_rollout(history, plots_dir / "avg_roll.png", title, env_switch_time)
    plot_rollout_dashboard(history, plots_dir / "roll.png", title, env_switch_time)
    if action_mode == "discrete":
        plot_action_usage(history, plots_dir / "act.png", title, action_levels=action_levels)
    else:
        _continuous_action_usage(history, plots_dir / "act.png", title)
    plot_state_trajectory(history, plots_dir / "traj.png", title)


def _predict_discrete_probs(model: Any, obs: np.ndarray) -> np.ndarray:
    tensor_obs, _ = model.policy.obs_to_tensor(obs.reshape(1, -1))
    dist = model.policy.get_distribution(tensor_obs)
    probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
    return np.asarray(probs, dtype=np.float64)


def plot_policy_gradient_slices(
    model: Any,
    history: dict[str, Any],
    state_variant: DQNStateVariant,
    out_path: str | Path,
    *,
    algo: str,
    action_levels: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> None:
    obs = history_array(history, "obs", dtype=np.float32)
    labels = history_array(history, "env_label", dtype=object)
    if obs.size == 0 or getattr(state_variant, "obs_dim", 0) < 2:
        return
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)

    x_idx, y_idx = 0, 1
    x_name = state_variant.feature_names[x_idx]
    y_name = state_variant.feature_names[y_idx]
    discrete = str(algo) == PG_ALGO_PPO_DISCRETE
    levels = _resolve_action_levels(action_levels)

    def _axis_grid(values: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(values, [5.0, 95.0])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = -1.0, 1.0
        if abs(hi - lo) < 1e-6:
            center = float(values[-1]) if values.size else 0.0
            lo, hi = center - 1.0, center + 1.0
        return np.linspace(lo, hi, 7, dtype=np.float64)

    x_centers = _axis_grid(np.asarray(obs[:, x_idx], dtype=np.float64))
    y_centers = _axis_grid(np.asarray(obs[:, y_idx], dtype=np.float64))
    templates: list[tuple[str, np.ndarray]] = []
    for label in ("skin", "fat"):
        mask = labels == label
        if np.any(mask):
            templates.append((label, np.median(obs[mask], axis=0).astype(np.float32)))
    if not templates:
        templates.append(("overall", np.median(obs, axis=0).astype(np.float32)))
    elif "overall" not in {name for name, _ in templates}:
        templates.append(("overall", np.median(obs, axis=0).astype(np.float32)))

    fig, axes = plt.subplots(len(templates), 3, figsize=(16, 4.8 * len(templates)))
    axes = np.atleast_2d(axes)
    for row_idx, (label, context_obs) in enumerate(templates):
        context_action, _ = model.predict(context_obs, deterministic=True)
        context_action_value = (
            float(levels[int(context_action)]) if discrete else float(np.asarray(context_action).reshape(-1)[0])
        )
        action_map = np.zeros((y_centers.size, x_centers.size), dtype=np.float64)
        aux1_map = np.zeros_like(action_map)
        aux2_map = np.zeros_like(action_map)

        for x_plot_idx, x_value in enumerate(x_centers):
            for y_plot_idx, y_value in enumerate(y_centers):
                variant_obs = np.array(context_obs, dtype=np.float32, copy=True)
                variant_obs[x_idx] = float(x_value)
                variant_obs[y_idx] = float(y_value)
                action, _ = model.predict(variant_obs, deterministic=True)

                if discrete:
                    probs = _predict_discrete_probs(model, variant_obs)
                    action_idx = int(np.asarray(action).reshape(-1)[0])
                    action_value = float(levels[action_idx])
                    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0))))
                    action_map[y_plot_idx, x_plot_idx] = action_value
                    aux1_map[y_plot_idx, x_plot_idx] = float(np.max(probs))
                    aux2_map[y_plot_idx, x_plot_idx] = entropy
                else:
                    action_value = float(np.asarray(action).reshape(-1)[0])
                    action_map[y_plot_idx, x_plot_idx] = action_value
                    aux1_map[y_plot_idx, x_plot_idx] = abs(action_value)
                    aux2_map[y_plot_idx, x_plot_idx] = action_value - context_action_value

        extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]
        im0 = axes[row_idx, 0].imshow(action_map, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
        axes[row_idx, 0].set_title(f"Action slice | {label}")
        plt.colorbar(im0, ax=axes[row_idx, 0], label="Voltage [V]")

        if discrete:
            im1 = axes[row_idx, 1].imshow(aux1_map, origin="lower", aspect="auto", extent=extent, cmap="viridis")
            axes[row_idx, 1].set_title(f"Max-prob slice | {label}")
            plt.colorbar(im1, ax=axes[row_idx, 1], label="max prob")

            im2 = axes[row_idx, 2].imshow(aux2_map, origin="lower", aspect="auto", extent=extent, cmap="plasma")
            axes[row_idx, 2].set_title(f"Policy entropy | {label}")
            plt.colorbar(im2, ax=axes[row_idx, 2], label="entropy")
        else:
            im1 = axes[row_idx, 1].imshow(aux1_map, origin="lower", aspect="auto", extent=extent, cmap="viridis")
            axes[row_idx, 1].set_title(f"|Action| slice | {label}")
            plt.colorbar(im1, ax=axes[row_idx, 1], label="|u_v|")

            im2 = axes[row_idx, 2].imshow(aux2_map, origin="lower", aspect="auto", extent=extent, cmap="plasma")
            axes[row_idx, 2].set_title(f"Action delta from context | {label}")
            plt.colorbar(im2, ax=axes[row_idx, 2], label="delta u_v")

        for col in range(3):
            axes[row_idx, col].set_xlabel(f"{x_name} [scaled]")
            axes[row_idx, col].set_ylabel(f"{y_name} [scaled]")
            axes[row_idx, col].grid(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _default_parallel_envs(algo: str) -> int:
    return 8 if str(algo) in {PG_ALGO_PPO_CONTINUOUS, PG_ALGO_PPO_DISCRETE} else 1


def _default_eval_every_episodes(algo: str) -> int:
    del algo
    return 100


def _build_vec_env(
    env_factory: Callable[[], PolicyGradientReplicaEnv],
    *,
    parallel_envs: int,
) -> Any:
    require_sb3()

    def _make_env(rank: int):
        def _thunk():
            env = env_factory()
            env.parallel_envs = int(parallel_envs)
            return env

        return _thunk

    env = DummyVecEnv([_make_env(idx) for idx in range(int(max(1, parallel_envs)))])
    return VecMonitor(env)


def _build_model(
    *,
    algo: str,
    env: Any,
    tensorboard_log: str,
    seed: int,
    total_timesteps: int,
) -> Any:
    require_sb3()
    algo = str(algo)
    common_kwargs = {
        "env": env,
        "tensorboard_log": tensorboard_log,
        "seed": int(seed),
        "verbose": 0,
        "device": "cpu",
    }

    if algo == PG_ALGO_PPO_CONTINUOUS:
        return PPO(
            "MlpPolicy",
            learning_rate=3e-4,
            n_steps=256,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            clip_range=0.2,
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
            **common_kwargs,
        )
    if algo == PG_ALGO_PPO_DISCRETE:
        return PPO(
            "MlpPolicy",
            learning_rate=3e-4,
            n_steps=256,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
            **common_kwargs,
        )
    if algo == PG_ALGO_TD3:
        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[-1], dtype=np.float64),
            sigma=0.20 * np.ones(env.action_space.shape[-1], dtype=np.float64),
        )
        return TD3(
            "MlpPolicy",
            learning_rate=3e-4,
            buffer_size=min(max(100_000, total_timesteps), 500_000),
            learning_starts=min(10_000, max(1_000, total_timesteps // 20)),
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            policy_kwargs={"net_arch": [256, 256]},
            **common_kwargs,
        )
    if algo == PG_ALGO_SAC:
        return SAC(
            "MlpPolicy",
            learning_rate=3e-4,
            buffer_size=min(max(100_000, total_timesteps), 500_000),
            learning_starts=min(10_000, max(1_000, total_timesteps // 20)),
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs={"net_arch": [256, 256]},
            **common_kwargs,
        )
    raise KeyError(f"Unknown policy-gradient algo: {algo}")


def train_policy_gradient_variant(
    *,
    algo: str,
    out_dir: str | Path,
    env_mode: str,
    env_kwargs: dict,
    state_variant: DQNStateVariant,
    reward_variant: RewardVariant,
    total_episodes: int,
    test_episodes: int,
    seed: int,
    label: str,
    total_timesteps: int | None = None,
    parallel_envs: int | None = None,
    eval_every_episodes: int | None = None,
) -> RunResult:
    require_sb3()

    algo = str(algo)
    dirs = mk_run_dirs(out_dir)
    env_factory = build_policy_gradient_env_factory(
        algo=algo,
        env_mode=env_mode,
        env_kwargs=env_kwargs,
        reward_variant=reward_variant,
        state_variant=state_variant,
    )
    total_timesteps = int(total_timesteps_from_episodes(env_kwargs, total_episodes) if total_timesteps is None else total_timesteps)
    parallel_envs = int(_default_parallel_envs(algo) if parallel_envs is None else max(1, parallel_envs))
    eval_every_episodes = int(_default_eval_every_episodes(algo) if eval_every_episodes is None else max(1, eval_every_episodes))
    probe_env = env_factory()
    action_levels = probe_env.action_levels.tolist()
    probe_env.close()

    train_env = _build_vec_env(env_factory, parallel_envs=parallel_envs)
    model = _build_model(
        algo=algo,
        env=train_env,
        tensorboard_log=dirs["tensorboard"],
        seed=seed,
        total_timesteps=total_timesteps,
    )

    callback = PolicyGradientMetricsCallback(
        total_episodes=total_episodes,
        eval_every_episodes=eval_every_episodes,
        eval_episodes=max(1, min(cfg.DQN_EVAL_EPISODES, test_episodes)),
        eval_fn=lambda mdl, eval_eps, seed_offset: evaluate_policy_gradient(
            mdl,
            env_factory,
            n_episodes=eval_eps,
            seed_offset=seed_offset,
        ),
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    train_env.close()

    episode_returns = np.asarray(callback.episode_returns, dtype=np.float64)
    episode_tracking = np.asarray(callback.episode_tracking_rmse, dtype=np.float64)
    episode_transparency = np.asarray(callback.episode_transparency_rmse, dtype=np.float64)
    episode_pre_tracking = np.asarray(callback.episode_pre_tracking_rmse, dtype=np.float64)
    episode_post_tracking = np.asarray(callback.episode_post_tracking_rmse, dtype=np.float64)
    episode_pre_transparency = np.asarray(callback.episode_pre_transparency_rmse, dtype=np.float64)
    episode_post_transparency = np.asarray(callback.episode_post_transparency_rmse, dtype=np.float64)
    episode_invalid = np.asarray(callback.episode_invalid, dtype=np.float64)
    eval_steps = np.asarray(callback.eval_steps, dtype=np.int64)
    eval_mean_reward = np.asarray(callback.eval_mean_reward, dtype=np.float64)
    eval_tracking = np.asarray(callback.eval_tracking_rmse, dtype=np.float64)
    eval_transparency = np.asarray(callback.eval_transparency_rmse, dtype=np.float64)

    model_path = Path(dirs["models"]) / f"{algo_output_dir_name(algo)}_model"
    model.save(str(model_path))
    np.savez(
        Path(dirs["logs"]) / "train.npz",
        episode_returns=episode_returns,
        episode_tracking_rmse=episode_tracking,
        episode_transparency_rmse=episode_transparency,
        episode_pre_switch_tracking_rmse=episode_pre_tracking,
        episode_post_switch_tracking_rmse=episode_post_tracking,
        episode_pre_switch_transparency_rmse=episode_pre_transparency,
        episode_post_switch_transparency_rmse=episode_post_transparency,
        episode_invalid=episode_invalid,
        eval_steps=eval_steps,
        eval_mean_reward=eval_mean_reward,
        eval_tracking_rmse=eval_tracking,
        eval_transparency_rmse=eval_transparency,
    )

    save_training_plot(
        episode_returns,
        episode_tracking,
        episode_transparency,
        Path(dirs["plots"]) / "train.png",
        label,
        losses=None,
        eval_payload={
            "steps": eval_steps,
            "mean_reward": eval_mean_reward,
            "tracking_rmse_m": eval_tracking,
            "transparency_rmse_w": eval_transparency,
        },
    )

    eval_metrics, history = evaluate_policy_gradient(
        model,
        env_factory,
        n_episodes=test_episodes,
        seed_offset=20_000,
    )
    save_history_npz(history, Path(dirs["episodes"]) / "test.npz")
    save_policy_gradient_visuals(
        history,
        dirs["plots"],
        label,
        env_switch_time=float(env_kwargs["env_switch_time"]),
        action_mode="discrete" if algo == PG_ALGO_PPO_DISCRETE else "continuous",
        action_levels=action_levels,
    )
    plot_policy_gradient_slices(
        model,
        history,
        state_variant,
        Path(dirs["plots"]) / "slices.png",
        algo=algo,
        action_levels=action_levels,
    )

    result = RunResult(
        label=label,
        family=algo,
        mean_reward=eval_metrics["mean_reward"],
        tracking_rmse_m=eval_metrics["tracking_rmse_m"],
        transparency_rmse_w=eval_metrics["transparency_rmse_w"],
        pre_switch_tracking_rmse_m=eval_metrics["pre_switch_tracking_rmse_m"],
        post_switch_tracking_rmse_m=eval_metrics["post_switch_tracking_rmse_m"],
        pre_switch_transparency_rmse_w=eval_metrics["pre_switch_transparency_rmse_w"],
        post_switch_transparency_rmse_w=eval_metrics["post_switch_transparency_rmse_w"],
        invalid_episode_rate=eval_metrics["invalid_episode"],
        history=history,
        out_dir=dirs["base"],
        tensorboard_dir=dirs["tensorboard"],
        model_path=str(model_path) + ".zip",
        reward_variant=reward_variant.name,
        state_variant=state_variant.name,
    )
    write_run_summary(
        dirs,
        result,
        extra={
            "algo": algo,
            "algo_display_name": algo_display_name(algo),
            "env_mode": env_mode,
            "master_input_mode": cfg.MASTER_INPUT_FORCE,
            "total_episodes": int(total_episodes),
            "total_timesteps": int(total_timesteps),
            "test_episodes": int(test_episodes),
            "evaluation_history_mode": "mean_over_test_episodes",
            "obs_dim": int(state_variant.obs_dim),
            "state_features": list(state_variant.feature_names),
            "episode_duration": float(env_kwargs["episode_duration"]),
            "env_switch_time": float(env_kwargs["env_switch_time"]),
            "terminate_on_error": bool(env_kwargs["terminate_on_error"]),
            "enforce_stroke_limit": bool(env_kwargs.get("enforce_stroke_limit", True)),
            "stroke_limit_mode": str(env_kwargs.get("stroke_limit_mode", "terminate")),
            "reset_options": dict(env_kwargs.get("reset_options", {})),
            "parallel_envs": int(parallel_envs),
            "eval_every_episodes": int(eval_every_episodes),
            "action_space_type": "discrete" if algo == PG_ALGO_PPO_DISCRETE else "continuous",
            "action_levels": action_levels,
        },
    )
    return result


def get_policy_gradient_state_variant(name: str) -> DQNStateVariant:
    return get_dqn_state_variant(name)


def get_policy_gradient_reward_variant(name: str) -> RewardVariant:
    return reward_variant_from_name(name)
