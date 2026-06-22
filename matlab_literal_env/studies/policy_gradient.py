from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from tqdm.auto import tqdm

try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
except ImportError as exc:  # pragma: no cover - optional until runtime
    PPO = SAC = TD3 = None  # type: ignore[assignment]
    BaseCallback = object  # type: ignore[assignment]
    NormalActionNoise = None  # type: ignore[assignment]
    DummyVecEnv = SubprocVecEnv = VecMonitor = None  # type: ignore[assignment]
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
    plot_control_effect_dashboard,
    plot_error_diagnostics,
    plot_eval_signal_performance,
    plot_rollout_dashboard,
    plot_state_trajectory,
    rollout_metrics,
    save_history_npz,
    save_training_plot,
    write_run_summary,
)
from .dqn_state_variants import DQNStateVariant, get_dqn_state_variant, load_custom_dqn_state_variant
from .rewarding import ReplicaRewardEnv, RewardVariant, load_reward_variant_from_json, reward_variant_from_name


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
PG_TRAIN_RESET_OPTIONS_POOL_KEY = "_pg_train_reset_options_pool"

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


def load_reset_options_json(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with open(Path(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("signals", "reset_options", "scenarios"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise TypeError(f"Reset-options JSON must contain a list, got {type(payload).__name__}")
    options: list[dict[str, Any]] = []
    for row in payload:
        row = dict(row)
        if isinstance(row.get("reset_options"), dict):
            merged = dict(row["reset_options"])
            if "name" in row and "name" not in merged:
                merged["name"] = row["name"]
            row = merged
        options.append(row)
    return options


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
        self.train_reset_options_pool = tuple(
            dict(row)
            for row in (self.env_kwargs.pop(PG_TRAIN_RESET_OPTIONS_POOL_KEY, None) or [])
        )
        self._reset_options_rng = np.random.default_rng(0)
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
        self._temporal_lags = self._state_temporal_lags()
        self._temporal_base_obs_dim = self._state_temporal_base_obs_dim()
        self._temporal_obs_history: list[np.ndarray] = []
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

    def _state_temporal_lags(self) -> tuple[int, ...]:
        metadata = dict(self.state_variant.metadata or {})
        temporal = metadata.get("temporal_stack")
        if not isinstance(temporal, dict):
            return ()
        lags = tuple(sorted({int(lag) for lag in temporal.get("lags", [])}))
        return lags if lags and 0 in lags else ()

    def _state_temporal_base_obs_dim(self) -> int:
        if not self._temporal_lags:
            return 0
        metadata = dict(self.state_variant.metadata or {})
        base_dim = int(metadata.get("base_obs_dim", 0) or 0)
        expected_dim = base_dim * len(self._temporal_lags)
        if base_dim <= 0 or expected_dim != int(self.state_variant.obs_dim):
            raise ValueError(
                "Temporal state metadata is inconsistent: "
                f"base_obs_dim={base_dim}, lags={self._temporal_lags}, obs_dim={self.state_variant.obs_dim}."
            )
        return base_dim

    def _transform(self, obs: np.ndarray, info: dict[str, Any] | None) -> np.ndarray:
        transformed = self.state_variant.extractor(np.asarray(obs, dtype=np.float32), info or {})
        if self._temporal_lags:
            current = np.asarray(transformed, dtype=np.float32).reshape(-1)[: self._temporal_base_obs_dim]
            return self._stack_temporal_observation(current)
        return np.asarray(transformed, dtype=np.float32)

    def _reset_temporal_observation(self, current: np.ndarray) -> np.ndarray:
        max_lag = max(self._temporal_lags)
        current = np.asarray(current, dtype=np.float32).reshape(-1)
        self._temporal_obs_history = [current.copy() for _ in range(max_lag + 1)]
        return np.concatenate([self._temporal_obs_history[lag] for lag in self._temporal_lags]).astype(np.float32, copy=False)

    def _stack_temporal_observation(self, current: np.ndarray) -> np.ndarray:
        current = np.asarray(current, dtype=np.float32).reshape(-1)
        if not self._temporal_obs_history:
            return self._reset_temporal_observation(current)
        max_lag = max(self._temporal_lags)
        self._temporal_obs_history.insert(0, current.copy())
        del self._temporal_obs_history[max_lag + 1 :]
        while len(self._temporal_obs_history) <= max_lag:
            self._temporal_obs_history.append(self._temporal_obs_history[-1].copy())
        return np.concatenate([self._temporal_obs_history[lag] for lag in self._temporal_lags]).astype(np.float32, copy=False)

    def set_reset_options_seed(self, seed: int) -> None:
        self._reset_options_rng = np.random.default_rng(int(seed))

    def _sample_train_reset_options(self) -> dict[str, Any] | None:
        if not self.train_reset_options_pool:
            return None
        idx = int(self._reset_options_rng.integers(0, len(self.train_reset_options_pool)))
        return dict(self.train_reset_options_pool[idx])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._reset_options_rng = np.random.default_rng(int(seed))
        if options is None:
            options = self._sample_train_reset_options()
        obs, info = self.reward_env.reset(seed=seed, options=options)
        self._temporal_obs_history = []
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
        if self._temporal_lags:
            merged["state_temporal_lags"] = list(self._temporal_lags)
            merged["state_temporal_base_obs_dim"] = int(self._temporal_base_obs_dim)
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
        total_timesteps: int,
        eval_every_episodes: int,
        eval_episodes: int,
        eval_fn: Callable[[Any, int, int], tuple[dict[str, float], dict[str, Any]]],
        progress_label: str = "PPO training",
        progress_update_timesteps: int = 50,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.total_episodes = int(max(1, total_episodes))
        self.total_timesteps = int(max(1, total_timesteps))
        self.eval_every_episodes = int(max(1, eval_every_episodes))
        self.eval_episodes = int(max(1, eval_episodes))
        self.eval_fn = eval_fn
        self.progress_label = str(progress_label)
        self.progress_update_timesteps = int(max(1, progress_update_timesteps))
        self._progress_bar: tqdm | None = None
        self._last_progress_timestep = 0
        self._final_episode_eval_recorded = False

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

    def _on_training_start(self) -> None:
        self._progress_bar = tqdm(
            total=self.total_timesteps,
            desc=self.progress_label,
            unit="ts",
            miniters=self.progress_update_timesteps,
            dynamic_ncols=True,
        )
        self._last_progress_timestep = 0

    def _update_progress(self, *, force: bool = False) -> None:
        if self._progress_bar is None:
            return
        current = min(int(self.num_timesteps), self.total_timesteps)
        delta = current - self._last_progress_timestep
        if delta <= 0:
            return
        if force or delta >= self.progress_update_timesteps or current >= self.total_timesteps:
            self._progress_bar.update(delta)
            self._last_progress_timestep = current
            self._progress_bar.set_postfix(
                episodes=f"{self.completed_episodes}/{self.total_episodes}",
                refresh=False,
            )

    def _on_training_end(self) -> None:
        self._update_progress(force=True)
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None

    def _record_eval(self) -> None:
        eval_metrics, _ = self.eval_fn(self.model, self.eval_episodes, 10_000 + self.completed_episodes)
        self.eval_steps.append(self.completed_episodes)
        self.eval_mean_reward.append(float(eval_metrics["mean_reward"]))
        self.eval_tracking_rmse.append(float(eval_metrics["tracking_rmse_m"]))
        self.eval_transparency_rmse.append(float(eval_metrics["transparency_rmse_w"]))
        for key, value in eval_metrics.items():
            try:
                self.logger.record(f"teleop_eval/{key}", float(value))
            except (TypeError, ValueError):
                continue
        self.logger.record("teleop_eval/completed_episodes", float(self.completed_episodes))
        self.logger.dump(self.num_timesteps)

    def _on_step(self) -> bool:
        self._update_progress()
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
            for key in (
                "mean_reward",
                "tracking_rmse_m",
                "tracking_mae_m",
                "tracking_max_abs_m",
                "velocity_error_rmse_mps",
                "acceleration_error_rmse_mps2",
                "transparency_ratio_mean",
                "transparency_ratio_error_rmse",
                "mean_abs_u_v",
                "rms_u_v",
                "control_energy_v2_s",
                "mean_abs_delta_u_v",
                "rms_delta_u_v",
                "saturation_fraction",
                "invalid_episode",
            ):
                if key in metrics:
                    self.logger.record(f"teleop_train/{key}", float(metrics[key]))
            self.logger.record("teleop_train/completed_episodes", float(self.completed_episodes))
            self.logger.dump(self.num_timesteps)

            should_eval = (
                self.completed_episodes == 1
                or self.completed_episodes % self.eval_every_episodes == 0
            )
            if self.completed_episodes >= self.total_episodes and not self._final_episode_eval_recorded:
                should_eval = True
                self._final_episode_eval_recorded = True
            if should_eval:
                self._record_eval()
        return True


def evaluate_policy_gradient(
    model: Any,
    env_factory: Callable[[], PolicyGradientReplicaEnv],
    *,
    n_episodes: int,
    seed_offset: int,
    reset_options_schedule: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
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
    eval_episode: list[int] = []
    eval_signal_name: list[str] = []
    eval_signal_waveform: list[str] = []
    eval_signal_amp_n: list[float] = []
    eval_signal_bias_n: list[float] = []
    eval_signal_omega_rad_s: list[float] = []
    eval_signal_phase_rad: list[float] = []
    eval_episode_tracking_rmse_m: list[float] = []
    eval_episode_force_rmse_n: list[float] = []
    eval_episode_transparency_rmse_w: list[float] = []
    eval_episode_mean_reward: list[float] = []
    eval_episode_mean_abs_u_v: list[float] = []
    eval_episode_rms_u_v: list[float] = []
    eval_episode_mean_abs_delta_u_v: list[float] = []
    eval_episode_saturation_fraction: list[float] = []
    eval_episode_requested_applied_rmse_v: list[float] = []
    eval_episode_completed: list[float] = []
    eval_episode_terminated: list[float] = []
    eval_episode_truncated: list[float] = []
    eval_episode_steps: list[int] = []
    eval_episode_seconds: list[float] = []
    eval_episode_termination_reason: list[str] = []
    reset_options_schedule = list(reset_options_schedule or [])

    def _option_float(options: dict[str, Any], *keys: str) -> float:
        for key in keys:
            if key in options and options[key] is not None:
                try:
                    return float(options[key])
                except (TypeError, ValueError):
                    continue
        return float("nan")

    for ep in range(int(max(1, n_episodes))):
        env = env_factory()
        reset_options = (
            dict(reset_options_schedule[ep % len(reset_options_schedule)])
            if reset_options_schedule
            else None
        )
        obs, info = env.reset(seed=seed_offset + ep, options=reset_options)
        if hasattr(model, "reset_recurrent_state"):
            model.reset_recurrent_state()
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
        if reset_options is not None:
            history["signal_name"] = str(reset_options.get("name", f"signal_{ep + 1:03d}"))
        metrics = rollout_metrics(history, env_switch_time=float(getattr(env.base_env, "env_switch_time", cfg.ENV_SWITCH_TIME)))
        episode_metrics.append(metrics)
        episode_histories.append(history)
        steps = int(len(history.get("time", [])))
        seconds = float(steps * cfg.RL_DT)
        episode_steps.append(steps)

        termination_reason = str(final_info.get("termination_reason", ""))
        completed = termination_reason == "max_steps" or final_truncated
        if completed:
            completed_episodes += 1
        if termination_reason == "stroke_limit":
            stroke_limit_episodes += 1
        if termination_reason == "tracking_error_fail":
            tracking_error_fail_episodes += 1
        if termination_reason == "volume_singularity":
            volume_singularity_episodes += 1
        if final_terminated:
            terminated_episodes += 1
        if final_truncated:
            truncated_episodes += 1

        signal_options = dict(reset_options or {})
        eval_episode.append(ep + 1)
        eval_signal_name.append(str(signal_options.get("name", f"signal_{ep + 1:03d}")))
        eval_signal_waveform.append(str(signal_options.get("force_waveform", signal_options.get("fh_waveform", "signal"))))
        eval_signal_amp_n.append(_option_float(signal_options, "force_amp", "fh_amp", "force_amp_N"))
        eval_signal_bias_n.append(_option_float(signal_options, "force_bias", "fh_bias", "force_bias_N"))
        omega_rad_s = _option_float(signal_options, "force_freq_rad", "fh_freq_rad", "omega", "omega_rad_s")
        if not np.isfinite(omega_rad_s):
            freq_hz = _option_float(signal_options, "force_freq", "fh_freq", "freq_hz")
            if np.isfinite(freq_hz):
                omega_rad_s = float(2.0 * np.pi * freq_hz)
        eval_signal_omega_rad_s.append(omega_rad_s)
        eval_signal_phase_rad.append(_option_float(signal_options, "force_phase", "fh_phase", "force_phase_rad"))
        eval_episode_tracking_rmse_m.append(float(metrics.get("tracking_rmse_m", 0.0)))
        eval_episode_force_rmse_n.append(float(metrics.get("force_rmse_n", 0.0)))
        eval_episode_transparency_rmse_w.append(float(metrics.get("transparency_rmse_w", 0.0)))
        eval_episode_mean_reward.append(float(metrics.get("mean_reward", 0.0)))
        u_v = history_array(history, "u_v", dtype=np.float64)
        requested_u_v = history_array(history, "requested_u_v", dtype=np.float64)
        if u_v.size:
            action_limit = float(np.max(np.abs(_resolve_action_levels(getattr(env.base_env, "action_levels", None)))))
            eval_episode_mean_abs_u_v.append(float(np.mean(np.abs(u_v))))
            eval_episode_rms_u_v.append(float(np.sqrt(np.mean(u_v ** 2))))
            eval_episode_mean_abs_delta_u_v.append(float(np.mean(np.abs(np.diff(u_v)))) if u_v.size >= 2 else 0.0)
            eval_episode_saturation_fraction.append(float(np.mean(np.abs(u_v) >= 0.98 * action_limit)))
            if requested_u_v.size == u_v.size:
                eval_episode_requested_applied_rmse_v.append(float(np.sqrt(np.mean((requested_u_v - u_v) ** 2))))
            else:
                eval_episode_requested_applied_rmse_v.append(float("nan"))
        else:
            eval_episode_mean_abs_u_v.append(0.0)
            eval_episode_rms_u_v.append(0.0)
            eval_episode_mean_abs_delta_u_v.append(0.0)
            eval_episode_saturation_fraction.append(0.0)
            eval_episode_requested_applied_rmse_v.append(float("nan"))
        eval_episode_completed.append(float(completed))
        eval_episode_terminated.append(float(final_terminated))
        eval_episode_truncated.append(float(final_truncated))
        eval_episode_steps.append(steps)
        eval_episode_seconds.append(seconds)
        eval_episode_termination_reason.append(termination_reason)
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
    aggregate_history["eval_episode"] = eval_episode
    aggregate_history["eval_signal_name"] = eval_signal_name
    aggregate_history["eval_signal_waveform"] = eval_signal_waveform
    aggregate_history["eval_signal_amp_n"] = eval_signal_amp_n
    aggregate_history["eval_signal_bias_n"] = eval_signal_bias_n
    aggregate_history["eval_signal_omega_rad_s"] = eval_signal_omega_rad_s
    aggregate_history["eval_signal_phase_rad"] = eval_signal_phase_rad
    aggregate_history["eval_episode_tracking_rmse_m"] = eval_episode_tracking_rmse_m
    aggregate_history["eval_episode_force_rmse_n"] = eval_episode_force_rmse_n
    aggregate_history["eval_episode_transparency_rmse_w"] = eval_episode_transparency_rmse_w
    aggregate_history["eval_episode_mean_reward"] = eval_episode_mean_reward
    aggregate_history["eval_episode_mean_abs_u_v"] = eval_episode_mean_abs_u_v
    aggregate_history["eval_episode_rms_u_v"] = eval_episode_rms_u_v
    aggregate_history["eval_episode_mean_abs_delta_u_v"] = eval_episode_mean_abs_delta_u_v
    aggregate_history["eval_episode_saturation_fraction"] = eval_episode_saturation_fraction
    aggregate_history["eval_episode_requested_applied_rmse_v"] = eval_episode_requested_applied_rmse_v
    aggregate_history["eval_episode_completed"] = eval_episode_completed
    aggregate_history["eval_episode_terminated"] = eval_episode_terminated
    aggregate_history["eval_episode_truncated"] = eval_episode_truncated
    aggregate_history["eval_episode_steps"] = eval_episode_steps
    aggregate_history["eval_episode_seconds"] = eval_episode_seconds
    aggregate_history["eval_episode_termination_reason"] = eval_episode_termination_reason
    if reset_options_schedule:
        aggregate_history["eval_signal_count"] = int(len(reset_options_schedule))
        aggregate_history["eval_signal_names"] = [str(row.get("name", f"signal_{idx + 1:03d}")) for idx, row in enumerate(reset_options_schedule)]
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
    plot_error_diagnostics(history, plots_dir / "err.png", title, env_switch_time)
    plot_eval_signal_performance(history, plots_dir / "sigperf.png", title)
    plot_control_effect_dashboard(history, plots_dir / "ctrl.png", title, env_switch_time)
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
    vec_env_type: str = "auto",
) -> Any:
    require_sb3()
    parallel_envs = int(max(1, parallel_envs))
    resolved_vec_env_type = _resolve_vec_env_type(vec_env_type, parallel_envs)

    def _make_env(rank: int):
        def _thunk():
            env = env_factory()
            env.parallel_envs = int(parallel_envs)
            if hasattr(env, "set_reset_options_seed"):
                env.set_reset_options_seed(10_000 + int(rank))
            return env

        return _thunk

    env_fns = [_make_env(idx) for idx in range(parallel_envs)]
    actual_vec_env_type = resolved_vec_env_type
    try:
        env = SubprocVecEnv(env_fns) if resolved_vec_env_type == "subproc" else DummyVecEnv(env_fns)
    except Exception as exc:
        if resolved_vec_env_type != "subproc":
            raise
        print(
            "[policy_gradient] SubprocVecEnv failed; falling back to DummyVecEnv. "
            f"Reason: {type(exc).__name__}: {exc}",
            flush=True,
        )
        actual_vec_env_type = "dummy"
        env = DummyVecEnv(env_fns)
    monitored_env = VecMonitor(env)
    setattr(monitored_env, "teleop_resolved_vec_env_type", actual_vec_env_type)
    return monitored_env


def _resolve_vec_env_type(vec_env_type: str, parallel_envs: int) -> str:
    vec_env_type = str(vec_env_type).strip().lower()
    if vec_env_type not in {"auto", "dummy", "subproc"}:
        raise ValueError(f"Unknown vec_env_type: {vec_env_type}")
    if vec_env_type != "auto":
        return vec_env_type
    if os.name != "nt" and int(parallel_envs) > 1:
        return "subproc"
    return "dummy"


def _build_model(
    *,
    algo: str,
    env: Any,
    tensorboard_log: str,
    seed: int,
    total_timesteps: int,
    ppo_n_steps: int | None = None,
    ppo_batch_size: int | None = None,
    ppo_n_epochs: int | None = None,
    ppo_device: str = "cpu",
) -> Any:
    require_sb3()
    algo = str(algo)
    common_kwargs = {
        "env": env,
        "tensorboard_log": tensorboard_log,
        "seed": int(seed),
        "verbose": 0,
        "device": str(ppo_device),
    }

    if algo == PG_ALGO_PPO_CONTINUOUS:
        ppo_kwargs = _model_hyperparameter_summary(
            algo,
            ppo_n_steps=ppo_n_steps,
            ppo_batch_size=ppo_batch_size,
            ppo_n_epochs=ppo_n_epochs,
        )
        return PPO(
            "MlpPolicy",
            **ppo_kwargs,
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


def _model_hyperparameter_summary(
    algo: str,
    *,
    ppo_n_steps: int | None = None,
    ppo_batch_size: int | None = None,
    ppo_n_epochs: int | None = None,
) -> dict[str, Any]:
    algo = str(algo)
    if algo == PG_ALGO_PPO_CONTINUOUS:
        hparams = {
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 512,
            "n_epochs": 6,
            "gamma": 0.997,
            "gae_lambda": 0.95,
            "ent_coef": 0.001,
            "clip_range": 0.2,
            "target_kl": 0.04,
            "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        }
        if ppo_n_steps is not None:
            hparams["n_steps"] = int(max(1, ppo_n_steps))
        if ppo_batch_size is not None:
            hparams["batch_size"] = int(max(1, ppo_batch_size))
        if ppo_n_epochs is not None:
            hparams["n_epochs"] = int(max(1, ppo_n_epochs))
        return hparams
    if algo == PG_ALGO_PPO_DISCRETE:
        return {
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        }
    if algo == PG_ALGO_TD3:
        return {
            "learning_rate": 3e-4,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "policy_kwargs": {"net_arch": [256, 256]},
        }
    if algo == PG_ALGO_SAC:
        return {
            "learning_rate": 3e-4,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "policy_kwargs": {"net_arch": [256, 256]},
        }
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
    vec_env_type: str = "auto",
    ppo_n_steps: int | None = None,
    ppo_batch_size: int | None = None,
    ppo_n_epochs: int | None = None,
    ppo_device: str = "cpu",
    train_reset_options_pool: list[dict[str, Any]] | None = None,
    eval_reset_options_schedule: list[dict[str, Any]] | None = None,
) -> RunResult:
    require_sb3()

    algo = str(algo)
    dirs = mk_run_dirs(out_dir)
    eval_env_factory = build_policy_gradient_env_factory(
        algo=algo,
        env_mode=env_mode,
        env_kwargs=env_kwargs,
        reward_variant=reward_variant,
        state_variant=state_variant,
    )
    train_env_kwargs = dict(env_kwargs)
    if train_reset_options_pool:
        train_env_kwargs[PG_TRAIN_RESET_OPTIONS_POOL_KEY] = [dict(row) for row in train_reset_options_pool]
    train_env_factory = build_policy_gradient_env_factory(
        algo=algo,
        env_mode=env_mode,
        env_kwargs=train_env_kwargs,
        reward_variant=reward_variant,
        state_variant=state_variant,
    )
    total_timesteps = int(total_timesteps_from_episodes(env_kwargs, total_episodes) if total_timesteps is None else total_timesteps)
    parallel_envs = int(_default_parallel_envs(algo) if parallel_envs is None else max(1, parallel_envs))
    eval_every_episodes = int(_default_eval_every_episodes(algo) if eval_every_episodes is None else max(1, eval_every_episodes))
    resolved_vec_env_type = _resolve_vec_env_type(vec_env_type, parallel_envs)
    model_hyperparameters = _model_hyperparameter_summary(
        algo,
        ppo_n_steps=ppo_n_steps,
        ppo_batch_size=ppo_batch_size,
        ppo_n_epochs=ppo_n_epochs,
    )
    probe_env = eval_env_factory()
    action_levels = probe_env.action_levels.tolist()
    probe_env.close()

    train_env = _build_vec_env(
        train_env_factory,
        parallel_envs=parallel_envs,
        vec_env_type=vec_env_type,
    )
    actual_vec_env_type = str(getattr(train_env, "teleop_resolved_vec_env_type", resolved_vec_env_type))
    model = _build_model(
        algo=algo,
        env=train_env,
        tensorboard_log=dirs["tensorboard"],
        seed=seed,
        total_timesteps=total_timesteps,
        ppo_n_steps=ppo_n_steps,
        ppo_batch_size=ppo_batch_size,
        ppo_n_epochs=ppo_n_epochs,
        ppo_device=ppo_device,
    )

    callback = PolicyGradientMetricsCallback(
        total_episodes=total_episodes,
        total_timesteps=total_timesteps,
        eval_every_episodes=eval_every_episodes,
        eval_episodes=max(1, min(cfg.DQN_EVAL_EPISODES, test_episodes)),
        eval_fn=lambda mdl, eval_eps, seed_offset: evaluate_policy_gradient(
            mdl,
            eval_env_factory,
            n_episodes=eval_eps,
            seed_offset=seed_offset,
            reset_options_schedule=eval_reset_options_schedule,
        ),
        progress_label=label,
        progress_update_timesteps=50,
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
        eval_env_factory,
        n_episodes=test_episodes,
        seed_offset=20_000,
        reset_options_schedule=eval_reset_options_schedule,
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
            "actual_train_timesteps": int(getattr(model, "num_timesteps", total_timesteps)),
            "test_episodes": int(test_episodes),
            "evaluation_history_mode": "mean_over_test_episodes",
            "obs_dim": int(state_variant.obs_dim),
            "state_features": list(state_variant.feature_names),
            "state_variant_description": str(state_variant.description),
            "state_spec": state_variant.metadata,
            "reward_config": asdict(reward_variant),
            "model_hyperparameters": model_hyperparameters,
            "ppo_n_steps": int(model_hyperparameters.get("n_steps", 0) or 0),
            "ppo_batch_size": int(model_hyperparameters.get("batch_size", 0) or 0),
            "ppo_n_epochs": int(model_hyperparameters.get("n_epochs", 0) or 0),
            "ppo_learning_rate": float(model_hyperparameters.get("learning_rate", 0.0) or 0.0),
            "ppo_gamma": float(model_hyperparameters.get("gamma", 0.0) or 0.0),
            "ppo_gae_lambda": float(model_hyperparameters.get("gae_lambda", 0.0) or 0.0),
            "ppo_ent_coef": float(model_hyperparameters.get("ent_coef", 0.0) or 0.0),
            "ppo_clip_range": float(model_hyperparameters.get("clip_range", 0.0) or 0.0),
            "ppo_target_kl": float(model_hyperparameters.get("target_kl", 0.0) or 0.0),
            "model_device": str(ppo_device),
            "episode_duration": float(env_kwargs["episode_duration"]),
            "env_switch_time": float(env_kwargs["env_switch_time"]),
            "terminate_on_error": bool(env_kwargs["terminate_on_error"]),
            "enforce_stroke_limit": bool(env_kwargs.get("enforce_stroke_limit", True)),
            "stroke_limit_mode": str(env_kwargs.get("stroke_limit_mode", "terminate")),
            "reset_options": dict(env_kwargs.get("reset_options", {})),
            "train_signal_count": int(len(train_reset_options_pool or [])),
            "train_reset_options_pool": list(train_reset_options_pool or []),
            "eval_signal_count": int(len(eval_reset_options_schedule or [])),
            "eval_reset_options_schedule": list(eval_reset_options_schedule or []),
            "parallel_envs": int(parallel_envs),
            "eval_every_episodes": int(eval_every_episodes),
            "vec_env_type": str(vec_env_type),
            "resolved_vec_env_type": actual_vec_env_type,
            "requested_resolved_vec_env_type": resolved_vec_env_type,
            "action_space_type": "discrete" if algo == PG_ALGO_PPO_DISCRETE else "continuous",
            "action_levels": action_levels,
            "eval_metrics": eval_metrics,
            "tracking_mae_m": float(eval_metrics.get("tracking_mae_m", 0.0)),
            "tracking_max_abs_m": float(eval_metrics.get("tracking_max_abs_m", 0.0)),
            "velocity_error_rmse_mps": float(eval_metrics.get("velocity_error_rmse_mps", 0.0)),
            "acceleration_error_rmse_mps2": float(eval_metrics.get("acceleration_error_rmse_mps2", 0.0)),
            "transparency_ratio_median": float(eval_metrics.get("transparency_ratio_median", 0.0)),
            "transparency_ratio_error_rmse": float(eval_metrics.get("transparency_ratio_error_rmse", 0.0)),
            "transparency_ratio_valid_fraction": float(eval_metrics.get("transparency_ratio_valid_fraction", 0.0)),
            "transparency_ratio_within_20pct": float(eval_metrics.get("transparency_ratio_within_20pct", 0.0)),
            "mean_abs_u_v": float(eval_metrics.get("mean_abs_u_v", 0.0)),
            "rms_u_v": float(eval_metrics.get("rms_u_v", 0.0)),
            "control_energy_v2_s": float(eval_metrics.get("control_energy_v2_s", 0.0)),
            "max_abs_u_v": float(eval_metrics.get("max_abs_u_v", 0.0)),
            "saturation_fraction": float(eval_metrics.get("saturation_fraction", 0.0)),
            "mean_abs_delta_u_v": float(eval_metrics.get("mean_abs_delta_u_v", 0.0)),
            "rms_delta_u_v": float(eval_metrics.get("rms_delta_u_v", 0.0)),
            "max_abs_delta_u_v": float(eval_metrics.get("max_abs_delta_u_v", 0.0)),
            "mean_abs_delta2_u_v": float(eval_metrics.get("mean_abs_delta2_u_v", 0.0)),
            "rms_delta2_u_v": float(eval_metrics.get("rms_delta2_u_v", 0.0)),
            "max_abs_delta2_u_v": float(eval_metrics.get("max_abs_delta2_u_v", 0.0)),
        },
    )
    return result


def get_policy_gradient_state_variant(name: str, spec_json: str | Path | None = None) -> DQNStateVariant:
    if spec_json:
        return load_custom_dqn_state_variant(spec_json)
    return get_dqn_state_variant(name)


def get_policy_gradient_reward_variant(name: str, spec_json: str | Path | None = None) -> RewardVariant:
    if spec_json:
        return load_reward_variant_from_json(spec_json)
    return reward_variant_from_name(name)
