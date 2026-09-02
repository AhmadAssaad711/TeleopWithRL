"""CLI study for PPO policies with an auxiliary GRU dynamics head."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is available in the study env
    tqdm = None

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL.matlab_env_python_replica.config import config as cfg
    from TeleopWithRL.matlab_env_python_replica.policy_gradient.paths import suite_root as policy_gradient_suite_root
    from TeleopWithRL.matlab_env_python_replica.policy_gradient.scripts.run_physics_reward_ablation_basic_obs import (
        BASIC_OBS_FEATURES,
        build_ablations,
        build_reward_spec,
        build_state_spec,
        calibrate_scale_catalog,
        write_csv,
    )
    from TeleopWithRL.matlab_env_python_replica.common.cli import replica_env_kwargs_from_args
    from TeleopWithRL.matlab_env_python_replica.environment.simuoriginal_replica import FE_MODE_DYNAMICS
    from TeleopWithRL.matlab_env_python_replica.common.study_utils import (
        RunResult,
        history_array,
        mk_run_dirs,
        moving_avg,
        rollout_metrics,
        save_history_npz,
        save_json,
        save_training_plot,
        write_run_summary,
    )
    from TeleopWithRL.matlab_env_python_replica.dqn.state_variants import build_custom_dqn_state_variant_from_spec
    from TeleopWithRL.matlab_env_python_replica.common.focused_evaluation import (
        build_focused_scenarios,
        compute_non_bode_metrics,
        evaluate_policy_on_scenario,
        plot_scenario_detail_results,
        plot_scenario_result,
        save_scenario_history_npz,
    )
    from TeleopWithRL.matlab_env_python_replica.policy_gradient.training import (
        PG_ALGO_PPO_CONTINUOUS,
        PG_TRAIN_RESET_OPTIONS_POOL_KEY,
        build_policy_gradient_env_factory,
        evaluate_policy_gradient,
        load_reset_options_json,
        save_policy_gradient_visuals,
    )
    from TeleopWithRL.matlab_env_python_replica.common.rewarding import reward_variant_from_spec
else:
    try:
        from ...config import config as cfg
    except ImportError:
        from TeleopWithRL.matlab_env_python_replica.config import config as cfg
    from ..paths import suite_root as policy_gradient_suite_root
    from .run_physics_reward_ablation_basic_obs import (
        BASIC_OBS_FEATURES,
        build_ablations,
        build_reward_spec,
        build_state_spec,
        calibrate_scale_catalog,
        write_csv,
    )
    from ...common.cli import replica_env_kwargs_from_args
    from ...environment.simuoriginal_replica import FE_MODE_DYNAMICS
    from ...common.study_utils import (
        RunResult,
        history_array,
        mk_run_dirs,
        moving_avg,
        rollout_metrics,
        save_history_npz,
        save_json,
        save_training_plot,
        write_run_summary,
    )
    from ...dqn.state_variants import build_custom_dqn_state_variant_from_spec
    from ...common.focused_evaluation import (
        build_focused_scenarios,
        compute_non_bode_metrics,
        evaluate_policy_on_scenario,
        plot_scenario_detail_results,
        plot_scenario_result,
        save_scenario_history_npz,
    )
    from ..training import (
        PG_ALGO_PPO_CONTINUOUS,
        PG_TRAIN_RESET_OPTIONS_POOL_KEY,
        build_policy_gradient_env_factory,
        evaluate_policy_gradient,
        load_reset_options_json,
        save_policy_gradient_visuals,
    )
    from ...common.rewarding import reward_variant_from_spec


AUX_ALGO = "gru_ppo_aux"
PREDICTION_TARGET_LABELS = ("delta_x_m", "delta_x_s")
HIDDEN_TARGET_LABELS = ("P_s1", "P_s2", "P_m1", "P_m2", "mdot_L1", "mdot_L2")
VARIANT_DIR_NAMES = {
    "G0_gru_ppo": "G0",
    "G1_gru_prediction": "G1p",
    "G2_gru_hidden_state": "G2h",
    "G3_gru_prediction_hidden": "G3ph",
}


@dataclass(frozen=True)
class AuxiliaryVariant:
    key: str
    label: str
    prediction_weight: float
    hidden_state_weight: float
    note: str


@dataclass
class EpisodeBatch:
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    prediction_targets: np.ndarray
    hidden_targets: np.ndarray
    metrics: dict[str, float]
    history: dict[str, Any]
    steps: int
    terminated: bool
    truncated: bool
    termination_reason: str


class GruAuxActorCritic(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        encoder_dim: int,
        hidden_dim: int,
        action_scale: float,
        init_log_std: float,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.encoder_dim = int(encoder_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_scale = float(action_scale)
        self.init_log_std = float(init_log_std)
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.encoder_dim),
            nn.Tanh(),
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRU(self.encoder_dim, self.hidden_dim, batch_first=True)
        self.actor_mean = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, len(PREDICTION_TARGET_LABELS)),
        )
        self.hidden_state_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, len(HIDDEN_TARGET_LABELS)),
        )
        self.log_std = nn.Parameter(torch.full((self.action_dim,), self.init_log_std))

    def distribution_and_value(
        self,
        obs_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[Normal, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs_seq)
        gru_out, next_hidden = self.gru(encoded, hidden)
        mean = self.action_scale * torch.tanh(self.actor_mean(gru_out))
        std = torch.exp(self.log_std).clamp(1e-4, self.action_scale).view(1, 1, -1).expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(gru_out).squeeze(-1)
        return dist, value, gru_out, next_hidden

    def predict_delta(self, gru_out: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        scaled_action = (actions.detach() / max(self.action_scale, 1e-9)).clamp(-2.0, 2.0)
        return self.prediction_head(torch.cat([gru_out, scaled_action], dim=-1))

    def predict_hidden_state(self, gru_out: torch.Tensor) -> torch.Tensor:
        return self.hidden_state_head(gru_out)


class GruAuxPolicy:
    def __init__(
        self,
        model: GruAuxActorCritic,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
    ):
        self.model = model
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(1, 1, -1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(1, 1, -1)
        self.device = device
        self.hidden: torch.Tensor | None = None

    def reset_recurrent_state(self) -> None:
        self.hidden = None

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        self.model.eval()
        obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, 1, -1), device=self.device)
        low = torch.as_tensor(self.action_low, device=self.device)
        high = torch.as_tensor(self.action_high, device=self.device)
        with torch.no_grad():
            dist, _, _, next_hidden = self.model.distribution_and_value(obs_tensor, self.hidden)
            action = dist.mean if deterministic else dist.sample()
            action = torch.max(torch.min(action, high), low)
        self.hidden = next_hidden.detach()
        return action.detach().cpu().numpy().reshape(-1), None


def build_auxiliary_variants() -> tuple[AuxiliaryVariant, ...]:
    return (
        AuxiliaryVariant(
            "G0_gru_ppo",
            "GRU-PPO",
            0.0,
            0.0,
            "Recurrent PPO baseline with the same reward and basic observation space.",
        ),
        AuxiliaryVariant(
            "G1_gru_prediction",
            "GRU + prediction",
            0.10,
            0.0,
            "Adds next-position-delta prediction from recurrent state and detached action.",
        ),
        AuxiliaryVariant(
            "G2_gru_hidden_state",
            "GRU + hidden state",
            0.0,
            0.05,
            "Adds privileged pneumatic pressure/flow reconstruction during training only.",
        ),
        AuxiliaryVariant(
            "G3_gru_prediction_hidden",
            "GRU + both aux heads",
            0.10,
            0.05,
            "Combines next-position-delta prediction with hidden pneumatic-state reconstruction.",
        ),
    )


def _plot_save_path(path: str | Path) -> str | Path:
    path = Path(path)
    if os.name != "nt":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    resolved = path.resolve()
    parent = str(path.parent.resolve())
    if parent.startswith("\\\\?\\"):
        parent_text = parent
    elif parent.startswith("\\\\"):
        parent_text = "\\\\?\\UNC\\" + parent.lstrip("\\")
    else:
        parent_text = "\\\\?\\" + parent
    os.makedirs(parent_text, exist_ok=True)
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text.lstrip("\\")
    return "\\\\?\\" + text


def _long_path(path: str | Path) -> str | Path:
    path = Path(path)
    if os.name != "nt":
        return path
    resolved = path.resolve()
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text.lstrip("\\")
    return "\\\\?\\" + text


def _file_exists(path: str | Path) -> bool:
    try:
        with open(_long_path(path), "rb"):
            return True
    except FileNotFoundError:
        return False


def _variant_dir_name(variant: AuxiliaryVariant) -> str:
    return VARIANT_DIR_NAMES.get(variant.key, variant.key)


def _resolve_device(device: str) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _configure_torch_runtime() -> None:
    try:
        num_threads = int(os.environ.get("TELEOP_TORCH_NUM_THREADS", "1"))
    except ValueError:
        num_threads = 1
    num_threads = max(1, int(num_threads))
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(1, int(os.environ.get("TELEOP_TORCH_INTEROP_THREADS", "1"))))
    except (RuntimeError, ValueError):
        pass


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _hidden_target_from_env(env: Any) -> np.ndarray:
    base = getattr(env, "base_env", None)
    state = np.asarray(getattr(base, "state", np.zeros(12, dtype=np.float64)), dtype=np.float64)
    pressure_scale = max(float(cfg.OBS_SCALE_PRESSURE), 1e-9)
    flow_scale = max(float(cfg.OBS_SCALE_FLOW), 1e-9)
    return np.asarray(
        [
            state[int(getattr(base, "IX_PS1", 6))] / pressure_scale,
            state[int(getattr(base, "IX_PS2", 7))] / pressure_scale,
            state[int(getattr(base, "IX_PM1", 4))] / pressure_scale,
            state[int(getattr(base, "IX_PM2", 5))] / pressure_scale,
            state[int(getattr(base, "IX_ML1", 8))] / flow_scale,
            state[int(getattr(base, "IX_ML2", 9))] / flow_scale,
        ],
        dtype=np.float32,
    )


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for step in reversed(range(rewards.size)):
        next_nonterminal = 1.0 - float(dones[step])
        next_value = 0.0 if step == rewards.size - 1 else float(values[step + 1])
        delta = float(rewards[step]) + float(gamma) * next_value * next_nonterminal - float(values[step])
        last_gae = delta + float(gamma) * float(gae_lambda) * next_nonterminal * last_gae
        advantages[step] = float(last_gae)
    returns = advantages + values.astype(np.float32, copy=False)
    return advantages, returns.astype(np.float32, copy=False)


def collect_episode(
    env: Any,
    model: GruAuxActorCritic,
    *,
    device: torch.device,
    action_low: np.ndarray,
    action_high: np.ndarray,
    seed: int,
) -> EpisodeBatch:
    obs, info = env.reset(seed=int(seed))
    hidden: torch.Tensor | None = None
    done = False
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    log_prob_rows: list[float] = []
    value_rows: list[float] = []
    reward_rows: list[float] = []
    done_rows: list[float] = []
    prediction_targets: list[np.ndarray] = []
    hidden_targets: list[np.ndarray] = []
    final_info = dict(info)
    final_terminated = False
    final_truncated = False
    low = torch.as_tensor(np.asarray(action_low, dtype=np.float32).reshape(1, 1, -1), device=device)
    high = torch.as_tensor(np.asarray(action_high, dtype=np.float32).reshape(1, 1, -1), device=device)

    model.eval()
    while not done:
        current_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        current_hidden_target = _hidden_target_from_env(env)
        obs_tensor = torch.as_tensor(current_obs.reshape(1, 1, -1), device=device)
        with torch.no_grad():
            dist, value, _, next_hidden = model.distribution_and_value(obs_tensor, hidden)
            action = torch.max(torch.min(dist.sample(), high), low)
            log_prob = dist.log_prob(action).sum(dim=-1)
        action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        next_obs_array = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        obs_rows.append(current_obs)
        action_rows.append(action_np)
        log_prob_rows.append(float(log_prob.detach().cpu().numpy().reshape(-1)[0]))
        value_rows.append(float(value.detach().cpu().numpy().reshape(-1)[0]))
        reward_rows.append(float(reward))
        done_rows.append(float(bool(terminated or truncated)))
        prediction_targets.append((next_obs_array[:2] - current_obs[:2]).astype(np.float32, copy=False))
        hidden_targets.append(current_hidden_target)

        hidden = next_hidden.detach()
        obs = next_obs
        done = bool(terminated or truncated)
        final_info = dict(info)
        final_terminated = bool(terminated)
        final_truncated = bool(truncated)

    history = env.render() or {}
    metrics = dict(final_info.get("episode_metrics") or {})
    if not metrics:
        env_switch_time = float(getattr(getattr(env, "base_env", None), "env_switch_time", cfg.ENV_SWITCH_TIME))
        metrics = rollout_metrics(history, env_switch_time=env_switch_time)
    return EpisodeBatch(
        obs=np.asarray(obs_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.float32),
        log_probs=np.asarray(log_prob_rows, dtype=np.float32),
        values=np.asarray(value_rows, dtype=np.float32),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        dones=np.asarray(done_rows, dtype=np.float32),
        prediction_targets=np.asarray(prediction_targets, dtype=np.float32),
        hidden_targets=np.asarray(hidden_targets, dtype=np.float32),
        metrics=metrics,
        history=history,
        steps=int(len(reward_rows)),
        terminated=final_terminated,
        truncated=final_truncated,
        termination_reason=str(final_info.get("termination_reason", "")),
    )


def update_gru_ppo(
    model: GruAuxActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: EpisodeBatch,
    *,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
    prediction_weight: float,
    hidden_state_weight: float,
    epochs: int,
    max_grad_norm: float,
) -> dict[str, float]:
    advantages, returns = _compute_gae(
        batch.rewards,
        batch.values,
        batch.dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    advantages = (advantages - float(np.mean(advantages))) / (float(np.std(advantages)) + 1e-8)

    obs = torch.as_tensor(batch.obs.reshape(1, batch.obs.shape[0], batch.obs.shape[1]), device=device)
    actions = torch.as_tensor(batch.actions.reshape(1, batch.actions.shape[0], batch.actions.shape[1]), device=device)
    old_log_probs = torch.as_tensor(batch.log_probs.reshape(1, -1), device=device)
    returns_tensor = torch.as_tensor(returns.reshape(1, -1), device=device)
    advantages_tensor = torch.as_tensor(advantages.reshape(1, -1), device=device)
    prediction_targets = torch.as_tensor(batch.prediction_targets.reshape(1, batch.prediction_targets.shape[0], -1), device=device)
    hidden_targets = torch.as_tensor(batch.hidden_targets.reshape(1, batch.hidden_targets.shape[0], -1), device=device)

    stats: dict[str, list[float]] = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "prediction_loss": [],
        "hidden_state_loss": [],
        "approx_kl": [],
    }
    model.train()
    for _ in range(int(max(1, epochs))):
        dist, values, gru_out, _ = model.distribution_and_value(obs, None)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range))
        policy_loss = -torch.mean(torch.minimum(ratio * advantages_tensor, clipped_ratio * advantages_tensor))
        value_loss = torch.mean((returns_tensor - values) ** 2)
        prediction = model.predict_delta(gru_out, actions)
        hidden_prediction = model.predict_hidden_state(gru_out)
        prediction_loss = torch.mean((prediction - prediction_targets) ** 2)
        hidden_state_loss = torch.mean((hidden_prediction - hidden_targets) ** 2)
        loss = (
            policy_loss
            + (float(value_coef) * value_loss)
            - (float(entropy_coef) * entropy)
            + (float(prediction_weight) * prediction_loss)
            + (float(hidden_state_weight) * hidden_state_loss)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()

        with torch.no_grad():
            approx_kl = torch.mean(old_log_probs - log_probs).abs()
        stats["loss"].append(float(loss.detach().cpu()))
        stats["policy_loss"].append(float(policy_loss.detach().cpu()))
        stats["value_loss"].append(float(value_loss.detach().cpu()))
        stats["entropy"].append(float(entropy.detach().cpu()))
        stats["prediction_loss"].append(float(prediction_loss.detach().cpu()))
        stats["hidden_state_loss"].append(float(hidden_state_loss.detach().cpu()))
        stats["approx_kl"].append(float(approx_kl.detach().cpu()))

    return {key: float(np.mean(values)) if values else 0.0 for key, values in stats.items()}


def _save_checkpoint(
    path: str | Path,
    model: GruAuxActorCritic,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    state_spec: dict[str, Any],
    reward_spec: dict[str, Any],
    variant: AuxiliaryVariant,
    model_hyperparameters: dict[str, Any],
) -> None:
    path = Path(path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
            "encoder_dim": model.encoder_dim,
            "hidden_dim": model.hidden_dim,
            "action_scale": model.action_scale,
            "init_log_std": model.init_log_std,
            "action_low": np.asarray(action_low, dtype=np.float32),
            "action_high": np.asarray(action_high, dtype=np.float32),
            "state_spec": state_spec,
            "reward_spec": reward_spec,
            "variant": asdict(variant),
            "model_hyperparameters": model_hyperparameters,
        },
        _plot_save_path(path),
    )


def _save_partial_checkpoint(
    path: str | Path,
    *,
    model: GruAuxActorCritic,
    optimizer: torch.optim.Optimizer,
    variant: AuxiliaryVariant,
    total_steps: int,
    episode_index: int,
    episode_returns: list[float],
    episode_tracking: list[float],
    episode_transparency: list[float],
    episode_ratio_error: list[float],
    episode_invalid: list[float],
    episode_steps: list[int],
    episode_completed: list[float],
    losses: dict[str, list[float]],
    eval_steps: list[int],
    eval_mean_reward: list[float],
    eval_tracking: list[float],
    eval_transparency: list[float],
    eval_ratio_error: list[float],
) -> None:
    torch.save(
        {
            "variant_key": variant.key,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "total_steps": int(total_steps),
            "episode_index": int(episode_index),
            "episode_returns": list(episode_returns),
            "episode_tracking": list(episode_tracking),
            "episode_transparency": list(episode_transparency),
            "episode_ratio_error": list(episode_ratio_error),
            "episode_invalid": list(episode_invalid),
            "episode_steps": list(episode_steps),
            "episode_completed": list(episode_completed),
            "losses": {str(key): list(values) for key, values in losses.items()},
            "eval_steps": list(eval_steps),
            "eval_mean_reward": list(eval_mean_reward),
            "eval_tracking": list(eval_tracking),
            "eval_transparency": list(eval_transparency),
            "eval_ratio_error": list(eval_ratio_error),
        },
        _plot_save_path(path),
    )


def _load_partial_checkpoint(path: str | Path, *, model: GruAuxActorCritic, optimizer: torch.optim.Optimizer, variant: AuxiliaryVariant, device: torch.device) -> dict[str, Any] | None:
    if not _file_exists(path):
        return None
    load_path = _long_path(path)
    try:
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(load_path, map_location=device)
    if str(checkpoint.get("variant_key", "")) != variant.key:
        return None
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return dict(checkpoint)


def load_gru_aux_policy(path: str | Path, *, device: torch.device) -> GruAuxPolicy:
    load_path = _plot_save_path(path)
    try:
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(load_path, map_location=device)
    model = GruAuxActorCritic(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        encoder_dim=int(checkpoint["encoder_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        action_scale=float(checkpoint["action_scale"]),
        init_log_std=float(checkpoint.get("init_log_std", -0.5)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return GruAuxPolicy(
        model,
        action_low=np.asarray(checkpoint["action_low"], dtype=np.float32),
        action_high=np.asarray(checkpoint["action_high"], dtype=np.float32),
        device=device,
    )


def plot_aux_training_curves(root: Path, label: str, train_payload: dict[str, np.ndarray]) -> None:
    returns = np.asarray(train_payload.get("episode_returns", []), dtype=np.float64)
    if returns.size == 0:
        return
    episodes = np.arange(1, returns.size + 1)
    panels = [
        ("episode_returns", "episode return", 1.0),
        ("episode_tracking_rmse", "tracking RMSE [mm]", 1000.0),
        ("episode_transparency_ratio_error_rmse", "transparency-ratio error RMSE", 1.0),
        ("loss", "total PPO loss", 1.0),
        ("prediction_loss", "prediction MSE", 1.0),
        ("hidden_state_loss", "hidden-state MSE", 1.0),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.0 * len(panels)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (key, ylabel, scale) in zip(axes, panels):
        values = np.asarray(train_payload.get(key, []), dtype=np.float64)
        if values.size == 0:
            continue
        n = min(values.size, episodes.size)
        plotted = values[:n] * float(scale)
        ax.plot(episodes[:n], plotted, lw=0.8, alpha=0.35, color="tab:blue")
        ax.plot(episodes[:n], moving_avg(plotted, min(8, max(1, n))), lw=1.8, color="tab:red")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(f"{label}: GRU-PPO auxiliary training curves")
    axes[-1].set_xlabel("episode")
    fig.savefig(_plot_save_path(root / "aux_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aux_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [str(row["key"]).replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    panels = [
        ("focused_tracking_rmse_mm", "focused tracking RMSE [mm]", "linear"),
        ("focused_transparency_ratio_error_rmse", "focused ratio-error RMSE", "linear"),
        ("focused_transparency_ratio_within_20pct", "ratio within +/-20%", "linear"),
        ("prediction_loss_final", "final prediction MSE", "log"),
        ("hidden_state_loss_final", "final hidden-state MSE", "log"),
        ("train_budget_coverage", "training budget coverage", "linear"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 3.2 * len(panels)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (key, ylabel, yscale) in zip(axes, panels):
        values = np.asarray([_as_float(row.get(key, 0.0), 0.0) for row in rows], dtype=np.float64)
        ax.bar(x, values, color="tab:blue", alpha=0.78)
        if key == "focused_transparency_ratio_within_20pct":
            ax.set_ylim(0.0, 1.0)
        if key == "train_budget_coverage":
            ax.axhline(1.0, color="tab:red", lw=1.2, ls="--")
        if yscale == "log" and np.any(values > 0.0):
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(float(np.min(values[values > 0.0])) * 0.5, 1e-10))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("GRU-PPO auxiliary ablation with fixed basic observation and R5 reward")
    fig.savefig(_plot_save_path(root / "auxiliary_gru_ppo_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    try:
        with open(_long_path(path), "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        return []


def aggregate_focused_metrics(path: Path) -> dict[str, float]:
    rows = _read_csv_rows(path)
    if not rows:
        return {}

    def mean(key: str) -> float:
        values = np.asarray([_as_float(row.get(key), 0.0) for row in rows], dtype=np.float64)
        return float(np.mean(values)) if values.size else 0.0

    return {
        "focused_scenario_count": float(len(rows)),
        "focused_tracking_rmse_mm": 1000.0 * mean("rms_error_m"),
        "focused_post_contact_rmse_mm": 1000.0 * mean("post_contact_rms_error_m"),
        "focused_transparency_rmse_w": mean("transparency_rmse_w"),
        "focused_transparency_ratio_median": mean("transparency_ratio_median"),
        "focused_transparency_ratio_error_rmse": mean("transparency_ratio_error_rmse"),
        "focused_transparency_ratio_valid_fraction": mean("transparency_ratio_valid_fraction"),
        "focused_transparency_ratio_within_20pct": mean("transparency_ratio_within_20pct"),
        "focused_rms_u_v": mean("rms_u_v"),
        "focused_mean_abs_delta_u_v": mean("control_smoothness_mean_abs_delta_v"),
        "focused_mean_abs_delta2_u_v": mean("control_smoothness_mean_abs_delta2_v"),
        "focused_failure_rate": mean("failure_flag"),
    }


def run_focused_eval_for_policy(
    *,
    policy: GruAuxPolicy,
    summary: dict[str, Any],
    env_factory: Any,
    out_dir: Path,
    seed: int,
    save_plots: bool,
    focused_limit: int | None,
) -> dict[str, float]:
    os.makedirs(_long_path(out_dir), exist_ok=True)
    scenarios = build_focused_scenarios(summary)
    if focused_limit is not None:
        scenarios = scenarios[: max(1, int(focused_limit))]
    action_levels = np.asarray(summary.get("action_levels", cfg.V_LEVELS), dtype=np.float64)
    action_limit = float(np.max(np.abs(action_levels))) if action_levels.size else 5.0
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        result = evaluate_policy_on_scenario(
            policy,
            env_factory,
            scenario,
            seed=int(seed) + index,
            deterministic=True,
        )
        metrics = compute_non_bode_metrics(result, action_limit=action_limit)
        rows.append(metrics)
        save_scenario_history_npz(result, out_dir / "histories", metrics)
        if save_plots:
            plot_scenario_result(result, out_dir / "plots" / "scenarios")
            plot_scenario_detail_results(result, out_dir / "plots" / "scenarios")
    write_csv(out_dir / "focused_eval_metrics.csv", rows)
    save_json(
        out_dir / "focused_eval_manifest.json",
        {
            "algo": AUX_ALGO,
            "scenario_count": int(len(scenarios)),
            "metrics_csv": "focused_eval_metrics.csv",
            "history_npz_dir": "histories",
            "scenario_plots_dir": "plots/scenarios",
            "artifacts": {
                "scenario_dashboard_pattern": "<scenario>.png",
                "tracking_plot_pattern": "<scenario>_tracking.png",
                "force_plot_pattern": "<scenario>_force.png",
                "control_plot_pattern": "<scenario>_control.png",
                "transparency_ratio_plot_pattern": "<scenario>_transparency_ratio.png",
            },
        },
    )
    return aggregate_focused_metrics(out_dir / "focused_eval_metrics.csv")


def _final_value(values: list[float]) -> float:
    return float(values[-1]) if values else 0.0


def train_variant(
    *,
    variant: AuxiliaryVariant,
    out_dir: Path,
    env_mode: str,
    env_kwargs: dict[str, Any],
    state_spec: dict[str, Any],
    reward_spec: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    train_reset_options_pool: list[dict[str, Any]] | None = None,
    eval_reset_options_schedule: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], GruAuxPolicy]:
    dirs = mk_run_dirs(out_dir)
    state_variant = build_custom_dqn_state_variant_from_spec(state_spec)
    reward_variant = reward_variant_from_spec(reward_spec)
    eval_env_factory = build_policy_gradient_env_factory(
        algo=PG_ALGO_PPO_CONTINUOUS,
        env_mode=env_mode,
        env_kwargs=env_kwargs,
        state_variant=state_variant,
        reward_variant=reward_variant,
    )
    train_env_kwargs = dict(env_kwargs)
    if train_reset_options_pool:
        train_env_kwargs[PG_TRAIN_RESET_OPTIONS_POOL_KEY] = [dict(row) for row in train_reset_options_pool]
    train_env_factory = build_policy_gradient_env_factory(
        algo=PG_ALGO_PPO_CONTINUOUS,
        env_mode=env_mode,
        env_kwargs=train_env_kwargs,
        state_variant=state_variant,
        reward_variant=reward_variant,
    )
    probe_env = eval_env_factory()
    action_low = np.asarray(probe_env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(probe_env.action_space.high, dtype=np.float32).reshape(-1)
    action_levels = np.asarray(getattr(probe_env, "action_levels", cfg.V_LEVELS), dtype=np.float64).reshape(-1).tolist()
    obs_dim = int(probe_env.obs_dim)
    action_dim = int(action_low.size)
    probe_env.close()

    action_scale = float(np.max(np.abs(np.concatenate([action_low, action_high]))))
    model_hyperparameters = {
        "encoder_dim": int(args.encoder_dim),
        "hidden_dim": int(args.hidden_dim),
        "learning_rate": float(args.learning_rate),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "clip_range": float(args.clip_range),
        "ppo_epochs": int(args.ppo_epochs),
        "value_coef": float(args.value_coef),
        "entropy_coef": float(args.entropy_coef),
        "max_grad_norm": float(args.max_grad_norm),
        "init_log_std": float(args.init_log_std),
        "action_scale": float(action_scale),
    }
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    model = GruAuxActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        encoder_dim=int(args.encoder_dim),
        hidden_dim=int(args.hidden_dim),
        action_scale=action_scale,
        init_log_std=float(args.init_log_std),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), eps=1e-5)
    train_env = train_env_factory()

    requested_episode_steps = max(1, int(round(float(env_kwargs["episode_duration"]) / float(cfg.RL_DT))))
    target_timesteps = int(
        max(1, args.total_timesteps)
        if args.total_timesteps is not None
        else max(1, int(args.train_episodes)) * requested_episode_steps
    )

    episode_returns: list[float] = []
    episode_tracking: list[float] = []
    episode_transparency: list[float] = []
    episode_ratio_error: list[float] = []
    episode_invalid: list[float] = []
    episode_steps: list[int] = []
    episode_completed: list[float] = []
    losses: dict[str, list[float]] = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "prediction_loss": [],
        "hidden_state_loss": [],
        "approx_kl": [],
    }
    eval_steps: list[int] = []
    eval_mean_reward: list[float] = []
    eval_tracking: list[float] = []
    eval_transparency: list[float] = []
    eval_ratio_error: list[float] = []
    total_steps = 0

    partial_checkpoint_path = Path(dirs["logs"]) / "partial_checkpoint.pt"
    if bool(args.resume_partial):
        partial = _load_partial_checkpoint(
            partial_checkpoint_path,
            model=model,
            optimizer=optimizer,
            variant=variant,
            device=device,
        )
        if partial:
            total_steps = int(partial.get("total_steps", 0) or 0)
            episode_returns = [float(value) for value in partial.get("episode_returns", [])]
            episode_tracking = [float(value) for value in partial.get("episode_tracking", [])]
            episode_transparency = [float(value) for value in partial.get("episode_transparency", [])]
            episode_ratio_error = [float(value) for value in partial.get("episode_ratio_error", [])]
            episode_invalid = [float(value) for value in partial.get("episode_invalid", [])]
            episode_steps = [int(value) for value in partial.get("episode_steps", [])]
            episode_completed = [float(value) for value in partial.get("episode_completed", [])]
            loaded_losses = dict(partial.get("losses", {}))
            for key in losses:
                losses[key] = [float(value) for value in loaded_losses.get(key, [])]
            eval_steps = [int(value) for value in partial.get("eval_steps", [])]
            eval_mean_reward = [float(value) for value in partial.get("eval_mean_reward", [])]
            eval_tracking = [float(value) for value in partial.get("eval_tracking", [])]
            eval_transparency = [float(value) for value in partial.get("eval_transparency", [])]
            eval_ratio_error = [float(value) for value in partial.get("eval_ratio_error", [])]
            print(
                f"[resume] {variant.key}: {total_steps}/{target_timesteps} timesteps, "
                f"{len(episode_returns)} episodes from {partial_checkpoint_path}",
                flush=True,
            )
    policy = GruAuxPolicy(model, action_low=action_low, action_high=action_high, device=device)
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            total=target_timesteps,
            initial=min(int(total_steps), int(target_timesteps)),
            desc=f"{variant.key} train",
            unit="ts",
            dynamic_ncols=True,
        )

    episode_index = int(len(episode_returns))
    checkpoint_every_timesteps = int(max(0, args.checkpoint_every_timesteps))
    last_checkpoint_steps = int(total_steps)
    while total_steps < target_timesteps:
        batch = collect_episode(
            train_env,
            model,
            device=device,
            action_low=action_low,
            action_high=action_high,
            seed=int(args.seed) + int(episode_index),
        )
        stats = update_gru_ppo(
            model,
            optimizer,
            batch,
            device=device,
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            value_coef=float(args.value_coef),
            entropy_coef=float(args.entropy_coef),
            prediction_weight=float(variant.prediction_weight),
            hidden_state_weight=float(variant.hidden_state_weight),
            epochs=int(args.ppo_epochs),
            max_grad_norm=float(args.max_grad_norm),
        )
        previous_steps = int(total_steps)
        total_steps += int(batch.steps)
        metrics = batch.metrics
        episode_returns.append(float(np.sum(batch.rewards)))
        episode_tracking.append(float(metrics.get("tracking_rmse_m", 0.0)))
        episode_transparency.append(float(metrics.get("transparency_rmse_w", 0.0)))
        episode_ratio_error.append(float(metrics.get("transparency_ratio_error_rmse", 0.0)))
        episode_invalid.append(float(metrics.get("invalid_episode", 0.0)))
        episode_steps.append(int(batch.steps))
        episode_completed.append(float(batch.termination_reason == "max_steps" or batch.truncated))
        for key in losses:
            losses[key].append(float(stats.get(key, 0.0)))
        if progress_bar is not None:
            progress_bar.update(max(0, min(total_steps, target_timesteps) - min(previous_steps, target_timesteps)))
            progress_bar.set_postfix(
                episodes=len(episode_returns),
                steps=total_steps,
                track_mm=f"{1000.0 * episode_tracking[-1]:.2f}",
                pred=f"{losses['prediction_loss'][-1]:.2e}",
                hidden=f"{losses['hidden_state_loss'][-1]:.2e}",
                refresh=False,
            )
        episode_number = int(episode_index) + 1
        should_eval = (
            episode_number == 1
            or episode_number % int(max(1, args.eval_every_episodes)) == 0
            or total_steps >= target_timesteps
        )
        if should_eval:
            eval_metrics, _ = evaluate_policy_gradient(
                policy,
                eval_env_factory,
                n_episodes=int(args.eval_episodes),
                seed_offset=10_000 + int(episode_number),
                reset_options_schedule=eval_reset_options_schedule,
            )
            eval_steps.append(episode_number)
            eval_mean_reward.append(float(eval_metrics.get("mean_reward", 0.0)))
            eval_tracking.append(float(eval_metrics.get("tracking_rmse_m", 0.0)))
            eval_transparency.append(float(eval_metrics.get("transparency_rmse_w", 0.0)))
            eval_ratio_error.append(float(eval_metrics.get("transparency_ratio_error_rmse", 0.0)))
        episode_index += 1
        if checkpoint_every_timesteps and total_steps - last_checkpoint_steps >= checkpoint_every_timesteps:
            _save_partial_checkpoint(
                partial_checkpoint_path,
                model=model,
                optimizer=optimizer,
                variant=variant,
                total_steps=total_steps,
                episode_index=episode_index,
                episode_returns=episode_returns,
                episode_tracking=episode_tracking,
                episode_transparency=episode_transparency,
                episode_ratio_error=episode_ratio_error,
                episode_invalid=episode_invalid,
                episode_steps=episode_steps,
                episode_completed=episode_completed,
                losses=losses,
                eval_steps=eval_steps,
                eval_mean_reward=eval_mean_reward,
                eval_tracking=eval_tracking,
                eval_transparency=eval_transparency,
                eval_ratio_error=eval_ratio_error,
            )
            last_checkpoint_steps = int(total_steps)
            gc.collect()
        elif episode_index % 25 == 0:
            gc.collect()

    if progress_bar is not None:
        progress_bar.close()
    train_env.close()

    final_eval_metrics, final_history = evaluate_policy_gradient(
        policy,
        eval_env_factory,
        n_episodes=int(args.test_episodes),
        seed_offset=20_000,
        reset_options_schedule=eval_reset_options_schedule,
    )
    save_history_npz(final_history, Path(dirs["episodes"]) / "test.npz")
    save_policy_gradient_visuals(
        final_history,
        dirs["plots"],
        variant.label,
        env_switch_time=float(env_kwargs["env_switch_time"]),
        action_mode="continuous",
        action_levels=action_levels,
    )

    model_path = Path(dirs["models"]) / "model.pt"
    _save_checkpoint(
        model_path,
        model,
        action_low=action_low,
        action_high=action_high,
        state_spec=state_spec,
        reward_spec=reward_spec,
        variant=variant,
        model_hyperparameters=model_hyperparameters,
    )

    train_payload: dict[str, np.ndarray] = {
        "episode_returns": np.asarray(episode_returns, dtype=np.float64),
        "episode_tracking_rmse": np.asarray(episode_tracking, dtype=np.float64),
        "episode_transparency_rmse": np.asarray(episode_transparency, dtype=np.float64),
        "episode_transparency_ratio_error_rmse": np.asarray(episode_ratio_error, dtype=np.float64),
        "episode_invalid": np.asarray(episode_invalid, dtype=np.float64),
        "episode_steps": np.asarray(episode_steps, dtype=np.int64),
        "episode_completed": np.asarray(episode_completed, dtype=np.float64),
        "eval_steps": np.asarray(eval_steps, dtype=np.int64),
        "eval_mean_reward": np.asarray(eval_mean_reward, dtype=np.float64),
        "eval_tracking_rmse": np.asarray(eval_tracking, dtype=np.float64),
        "eval_transparency_rmse": np.asarray(eval_transparency, dtype=np.float64),
        "eval_transparency_ratio_error_rmse": np.asarray(eval_ratio_error, dtype=np.float64),
    }
    for key, values in losses.items():
        train_payload[key] = np.asarray(values, dtype=np.float64)
    np.savez(Path(dirs["logs"]) / "train.npz", **train_payload)
    save_training_plot(
        train_payload["episode_returns"],
        train_payload["episode_tracking_rmse"],
        train_payload["episode_transparency_rmse"],
        Path(dirs["plots"]) / "train.png",
        variant.label,
        losses=train_payload["loss"],
        eval_payload={
            "steps": train_payload["eval_steps"],
            "mean_reward": train_payload["eval_mean_reward"],
            "tracking_rmse_m": train_payload["eval_tracking_rmse"],
            "transparency_rmse_w": train_payload["eval_transparency_rmse"],
        },
    )
    plot_aux_training_curves(Path(dirs["plots"]), variant.label, train_payload)

    result = RunResult(
        label=variant.label,
        family=AUX_ALGO,
        mean_reward=float(final_eval_metrics.get("mean_reward", 0.0)),
        tracking_rmse_m=float(final_eval_metrics.get("tracking_rmse_m", 0.0)),
        transparency_rmse_w=float(final_eval_metrics.get("transparency_rmse_w", 0.0)),
        pre_switch_tracking_rmse_m=float(final_eval_metrics.get("pre_switch_tracking_rmse_m", 0.0)),
        post_switch_tracking_rmse_m=float(final_eval_metrics.get("post_switch_tracking_rmse_m", 0.0)),
        pre_switch_transparency_rmse_w=float(final_eval_metrics.get("pre_switch_transparency_rmse_w", 0.0)),
        post_switch_transparency_rmse_w=float(final_eval_metrics.get("post_switch_transparency_rmse_w", 0.0)),
        invalid_episode_rate=float(final_eval_metrics.get("invalid_episode", 0.0)),
        history=final_history,
        out_dir=dirs["base"],
        tensorboard_dir=dirs["tensorboard"],
        model_path=str(model_path),
        reward_variant=reward_variant.name,
        state_variant=state_variant.name,
    )
    summary = write_run_summary(
        dirs,
        result,
        extra={
            "algo": AUX_ALGO,
            "algo_display_name": "GRU-PPO with auxiliary dynamics heads",
            "env_mode": env_mode,
            "master_input_mode": cfg.MASTER_INPUT_FORCE,
            "total_episodes": int(args.train_episodes),
            "total_timesteps": int(total_steps),
            "train_requested_timesteps": int(target_timesteps),
            "train_requested_episode_steps": int(requested_episode_steps),
            "train_requested_episodes": int(args.train_episodes),
            "train_finished_episodes": int(len(episode_returns)),
            "train_completed_episodes": int(np.sum(train_payload["episode_completed"])),
            "train_episode_coverage": float(total_steps / max(target_timesteps, 1)),
            "train_budget_coverage": float(total_steps / max(target_timesteps, 1)),
            "train_signal_count": int(len(train_reset_options_pool or [])),
            "train_reset_options_pool": list(train_reset_options_pool or []),
            "eval_signal_count": int(len(eval_reset_options_schedule or [])),
            "eval_reset_options_schedule": list(eval_reset_options_schedule or []),
            "test_episodes": int(args.test_episodes),
            "eval_episodes": int(args.eval_episodes),
            "eval_every_episodes": int(args.eval_every_episodes),
            "evaluation_history_mode": "mean_over_test_episodes",
            "obs_dim": int(obs_dim),
            "state_features": list(BASIC_OBS_FEATURES),
            "state_variant_description": str(state_variant.description),
            "state_spec": state_variant.metadata,
            "reward_config": asdict(reward_variant),
            "reward_spec": reward_spec,
            "reward_basis": "R5_second_order from the configured physics reward ablation study unless --reward-key is changed.",
            "auxiliary_variant": asdict(variant),
            "auxiliary_losses": {
                "prediction_loss": {
                    "enabled": bool(variant.prediction_weight > 0.0),
                    "weight": float(variant.prediction_weight),
                    "target": list(PREDICTION_TARGET_LABELS),
                    "action_input": "detached applied continuous action",
                    "description": "Predict next normalized position deltas for x_m and x_s.",
                },
                "hidden_state_loss": {
                    "enabled": bool(variant.hidden_state_weight > 0.0),
                    "weight": float(variant.hidden_state_weight),
                    "target": list(HIDDEN_TARGET_LABELS),
                    "deployment": "training-only privileged simulator targets",
                },
            },
            "model_hyperparameters": model_hyperparameters,
            "model_device": str(device),
            "episode_duration": float(env_kwargs["episode_duration"]),
            "env_switch_time": float(env_kwargs["env_switch_time"]),
            "terminate_on_error": bool(env_kwargs["terminate_on_error"]),
            "enforce_stroke_limit": bool(env_kwargs.get("enforce_stroke_limit", True)),
            "stroke_limit_mode": str(env_kwargs.get("stroke_limit_mode", "terminate")),
            "reset_options": dict(env_kwargs.get("reset_options", {})),
            "action_space_type": "continuous",
            "action_levels": action_levels,
            "eval_metrics": final_eval_metrics,
            "tracking_mae_m": float(final_eval_metrics.get("tracking_mae_m", 0.0)),
            "tracking_max_abs_m": float(final_eval_metrics.get("tracking_max_abs_m", 0.0)),
            "velocity_error_rmse_mps": float(final_eval_metrics.get("velocity_error_rmse_mps", 0.0)),
            "acceleration_error_rmse_mps2": float(final_eval_metrics.get("acceleration_error_rmse_mps2", 0.0)),
            "transparency_ratio_median": float(final_eval_metrics.get("transparency_ratio_median", 0.0)),
            "transparency_ratio_error_rmse": float(final_eval_metrics.get("transparency_ratio_error_rmse", 0.0)),
            "transparency_ratio_valid_fraction": float(final_eval_metrics.get("transparency_ratio_valid_fraction", 0.0)),
            "transparency_ratio_within_20pct": float(final_eval_metrics.get("transparency_ratio_within_20pct", 0.0)),
            "mean_abs_u_v": float(final_eval_metrics.get("mean_abs_u_v", 0.0)),
            "rms_u_v": float(final_eval_metrics.get("rms_u_v", 0.0)),
            "control_energy_v2_s": float(final_eval_metrics.get("control_energy_v2_s", 0.0)),
            "max_abs_u_v": float(final_eval_metrics.get("max_abs_u_v", 0.0)),
            "saturation_fraction": float(final_eval_metrics.get("saturation_fraction", 0.0)),
            "mean_abs_delta_u_v": float(final_eval_metrics.get("mean_abs_delta_u_v", 0.0)),
            "rms_delta_u_v": float(final_eval_metrics.get("rms_delta_u_v", 0.0)),
            "max_abs_delta_u_v": float(final_eval_metrics.get("max_abs_delta_u_v", 0.0)),
            "mean_abs_delta2_u_v": float(final_eval_metrics.get("mean_abs_delta2_u_v", 0.0)),
            "rms_delta2_u_v": float(final_eval_metrics.get("rms_delta2_u_v", 0.0)),
            "max_abs_delta2_u_v": float(final_eval_metrics.get("max_abs_delta2_u_v", 0.0)),
            "prediction_loss_final": _final_value(losses["prediction_loss"]),
            "hidden_state_loss_final": _final_value(losses["hidden_state_loss"]),
            "policy_loss_final": _final_value(losses["policy_loss"]),
            "value_loss_final": _final_value(losses["value_loss"]),
        },
    )
    if _file_exists(partial_checkpoint_path):
        os.remove(_long_path(partial_checkpoint_path))
    return summary, policy


def row_from_summary(
    variant: AuxiliaryVariant,
    summary: dict[str, Any],
    focused: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": variant.key,
        "label": variant.label,
        "note": variant.note,
        "prediction_weight": float(variant.prediction_weight),
        "hidden_state_weight": float(variant.hidden_state_weight),
        "obs_dim": int(summary.get("obs_dim", len(BASIC_OBS_FEATURES))),
        "state_features": " ".join(str(feature) for feature in summary.get("state_features", BASIC_OBS_FEATURES)),
        "reward_basis": str(summary.get("reward_basis", "")),
        "out_dir": str(summary.get("out_dir", "")),
        "model_path": str(summary.get("model_path", "")),
        "train_requested_episodes": int(summary.get("train_requested_episodes", summary.get("total_episodes", 0)) or 0),
        "train_requested_timesteps": int(summary.get("train_requested_timesteps", 0) or 0),
        "train_finished_episodes": int(summary.get("train_finished_episodes", 0) or 0),
        "train_completed_episodes": int(summary.get("train_completed_episodes", 0) or 0),
        "train_episode_coverage": float(summary.get("train_episode_coverage", 0.0) or 0.0),
        "train_budget_coverage": float(summary.get("train_budget_coverage", summary.get("train_episode_coverage", 0.0)) or 0.0),
        "total_timesteps": int(summary.get("total_timesteps", 0) or 0),
        "eval_episodes": int(summary.get("eval_episodes", 0) or 0),
        "test_episodes": int(summary.get("test_episodes", 0) or 0),
        "eval_every_episodes": int(summary.get("eval_every_episodes", 0) or 0),
        "train_signal_count": int(summary.get("train_signal_count", 0) or 0),
        "eval_signal_count": int(summary.get("eval_signal_count", 0) or 0),
        "prediction_loss_final": float(summary.get("prediction_loss_final", 0.0) or 0.0),
        "hidden_state_loss_final": float(summary.get("hidden_state_loss_final", 0.0) or 0.0),
        "policy_loss_final": float(summary.get("policy_loss_final", 0.0) or 0.0),
        "value_loss_final": float(summary.get("value_loss_final", 0.0) or 0.0),
    }
    for key in (
        "mean_reward",
        "tracking_rmse_m",
        "tracking_mae_m",
        "tracking_max_abs_m",
        "velocity_error_rmse_mps",
        "acceleration_error_rmse_mps2",
        "transparency_rmse_w",
        "transparency_ratio_median",
        "transparency_ratio_error_rmse",
        "transparency_ratio_valid_fraction",
        "transparency_ratio_within_20pct",
        "rms_u_v",
        "mean_abs_delta_u_v",
        "mean_abs_delta2_u_v",
        "invalid_episode_rate",
    ):
        row[key] = summary.get(key, "")
    row.update(focused)
    return row


def load_summary(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_available_rows(
    root: Path,
    variants: tuple[AuxiliaryVariant, ...],
    current_rows: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    current_rows = dict(current_rows or {})
    rows: list[dict[str, Any]] = []
    for variant in variants:
        if variant.key in current_rows:
            rows.append(current_rows[variant.key])
            continue
        summary_path = root / _variant_dir_name(variant) / "gru" / "l" / "summary.json"
        if not summary_path.exists():
            continue
        focused_csv = root / _variant_dir_name(variant) / "focused_eval" / "focused_eval_metrics.csv"
        rows.append(row_from_summary(variant, load_summary(summary_path), aggregate_focused_metrics(focused_csv)))
    return rows


def _select_reward_ablation(key: str):
    for ablation in build_ablations():
        if ablation.key == key:
            return ablation
    known = ", ".join(ablation.key for ablation in build_ablations())
    raise KeyError(f"Unknown reward ablation key {key!r}. Known keys: {known}")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    _configure_torch_runtime()
    device = _resolve_device(args.device)
    root = policy_gradient_suite_root(args.fe_mode, args.study_name)
    root.mkdir(parents=True, exist_ok=True)
    specs_root = root / "specs"
    specs_root.mkdir(parents=True, exist_ok=True)
    train_reset_options_pool = load_reset_options_json(args.train_reset_options_json)
    eval_reset_options_schedule = load_reset_options_json(args.eval_reset_options_json)

    scale_catalog = calibrate_scale_catalog(root, args.calibration_study)
    reward_ablation = _select_reward_ablation(args.reward_key)
    reward_spec = build_reward_spec(reward_ablation, scale_catalog)
    state_spec = build_state_spec()
    save_json(specs_root / "reward_scale_catalog.json", scale_catalog)
    save_json(specs_root / "basic_obs_state.json", state_spec)
    save_json(specs_root / f"{reward_ablation.key}_reward.json", reward_spec)

    env_args = SimpleNamespace(
        episode_duration=args.episode_duration,
        env_switch_time=args.env_switch_time,
        disable_terminate_on_error=args.disable_terminate_on_error,
        legacy_baseline_env=False,
        disable_stroke_limit=False,
        stroke_limit_mode=args.stroke_limit_mode,
        reset_position_mode=args.reset_position_mode,
        action_levels=None,
        force_amp=args.force_amp,
        force_bias=args.force_bias,
        force_freq_rad=args.force_freq_rad,
        force_phase=args.force_phase,
        force_waveform=args.force_waveform,
        fe_mode=args.fe_mode,
    )
    env_kwargs = replica_env_kwargs_from_args(env_args)
    state_variant = build_custom_dqn_state_variant_from_spec(state_spec)
    reward_variant = reward_variant_from_spec(reward_spec)
    eval_env_factory = build_policy_gradient_env_factory(
        algo=PG_ALGO_PPO_CONTINUOUS,
        env_mode=args.env_mode,
        env_kwargs=env_kwargs,
        state_variant=state_variant,
        reward_variant=reward_variant,
    )

    variants = build_auxiliary_variants()
    if args.only:
        requested = {str(key) for key in args.only}
        variants = tuple(
            variant
            for variant in variants
            if variant.key in requested or _variant_dir_name(variant) in requested
        )
        if not variants:
            raise ValueError(f"No auxiliary variants matched --only={sorted(requested)}")
    current_rows: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = collect_available_rows(root, variants, current_rows)
    for index, variant in enumerate(variants, start=1):
        out_dir = root / _variant_dir_name(variant) / "gru"
        summary_path = out_dir / "l" / "summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[{index}/{len(variants)}] skip existing train {variant.key}", flush=True)
            summary = load_summary(summary_path)
            policy = load_gru_aux_policy(str(summary["model_path"]), device=device)
        else:
            print(f"[{index}/{len(variants)}] train {variant.key}: {variant.note}", flush=True)
            summary, policy = train_variant(
                variant=variant,
                out_dir=out_dir,
                env_mode=args.env_mode,
                env_kwargs=env_kwargs,
                state_spec=state_spec,
                reward_spec=reward_spec,
                args=args,
                device=device,
                train_reset_options_pool=train_reset_options_pool,
                eval_reset_options_schedule=eval_reset_options_schedule,
            )

        focused_dir = root / _variant_dir_name(variant) / "focused_eval"
        focused_csv = focused_dir / "focused_eval_metrics.csv"
        if args.no_focused_eval:
            print(f"[{index}/{len(variants)}] focused eval disabled {variant.key}", flush=True)
            focused = {}
        elif args.skip_existing and _file_exists(focused_csv):
            print(f"[{index}/{len(variants)}] skip existing focused eval {variant.key}", flush=True)
            focused = aggregate_focused_metrics(focused_csv)
        else:
            print(f"[{index}/{len(variants)}] focused eval {variant.key}", flush=True)
            focused = run_focused_eval_for_policy(
                policy=policy,
                summary=summary,
                env_factory=eval_env_factory,
                out_dir=focused_dir,
                seed=int(args.focused_seed),
                save_plots=not bool(args.no_plots),
                focused_limit=args.focused_limit,
            )
        current_rows[variant.key] = row_from_summary(variant, summary, focused)
        rows = collect_available_rows(root, variants, current_rows)
        write_csv(root / "summary.csv", rows)
        plot_aux_summary(root, rows)

    rows = collect_available_rows(root, variants, current_rows)
    save_json(
        root / "study_manifest.json",
        {
            "study_name": args.study_name,
            "objective": "GRU-PPO auxiliary dynamics-head ablation with fixed basic observation space.",
            "reward_key": reward_ablation.key,
            "reward_terms": [term["name"] for term in reward_ablation.terms],
            "basic_observation_features": list(BASIC_OBS_FEATURES),
            "prediction_target_labels": list(PREDICTION_TARGET_LABELS),
            "hidden_target_labels": list(HIDDEN_TARGET_LABELS),
            "device": str(device),
            "focused_eval_seed": int(args.focused_seed),
            "training_protocol": {
                "train_episodes": int(args.train_episodes),
                "total_timesteps": None if args.total_timesteps is None else int(args.total_timesteps),
                "test_episodes": int(args.test_episodes),
                "eval_episodes": int(args.eval_episodes),
                "eval_every_episodes": int(args.eval_every_episodes),
                "learning_rate": float(args.learning_rate),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "clip_range": float(args.clip_range),
                "ppo_epochs": int(args.ppo_epochs),
                "entropy_coef": float(args.entropy_coef),
                "train_signal_count": int(len(train_reset_options_pool)),
                "eval_signal_count": int(len(eval_reset_options_schedule)),
                "train_reset_options_json": None if args.train_reset_options_json is None else str(args.train_reset_options_json),
                "eval_reset_options_json": None if args.eval_reset_options_json is None else str(args.eval_reset_options_json),
            },
            "rows": rows,
        },
    )
    rows = collect_available_rows(root, variants, current_rows)
    write_csv(root / "summary.csv", rows)
    plot_aux_summary(root, rows)
    print(f"summary_csv={root / 'summary.csv'}", flush=True)
    print(f"summary_plot={root / 'auxiliary_gru_ppo_summary.png'}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRU-PPO auxiliary dynamics-head ablations.")
    parser.add_argument("--study-name", default="physics_auxiliary_gru_ppo_02_fair_500k")
    parser.add_argument("--calibration-study", default="physics_informed_formulations_02_fair_500k")
    parser.add_argument("--reward-key", default="R5_second_order")
    parser.add_argument("--env-mode", default=cfg.ENV_MODE_CHANGING, choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING])
    parser.add_argument("--fe-mode", default=FE_MODE_DYNAMICS)
    parser.add_argument("--episode-duration", type=float, default=30.0)
    parser.add_argument("--env-switch-time", type=float, default=10.0)
    parser.add_argument("--reset-position-mode", default="midpoint", choices=["midpoint", "zero"])
    parser.add_argument("--stroke-limit-mode", default="clamp", choices=["terminate", "clamp"])
    parser.add_argument("--force-amp", type=float, default=10.0)
    parser.add_argument("--force-bias", type=float, default=0.0)
    parser.add_argument("--force-freq-rad", type=float, default=1.0)
    parser.add_argument("--force-phase", type=float, default=0.0)
    parser.add_argument("--force-waveform", default="sine", choices=["sine", "cosine", "square", "ramp", "multisine"])
    parser.add_argument("--train-episodes", type=int, default=334)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Override the fixed interaction budget. The default matches the full PPO baseline notebook budget.",
    )
    parser.add_argument("--test-episodes", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-every-episodes", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focused-seed", type=int, default=42)
    parser.add_argument("--encoder-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--init-log-std", type=float, default=-0.5)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--train-reset-options-json", default=None)
    parser.add_argument("--eval-reset-options-json", default=None)
    parser.add_argument("--no-focused-eval", action="store_true")
    parser.add_argument("--focused-limit", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Optional auxiliary variant key/dir filter, e.g. G1_gru_prediction or G1p.")
    parser.add_argument("--checkpoint-every-timesteps", type=int, default=50_000)
    parser.add_argument("--resume-partial", dest="resume_partial", action="store_true", default=True)
    parser.add_argument("--no-resume-partial", dest="resume_partial", action="store_false")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
