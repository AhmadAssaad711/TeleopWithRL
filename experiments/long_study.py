"""Long-form study runner for Q-learning, MRAC, and DQN reward ablations."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
import tempfile
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional until runtime
    SummaryWriter = None

from .. import config as cfg
from ..dqn_agent import DQNAgent
from ..mrac_controller import FilteredMRACController, baseline_mrac_inputs
from ..q_learning_agent import QLearningAgent
from ..teleop_env import TeleopEnv


EnvFactory = Callable[[], Any]


@dataclass(frozen=True)
class RewardVariant:
    name: str
    tracking_weight: float
    transparency_weight: float
    jerk_weight: float
    use_jerk: bool


@dataclass
class RunResult:
    label: str
    family: str
    mean_reward: float
    tracking_rmse_m: float
    transparency_rmse_w: float
    history: dict[str, Any]
    out_dir: str
    tensorboard_dir: str
    model_path: str


def _require_tensorboard():
    if SummaryWriter is None:
        raise ImportError(
            "TensorBoard logging requires 'tensorboard'. Install it with "
            "'pip install tensorboard' in the active environment."
        )
    return SummaryWriter


def _moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x
    width = max(1, min(int(window), x.size))
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(x, kernel, mode="same")


def _history_array(history: dict[str, Any], key: str, dtype=np.float64) -> np.ndarray:
    values = history.get(key, [])
    try:
        return np.asarray(values, dtype=dtype)
    except (TypeError, ValueError):
        return np.asarray(values, dtype=object)


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def _study_root(study_name: str | None) -> str:
    name = study_name or f"long_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    package_root = os.path.dirname(os.path.dirname(__file__))
    root = os.path.join(package_root, cfg.RESULTS_ROOT_DIR, name)
    os.makedirs(root, exist_ok=True)
    return root


def _tensorboard_root() -> str:
    base = os.getenv("LOCALAPPDATA") or tempfile.gettempdir()
    root = os.path.join(base, "TeleopWithRL_tb")
    os.makedirs(root, exist_ok=True)
    return root


def _tensorboard_study_root(study_root: str) -> str:
    package_root = os.path.dirname(os.path.dirname(__file__))
    rel = os.path.relpath(study_root, package_root)
    parts = [part for part in rel.split(os.sep) if part not in (".", "..")]
    root = os.path.join(_tensorboard_root(), *parts)
    os.makedirs(root, exist_ok=True)
    return root


def _configure_process_env(cpu_threads: int) -> int:
    threads = max(1, int(cpu_threads))
    for env_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[env_var] = str(threads)
    return threads


def _worker_runtime_init(torch_threads: int) -> None:
    threads = max(1, int(torch_threads))
    try:
        import torch

        torch.set_num_threads(threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass


def _mk_dirs(base_dir: str) -> dict[str, str]:
    package_root = os.path.dirname(os.path.dirname(__file__))
    rel = os.path.relpath(base_dir, package_root)
    tb_parts = [part for part in rel.split(os.sep) if part not in (".", "..")]
    paths = {
        "base": base_dir,
        "models": os.path.join(base_dir, "m"),
        "logs": os.path.join(base_dir, "l"),
        "plots": os.path.join(base_dir, "p"),
        "episodes": os.path.join(base_dir, "e"),
        "tensorboard": os.path.join(_tensorboard_root(), *tb_parts),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def _save_history_npz(history: dict[str, Any], out_path: str) -> None:
    payload: dict[str, Any] = {}
    for key, value in history.items():
        if isinstance(value, list):
            try:
                payload[key] = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                payload[key] = np.asarray(value, dtype=object)
        else:
            payload[key] = value
    np.savez(out_path, **payload)


def _compute_eval_metrics(history: dict[str, Any]) -> tuple[float, float, float]:
    rewards = _history_array(history, "reward", dtype=np.float64)
    pos_error = _history_array(history, "pos_error", dtype=np.float64)
    transparency_error = _history_array(history, "transparency_error", dtype=np.float64)
    mean_reward = float(rewards.sum()) if rewards.size else 0.0
    tracking_rmse = float(np.sqrt(np.mean(pos_error ** 2))) if pos_error.size else 0.0
    transparency_rmse = (
        float(np.sqrt(np.mean(transparency_error ** 2))) if transparency_error.size else 0.0
    )
    return mean_reward, tracking_rmse, transparency_rmse


def _greedy_q_action(q_values: np.ndarray) -> int:
    max_q = float(np.max(q_values))
    best = np.flatnonzero(q_values == max_q)
    zero_action = int(np.argmin(np.abs(cfg.V_LEVELS)))
    return zero_action if zero_action in best else int(best[0])


def _build_reward_variants() -> list[RewardVariant]:
    return [
        RewardVariant("dqn_r01_t40_tr06_j005", 40.0, 6.0, 0.05, True),
        RewardVariant("dqn_r02_t50_tr06_j005", 50.0, 6.0, 0.05, True),
        RewardVariant("dqn_r03_t50_tr08_j010", 50.0, 8.0, 0.10, True),
        RewardVariant("dqn_r04_t60_tr08_j010", 60.0, 8.0, 0.10, True),
        RewardVariant("dqn_r05_t60_tr10_j015", 60.0, 10.0, 0.15, True),
        RewardVariant("dqn_r06_t70_tr10_j020", 70.0, 10.0, 0.20, True),
        RewardVariant("dqn_r07_t80_tr12_j025", 80.0, 12.0, 0.25, True),
        RewardVariant("dqn_r08_t50_tr06_nojerk", 50.0, 6.0, 0.00, False),
        RewardVariant("dqn_r09_t60_tr08_nojerk", 60.0, 8.0, 0.00, False),
        RewardVariant("dqn_r10_t70_tr10_nojerk", 70.0, 10.0, 0.00, False),
        RewardVariant("dqn_r11_t40_tr06_nojerk", 40.0, 6.0, 0.00, False),
    ]


def _reward_variant_from_dict(payload: dict[str, Any]) -> RewardVariant:
    return RewardVariant(
        name=str(payload["name"]),
        tracking_weight=float(payload["tracking_weight"]),
        transparency_weight=float(payload["transparency_weight"]),
        jerk_weight=float(payload["jerk_weight"]),
        use_jerk=bool(payload["use_jerk"]),
    )


class RewardAblationEnv:
    """Wrap TeleopEnv so DQN reward variants can change structure without forking the env."""

    def __init__(self, base_env: TeleopEnv, variant: RewardVariant):
        self.base_env = base_env
        self.variant = variant
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        self._reward_history: dict[str, list[Any]] = {}
        self._prev_u_v = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    def reset(self, *args, **kwargs):
        obs, info = self.base_env.reset(*args, **kwargs)
        self._prev_u_v = 0.0
        self._reward_history = {
            "reward": [],
            "reward_track": [],
            "reward_transparency": [],
            "reward_jerk": [],
            "action_delta": [],
            "reward_variant_name": [],
        }
        return obs, info

    def _compute_reward(self) -> tuple[float, float, float, float, float]:
        history = self.base_env.render() or {}
        pos_error = float(history.get("pos_error", [0.0])[-1])
        transparency_error = float(history.get("transparency_error", [0.0])[-1])
        u_v = float(history.get("u_v", [0.0])[-1])
        action_delta = u_v - self._prev_u_v
        self._prev_u_v = u_v

        norm_pos_error = float(
            np.clip(pos_error / cfg.MAX_POSITION_ERROR, -cfg.POS_ERR_NORM_CLIP, cfg.POS_ERR_NORM_CLIP)
        )
        norm_transparency_error = transparency_error / cfg.MAX_POWER_ERROR

        track_term = self.variant.tracking_weight * (norm_pos_error ** 2)
        transparency_term = self.variant.transparency_weight * (norm_transparency_error ** 2)
        jerk_term = self.variant.jerk_weight * (action_delta ** 2) if self.variant.use_jerk else 0.0
        reward = -(track_term + transparency_term + jerk_term)
        return reward, track_term, transparency_term, jerk_term, action_delta

    def _record_reward_terms(
        self,
        reward: float,
        track_term: float,
        transparency_term: float,
        jerk_term: float,
        action_delta: float,
    ) -> None:
        self._reward_history["reward"].append(float(reward))
        self._reward_history["reward_track"].append(float(track_term))
        self._reward_history["reward_transparency"].append(float(transparency_term))
        self._reward_history["reward_jerk"].append(float(jerk_term))
        self._reward_history["action_delta"].append(float(action_delta))
        self._reward_history["reward_variant_name"].append(self.variant.name)

    def step(self, action):
        obs, _, terminated, truncated, info = self.base_env.step(action)
        reward, track_term, transparency_term, jerk_term, action_delta = self._compute_reward()
        self._record_reward_terms(reward, track_term, transparency_term, jerk_term, action_delta)
        return obs, reward, terminated, truncated, info

    def step_voltage(self, u_v: float):
        obs, _, terminated, truncated, info = self.base_env.step_voltage(u_v)
        reward, track_term, transparency_term, jerk_term, action_delta = self._compute_reward()
        self._record_reward_terms(reward, track_term, transparency_term, jerk_term, action_delta)
        return obs, reward, terminated, truncated, info

    def render(self):
        base_history = self.base_env.render() or {}
        merged: dict[str, Any] = {}
        for key, value in base_history.items():
            merged[key] = list(value) if isinstance(value, list) else value
        merged["base_reward"] = list(base_history.get("reward", []))
        merged["base_reward_track"] = list(base_history.get("reward_track", []))
        merged["base_reward_transparency"] = list(base_history.get("reward_transparency", []))
        merged["reward"] = list(self._reward_history.get("reward", []))
        merged["reward_track"] = list(self._reward_history.get("reward_track", []))
        merged["reward_transparency"] = list(self._reward_history.get("reward_transparency", []))
        merged["reward_jerk"] = list(self._reward_history.get("reward_jerk", []))
        merged["action_delta"] = list(self._reward_history.get("action_delta", []))
        merged["reward_variant_name"] = list(self._reward_history.get("reward_variant_name", []))
        return merged


def _save_training_plot(
    returns: np.ndarray,
    tracking_rmse: np.ndarray,
    transparency_rmse: np.ndarray,
    out_path: str,
    title: str,
    losses: np.ndarray | None = None,
) -> None:
    n_rows = 4 if losses is not None and losses.size else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.5 * n_rows), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)
    episodes = np.arange(1, len(returns) + 1, dtype=np.int64)

    axes[0].plot(episodes, returns, lw=0.7, alpha=0.30, color="tab:blue")
    axes[0].plot(episodes, _moving_avg(returns, 100), lw=1.8, color="tab:red")
    axes[0].set_ylabel("Return")
    axes[0].set_title(f"{title}: episode return")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episodes, tracking_rmse * 1000.0, lw=0.7, alpha=0.30, color="tab:green")
    axes[1].plot(episodes, _moving_avg(tracking_rmse * 1000.0, 100), lw=1.8, color="tab:olive")
    axes[1].set_ylabel("Track [mm]")
    axes[1].set_title(f"{title}: tracking RMSE")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(episodes, transparency_rmse, lw=0.7, alpha=0.30, color="tab:purple")
    axes[2].plot(episodes, _moving_avg(transparency_rmse, 100), lw=1.8, color="tab:pink")
    axes[2].set_ylabel("Transp [W]")
    axes[2].set_title(f"{title}: transparency RMSE")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Episode")

    if n_rows == 4 and losses is not None:
        axes[3].plot(episodes, losses, lw=0.7, alpha=0.35, color="tab:orange")
        axes[3].plot(episodes, _moving_avg(losses, 100), lw=1.8, color="tab:brown")
        axes[3].set_ylabel("Loss")
        axes[3].set_title(f"{title}: mean TD loss")
        axes[3].set_xlabel("Episode")
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_rollout_dashboard(history: dict[str, Any], out_path: str, title: str, env_switch_time: float) -> None:
    t = _history_array(history, "time", dtype=np.float64)
    if t.size == 0:
        return

    x_m = _history_array(history, "x_m", dtype=np.float64) * 1000.0
    x_s = _history_array(history, "x_s", dtype=np.float64) * 1000.0
    f_h = _history_array(history, "F_h", dtype=np.float64)
    f_e = _history_array(history, "F_e", dtype=np.float64)
    pos_error = _history_array(history, "pos_error", dtype=np.float64) * 1000.0
    transparency_error = _history_array(history, "transparency_error", dtype=np.float64)
    u_v = _history_array(history, "u_v", dtype=np.float64)
    reward = _history_array(history, "reward", dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

    axes[0].plot(t, x_m, lw=1.7, color="tab:blue", label="Master")
    axes[0].plot(t, x_s, lw=1.7, color="tab:orange", label="Slave")
    axes[0].set_ylabel("Pos [mm]")
    axes[0].set_title(f"{title}: rollout dashboard")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, f_h, lw=1.6, color="tab:green", label="F_h")
    axes[1].plot(t, f_e, lw=1.6, color="tab:red", label="F_e")
    axes[1].set_ylabel("Force [N]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, pos_error, lw=1.5, color="tab:purple", label="Tracking error [mm]")
    twin = axes[2].twinx()
    twin.plot(t, transparency_error, lw=1.2, color="tab:brown", alpha=0.85, label="Transparency error [W]")
    axes[2].set_ylabel("Track [mm]")
    twin.set_ylabel("Transp [W]")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, u_v, lw=1.5, color="tab:cyan", label="u_v")
    twin2 = axes[3].twinx()
    twin2.plot(t, reward, lw=1.2, color="tab:gray", alpha=0.85, label="Reward")
    axes[3].set_ylabel("u_v [V]")
    twin2.set_ylabel("Reward")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)

    for ax in axes:
        ax.axvline(env_switch_time, color="gray", lw=1.0, ls="--", alpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_action_usage(history: dict[str, Any], out_path: str, title: str) -> None:
    actions = _history_array(history, "u_v", dtype=np.float64)
    labels = _history_array(history, "env_label", dtype=object)
    if actions.size == 0:
        return

    envs = [("skin", "tab:blue"), ("fat", "tab:orange")]
    x = np.arange(cfg.N_ACTIONS)
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, (env_name, color) in enumerate(envs):
        mask = labels == env_name
        counts = np.array([(actions[mask] == level).sum() for level in cfg.V_LEVELS], dtype=np.float64)
        total = max(1.0, counts.sum())
        ax.bar(x + ((idx - 0.5) * width), counts / total, width=width, color=color, alpha=0.85, label=env_name)

    ax.set_xticks(x, [f"{v:.0f}" for v in cfg.V_LEVELS])
    ax.set_xlabel("Voltage action [V]")
    ax.set_ylabel("Action frequency")
    ax.set_title(f"{title}: action usage by environment")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_state_trajectory(history: dict[str, Any], out_path: str, title: str) -> None:
    tracking_error = _history_array(history, "pos_error", dtype=np.float64)
    v_m = _history_array(history, "v_m", dtype=np.float64)
    v_s = _history_array(history, "v_s", dtype=np.float64)
    t = _history_array(history, "time", dtype=np.float64)
    if tracking_error.size == 0 or v_m.size == 0 or v_s.size == 0:
        return

    velocity_error = v_m - v_s
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.plot(tracking_error * 1000.0, velocity_error, lw=0.9, alpha=0.35, color="tab:gray")
    sc = ax.scatter(
        tracking_error * 1000.0,
        velocity_error,
        c=t,
        s=16,
        cmap="viridis",
        alpha=0.90,
        edgecolor="none",
    )
    ax.scatter([tracking_error[0] * 1000.0], [velocity_error[0]], color="black", s=45, label="Start")
    ax.scatter([tracking_error[-1] * 1000.0], [velocity_error[-1]], color="tab:red", s=45, label="End")
    ax.set_xlabel("Tracking error [mm]")
    ax.set_ylabel("Velocity error [m/s]")
    ax.set_title(f"{title}: trajectory in state space")
    ax.grid(True, alpha=0.3)
    ax.legend()
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Time [s]")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.float64).reshape(-1)
    if edges.size == 0:
        return np.zeros(1, dtype=np.float64)
    if edges.size == 1:
        width = 1.0
        return np.array([edges[0] - width, edges[0], edges[0] + width], dtype=np.float64)
    mids = 0.5 * (edges[:-1] + edges[1:])
    left = edges[0] - 0.5 * (edges[1] - edges[0])
    right = edges[-1] + 0.5 * (edges[-1] - edges[-2])
    return np.concatenate(([left], mids, [right]))


def _q_gap(q_values: np.ndarray) -> float:
    if q_values.size <= 1:
        return float(q_values[0]) if q_values.size else 0.0
    sorted_q = np.sort(np.asarray(q_values, dtype=np.float64))
    return float(sorted_q[-1] - sorted_q[-2])


def _representative_indices(size: int) -> list[int]:
    if size <= 1:
        return [0]
    picks = {1, size // 2, max(0, size - 2)}
    return sorted(int(np.clip(idx, 0, size - 1)) for idx in picks)


def _plot_qlearning_policy_maps(agent: QLearningAgent, out_path: str) -> None:
    te_centers = _bin_centers(cfg.REDUCED_TRACKING_ERROR_BINS) * 1000.0
    ve_centers = _bin_centers(cfg.REDUCED_VELOCITY_ERROR_BINS)
    sp_centers = _bin_centers(cfg.REDUCED_SLAVE_PRESSURE_DIFF_BINS) / 1000.0
    mp_centers = _bin_centers(cfg.REDUCED_MASTER_PRESSURE_DIFF_BINS) / 1000.0

    te_dim, ve_dim, sp_dim, mp_dim = agent.state_dims
    slice_indices = _representative_indices(min(sp_dim, mp_dim))
    zero_visits = np.zeros(agent.n_actions, dtype=np.int64)

    fig, axes = plt.subplots(len(slice_indices), 3, figsize=(16, 4.8 * len(slice_indices)))
    if len(slice_indices) == 1:
        axes = np.asarray([axes], dtype=object)

    for row_idx, slice_idx in enumerate(slice_indices):
        action_map = np.zeros((ve_dim, te_dim), dtype=np.float64)
        visit_map = np.zeros((ve_dim, te_dim), dtype=np.float64)
        gap_map = np.zeros((ve_dim, te_dim), dtype=np.float64)
        for te_idx in range(te_dim):
            for ve_idx in range(ve_dim):
                state = (te_idx, ve_idx, slice_idx, slice_idx)
                q_values = agent.q_values(state)
                visits = agent.visit_count.get(state, zero_visits)
                action_map[ve_idx, te_idx] = cfg.V_LEVELS[_greedy_q_action(q_values)]
                visit_map[ve_idx, te_idx] = float(np.sum(visits))
                gap_map[ve_idx, te_idx] = _q_gap(q_values)

        extent = [te_centers[0], te_centers[-1], ve_centers[0], ve_centers[-1]]
        title_suffix = (
            f"dp_s={sp_centers[slice_idx]:.1f} kPa, dp_m={mp_centers[slice_idx]:.1f} kPa"
        )

        im0 = axes[row_idx, 0].imshow(action_map, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
        axes[row_idx, 0].set_title(f"Action map | {title_suffix}")
        axes[row_idx, 0].set_ylabel("Velocity error [m/s]")
        plt.colorbar(im0, ax=axes[row_idx, 0], label="Voltage [V]")

        im1 = axes[row_idx, 1].imshow(np.log1p(visit_map), origin="lower", aspect="auto", extent=extent, cmap="magma")
        axes[row_idx, 1].set_title(f"Visitation trust | {title_suffix}")
        plt.colorbar(im1, ax=axes[row_idx, 1], label="log(1 + visits)")

        im2 = axes[row_idx, 2].imshow(gap_map, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        axes[row_idx, 2].set_title(f"Q-gap confidence | {title_suffix}")
        plt.colorbar(im2, ax=axes[row_idx, 2], label="Q gap")

        for col in range(3):
            axes[row_idx, col].set_xlabel("Tracking error [mm]")
            axes[row_idx, col].grid(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _default_context_obs() -> np.ndarray:
    return np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            cfg.P_ATM / cfg.OBS_SCALE_PRESSURE,
            cfg.P_ATM / cfg.OBS_SCALE_PRESSURE,
            cfg.P_ATM / cfg.OBS_SCALE_PRESSURE,
            cfg.P_ATM / cfg.OBS_SCALE_PRESSURE,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _dqn_slice_templates(history: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    obs = _history_array(history, "obs", dtype=np.float32)
    labels = _history_array(history, "env_label", dtype=object)
    if obs.size == 0:
        return [("overall", _default_context_obs())]

    if obs.ndim == 1:
        obs = obs.reshape(1, -1)

    templates: list[tuple[str, np.ndarray]] = []
    for label in ("skin", "fat"):
        mask = labels == label
        if np.any(mask):
            templates.append((label, np.median(obs[mask], axis=0).astype(np.float32)))
    templates.append(("overall", np.median(obs, axis=0).astype(np.float32)))
    return templates


def _plot_dqn_policy_slices(agent: DQNAgent, history: dict[str, Any], out_path: str) -> None:
    te_centers = _bin_centers(cfg.REDUCED_TRACKING_ERROR_BINS) * 1000.0
    ve_centers = _bin_centers(cfg.REDUCED_VELOCITY_ERROR_BINS)
    templates = _dqn_slice_templates(history)

    fig, axes = plt.subplots(len(templates), 3, figsize=(16, 4.8 * len(templates)))
    if len(templates) == 1:
        axes = np.asarray([axes], dtype=object)

    for row_idx, (label, context_obs) in enumerate(templates):
        action_map = np.zeros((ve_centers.size, te_centers.size), dtype=np.float64)
        maxq_map = np.zeros_like(action_map)
        gap_map = np.zeros_like(action_map)

        master_pos = float(context_obs[1]) * cfg.OBS_SCALE_POS
        master_vel = float(context_obs[3]) * cfg.OBS_SCALE_VEL

        for te_idx, track_mm in enumerate(te_centers):
            track_error = track_mm / 1000.0
            slave_pos = master_pos - track_error
            for ve_idx, vel_error in enumerate(ve_centers):
                obs_slice = np.array(context_obs, dtype=np.float32, copy=True)
                obs_slice[0] = np.clip(slave_pos / cfg.OBS_SCALE_POS, -2.0, 2.0)
                obs_slice[2] = np.clip((master_vel - vel_error) / cfg.OBS_SCALE_VEL, -2.0, 2.0)
                q_values = agent.q_values(obs_slice)
                action_map[ve_idx, te_idx] = cfg.V_LEVELS[int(np.argmax(q_values))]
                maxq_map[ve_idx, te_idx] = float(np.max(q_values))
                gap_map[ve_idx, te_idx] = _q_gap(q_values)

        extent = [te_centers[0], te_centers[-1], ve_centers[0], ve_centers[-1]]
        im0 = axes[row_idx, 0].imshow(action_map, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
        axes[row_idx, 0].set_title(f"DQN action slice | {label}")
        plt.colorbar(im0, ax=axes[row_idx, 0], label="Voltage [V]")

        im1 = axes[row_idx, 1].imshow(maxq_map, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        axes[row_idx, 1].set_title(f"Max-Q slice | {label}")
        plt.colorbar(im1, ax=axes[row_idx, 1], label="max Q")

        im2 = axes[row_idx, 2].imshow(gap_map, origin="lower", aspect="auto", extent=extent, cmap="plasma")
        axes[row_idx, 2].set_title(f"Q-gap confidence | {label}")
        plt.colorbar(im2, ax=axes[row_idx, 2], label="Q gap")

        for col in range(3):
            axes[row_idx, col].set_xlabel("Tracking error [mm]")
            axes[row_idx, col].set_ylabel("Velocity error [m/s]")
            axes[row_idx, col].grid(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _history_with_obs(history: dict[str, Any], obs_trace: list[np.ndarray]) -> dict[str, Any]:
    merged = dict(history)
    merged["obs"] = [np.asarray(obs, dtype=np.float32) for obs in obs_trace]
    return merged


def _evaluate_qlearning(
    agent: QLearningAgent,
    env_factory: EnvFactory,
    n_episodes: int,
    seed_offset: int,
) -> tuple[float, float, float, dict[str, Any]]:
    rewards, tracking, transparency = [], [], []
    rep_history: dict[str, Any] = {}

    for ep in range(n_episodes):
        env = env_factory()
        obs, _ = env.reset(seed=seed_offset + ep)
        state = env.discretise_obs_reduced(obs)
        done = False
        obs_trace: list[np.ndarray] = []

        while not done:
            obs_trace.append(np.asarray(obs, dtype=np.float32).copy())
            action = _greedy_q_action(agent.q_values(state))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = env.discretise_obs_reduced(obs)

        history = _history_with_obs(env.render() or {}, obs_trace)
        reward, track_rmse, transp_rmse = _compute_eval_metrics(history)
        rewards.append(reward)
        tracking.append(track_rmse)
        transparency.append(transp_rmse)
        if ep == 0:
            rep_history = history

    return float(np.mean(rewards)), float(np.mean(tracking)), float(np.mean(transparency)), rep_history


def _evaluate_dqn(
    agent: DQNAgent,
    env_factory: EnvFactory,
    n_episodes: int,
    seed_offset: int,
) -> tuple[float, float, float, dict[str, Any]]:
    rewards, tracking, transparency = [], [], []
    rep_history: dict[str, Any] = {}

    old_eps = float(agent.epsilon)
    agent.epsilon = 0.0
    try:
        for ep in range(n_episodes):
            env = env_factory()
            obs, _ = env.reset(seed=seed_offset + ep)
            done = False
            obs_trace: list[np.ndarray] = []

            while not done:
                obs_trace.append(np.asarray(obs, dtype=np.float32).copy())
                action = agent.select_action(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            history = _history_with_obs(env.render() or {}, obs_trace)
            reward, track_rmse, transp_rmse = _compute_eval_metrics(history)
            rewards.append(reward)
            tracking.append(track_rmse)
            transparency.append(transp_rmse)
            if ep == 0:
                rep_history = history
    finally:
        agent.epsilon = old_eps

    return float(np.mean(rewards)), float(np.mean(tracking)), float(np.mean(transparency)), rep_history


def _evaluate_mrac(
    env_factory: EnvFactory,
    n_episodes: int,
    seed_offset: int,
) -> tuple[float, float, float, dict[str, Any]]:
    rewards, tracking, transparency = [], [], []
    rep_history: dict[str, Any] = {}

    for ep in range(n_episodes):
        env = env_factory()
        ctrl = FilteredMRACController()
        ctrl.reset()
        obs, info = env.reset(seed=seed_offset + ep)
        done = False
        obs_trace: list[np.ndarray] = []

        while not done:
            obs_trace.append(np.asarray(obs, dtype=np.float32).copy())
            y, u_c = baseline_mrac_inputs(info)
            u_v = ctrl.step_voltage(pos_error=y, u_c=u_c)
            obs, _, terminated, truncated, info = env.step_voltage(u_v)
            done = terminated or truncated

        history = _history_with_obs(env.render() or {}, obs_trace)
        reward, track_rmse, transp_rmse = _compute_eval_metrics(history)
        rewards.append(reward)
        tracking.append(track_rmse)
        transparency.append(transp_rmse)
        if ep == 0:
            rep_history = history

    return float(np.mean(rewards)), float(np.mean(tracking)), float(np.mean(transparency)), rep_history


def _save_common_visuals(history: dict[str, Any], plots_dir: str, title: str, env_switch_time: float) -> None:
    _plot_rollout_dashboard(
        history,
        os.path.join(plots_dir, "rollout_dashboard.png"),
        title,
        env_switch_time=env_switch_time,
    )
    _plot_action_usage(history, os.path.join(plots_dir, "action_usage_by_context.png"), title)
    _plot_state_trajectory(history, os.path.join(plots_dir, "state_trajectory.png"), title)


def _write_run_summary(
    dirs: dict[str, str],
    result: RunResult,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "label": result.label,
        "family": result.family,
        "mean_reward": result.mean_reward,
        "tracking_rmse_m": result.tracking_rmse_m,
        "transparency_rmse_w": result.transparency_rmse_w,
        "out_dir": result.out_dir,
        "tensorboard_dir": result.tensorboard_dir,
        "model_path": result.model_path,
    }
    if extra:
        payload.update(extra)
    _save_json(os.path.join(dirs["logs"], "summary.json"), payload)

    with open(os.path.join(dirs["logs"], "summary.txt"), "w", encoding="utf-8") as fh:
        for key, value in payload.items():
            fh.write(f"{key}={value}\n")


def _train_qlearning_study(
    study_root: str,
    env_mode: str,
    master_input_mode: str,
    total_episodes: int,
    test_episodes: int,
    seed: int,
) -> RunResult:
    writer_cls = _require_tensorboard()
    dirs = _mk_dirs(os.path.join(study_root, "ql"))
    writer = writer_cls(log_dir=dirs["tensorboard"])

    env_factory = lambda: TeleopEnv(env_mode=env_mode, master_input_mode=master_input_mode)
    train_env = env_factory()
    state_dims = train_env.get_state_dims_reduced()
    agent = QLearningAgent(state_dims=state_dims, n_actions=cfg.N_ACTIONS, seed=seed)

    ep_returns = np.zeros(total_episodes, dtype=np.float64)
    ep_track = np.zeros(total_episodes, dtype=np.float64)
    ep_transp = np.zeros(total_episodes, dtype=np.float64)

    for ep in range(total_episodes):
        obs, _ = train_env.reset(seed=seed + ep)
        state = train_env.discretise_obs_reduced(obs)
        done = False
        ep_return = 0.0

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            next_state = train_env.discretise_obs_reduced(next_obs)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_return += reward

        agent.decay_epsilon()
        history = train_env.render() or {}
        _, track_rmse, transp_rmse = _compute_eval_metrics(history)
        ep_returns[ep] = ep_return
        ep_track[ep] = track_rmse
        ep_transp[ep] = transp_rmse

        step = ep + 1
        writer.add_scalar("train/episode_return", ep_return, step)
        writer.add_scalar("train/tracking_rmse_m", track_rmse, step)
        writer.add_scalar("train/transparency_rmse_w", transp_rmse, step)
        writer.add_scalar("train/epsilon", agent.epsilon, step)
        writer.add_scalar("train/discovered_states", agent.discovered_states(), step)
        writer.add_scalar("train/action_coverage", agent.coverage(), step)

        if step == 1 or step % 1000 == 0 or step == total_episodes:
            start = max(0, ep + 1 - min(100, ep + 1))
            print(
                f"[Q-learning] ep {step}/{total_episodes} | "
                f"avgR(100) {np.mean(ep_returns[start:ep+1]):+.2f} | "
                f"track {np.mean(ep_track[start:ep+1]) * 1000.0:.2f} mm | "
                f"transp {np.mean(ep_transp[start:ep+1]):.4f} W | "
                f"eps {agent.epsilon:.4f} | states {agent.discovered_states()}"
            )

        if step == 1 or step % cfg.EVAL_EVERY == 0 or step == total_episodes:
            eval_reward, eval_track, eval_transp, _ = _evaluate_qlearning(
                agent,
                env_factory,
                n_episodes=max(1, min(cfg.EVAL_EPISODES, test_episodes)),
                seed_offset=10_000 + step,
            )
            writer.add_scalar("eval/mean_reward", eval_reward, step)
            writer.add_scalar("eval/tracking_rmse_m", eval_track, step)
            writer.add_scalar("eval/transparency_rmse_w", eval_transp, step)

    writer.flush()
    writer.close()

    agent_path = os.path.join(dirs["models"], "q_table.npy")
    agent.save(agent_path)
    np.savez(
        os.path.join(dirs["logs"], "training_metrics.npz"),
        episode_returns=ep_returns,
        episode_tracking_rmse=ep_track,
        episode_transparency_rmse=ep_transp,
    )
    _save_training_plot(
        ep_returns,
        ep_track,
        ep_transp,
        os.path.join(dirs["plots"], "training_metrics.png"),
        "Q-learning",
    )

    mean_reward, tracking_rmse, transparency_rmse, history = _evaluate_qlearning(
        agent,
        env_factory,
        n_episodes=test_episodes,
        seed_offset=20_000,
    )
    _save_history_npz(history, os.path.join(dirs["episodes"], "test_episode.npz"))
    _save_common_visuals(history, dirs["plots"], "Q-learning", env_switch_time=train_env.env_switch_time)
    _plot_qlearning_policy_maps(agent, os.path.join(dirs["plots"], "policy_maps.png"))

    result = RunResult(
        label="q_learning",
        family="q_learning",
        mean_reward=mean_reward,
        tracking_rmse_m=tracking_rmse,
        transparency_rmse_w=transparency_rmse,
        history=history,
        out_dir=dirs["base"],
        tensorboard_dir=dirs["tensorboard"],
        model_path=agent_path,
    )
    _write_run_summary(
        dirs,
        result,
        extra={
            "env_mode": env_mode,
            "master_input_mode": master_input_mode,
            "total_episodes": total_episodes,
            "test_episodes": test_episodes,
            "state_dims": state_dims,
        },
    )
    return result


def _train_dqn_variant(
    study_root: str,
    env_mode: str,
    master_input_mode: str,
    total_episodes: int,
    test_episodes: int,
    seed: int,
    variant: RewardVariant,
    variant_index: int,
) -> RunResult:
    writer_cls = _require_tensorboard()
    dirs = _mk_dirs(os.path.join(study_root, "dqn", f"v{variant_index:02d}"))
    writer = writer_cls(log_dir=dirs["tensorboard"])

    env_factory = lambda: RewardAblationEnv(
        TeleopEnv(env_mode=env_mode, master_input_mode=master_input_mode),
        variant,
    )
    train_env = env_factory()
    agent = DQNAgent(obs_dim=10, n_actions=cfg.N_ACTIONS, seed=seed)

    ep_returns = np.zeros(total_episodes, dtype=np.float64)
    ep_track = np.zeros(total_episodes, dtype=np.float64)
    ep_transp = np.zeros(total_episodes, dtype=np.float64)
    ep_loss = np.full(total_episodes, np.nan, dtype=np.float64)

    for ep in range(total_episodes):
        obs, _ = train_env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        losses: list[float] = []

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(float(loss))
            obs = next_obs
            ep_return += reward

        agent.decay_epsilon()
        history = train_env.render() or {}
        _, track_rmse, transp_rmse = _compute_eval_metrics(history)
        ep_returns[ep] = ep_return
        ep_track[ep] = track_rmse
        ep_transp[ep] = transp_rmse
        if losses:
            ep_loss[ep] = float(np.mean(losses))

        step = ep + 1
        writer.add_scalar("train/episode_return", ep_return, step)
        writer.add_scalar("train/tracking_rmse_m", track_rmse, step)
        writer.add_scalar("train/transparency_rmse_w", transp_rmse, step)
        writer.add_scalar("train/epsilon", agent.epsilon, step)
        writer.add_scalar("train/replay_size", len(agent.replay_buffer), step)
        writer.add_scalar("train/grad_steps", agent.train_step_count, step)
        if losses:
            writer.add_scalar("train/mean_td_loss", float(np.mean(losses)), step)

        if step == 1 or step % 1000 == 0 or step == total_episodes:
            start = max(0, ep + 1 - min(100, ep + 1))
            mean_loss = float(np.nanmean(ep_loss[start:ep+1])) if np.any(~np.isnan(ep_loss[start:ep+1])) else float("nan")
            print(
                f"[{variant.name}] ep {step}/{total_episodes} | "
                f"avgR(100) {np.mean(ep_returns[start:ep+1]):+.2f} | "
                f"track {np.mean(ep_track[start:ep+1]) * 1000.0:.2f} mm | "
                f"transp {np.mean(ep_transp[start:ep+1]):.4f} W | "
                f"eps {agent.epsilon:.4f} | buf {len(agent.replay_buffer)} | "
                f"grad {agent.train_step_count} | loss {mean_loss:.5f}"
            )

        if step == 1 or step % cfg.DQN_EVAL_EVERY == 0 or step == total_episodes:
            eval_reward, eval_track, eval_transp, _ = _evaluate_dqn(
                agent,
                env_factory,
                n_episodes=max(1, min(cfg.DQN_EVAL_EPISODES, test_episodes)),
                seed_offset=30_000 + step,
            )
            writer.add_scalar("eval/mean_reward", eval_reward, step)
            writer.add_scalar("eval/tracking_rmse_m", eval_track, step)
            writer.add_scalar("eval/transparency_rmse_w", eval_transp, step)

    writer.flush()
    writer.close()

    agent_path = os.path.join(dirs["models"], "dqn_model.pt")
    agent.save(agent_path)
    np.savez(
        os.path.join(dirs["logs"], "training_metrics.npz"),
        episode_returns=ep_returns,
        episode_tracking_rmse=ep_track,
        episode_transparency_rmse=ep_transp,
        episode_loss=ep_loss,
    )
    _save_training_plot(
        ep_returns,
        ep_track,
        ep_transp,
        os.path.join(dirs["plots"], "training_metrics.png"),
        variant.name,
        losses=ep_loss,
    )

    mean_reward, tracking_rmse, transparency_rmse, history = _evaluate_dqn(
        agent,
        env_factory,
        n_episodes=test_episodes,
        seed_offset=40_000,
    )
    _save_history_npz(history, os.path.join(dirs["episodes"], "test_episode.npz"))
    _save_common_visuals(history, dirs["plots"], variant.name, env_switch_time=train_env.env_switch_time)
    _plot_dqn_policy_slices(agent, history, os.path.join(dirs["plots"], "policy_slices.png"))

    result = RunResult(
        label=variant.name,
        family="dqn",
        mean_reward=mean_reward,
        tracking_rmse_m=tracking_rmse,
        transparency_rmse_w=transparency_rmse,
        history=history,
        out_dir=dirs["base"],
        tensorboard_dir=dirs["tensorboard"],
        model_path=agent_path,
    )
    _write_run_summary(
        dirs,
        result,
        extra={
            "env_mode": env_mode,
            "master_input_mode": master_input_mode,
            "total_episodes": total_episodes,
            "test_episodes": test_episodes,
            "reward_variant": asdict(variant),
        },
    )
    return result


def _run_result_payload(result: RunResult) -> dict[str, Any]:
    return {
        "label": result.label,
        "family": result.family,
        "mean_reward": float(result.mean_reward),
        "tracking_rmse_m": float(result.tracking_rmse_m),
        "transparency_rmse_w": float(result.transparency_rmse_w),
        "out_dir": str(result.out_dir),
        "tensorboard_dir": str(result.tensorboard_dir),
        "model_path": str(result.model_path),
    }


def _run_result_from_payload(payload: dict[str, Any]) -> RunResult:
    return RunResult(
        label=str(payload["label"]),
        family=str(payload["family"]),
        mean_reward=float(payload["mean_reward"]),
        tracking_rmse_m=float(payload["tracking_rmse_m"]),
        transparency_rmse_w=float(payload["transparency_rmse_w"]),
        history={},
        out_dir=str(payload["out_dir"]),
        tensorboard_dir=str(payload["tensorboard_dir"]),
        model_path=str(payload["model_path"]),
    )


def _train_dqn_variant_task(task: dict[str, Any]) -> dict[str, Any]:
    variant = _reward_variant_from_dict(dict(task["variant"]))
    result = _train_dqn_variant(
        study_root=str(task["study_root"]),
        env_mode=str(task["env_mode"]),
        master_input_mode=str(task["master_input_mode"]),
        total_episodes=int(task["dqn_episodes"]),
        test_episodes=int(task["test_episodes"]),
        seed=int(task["seed"]),
        variant=variant,
        variant_index=int(task["variant_index"]),
    )
    return {
        "order": int(task["order"]),
        "variant_name": variant.name,
        "result": _run_result_payload(result),
    }


def _run_mrac_baseline(
    study_root: str,
    env_mode: str,
    master_input_mode: str,
    test_episodes: int,
    seed: int,
) -> RunResult:
    writer_cls = _require_tensorboard()
    dirs = _mk_dirs(os.path.join(study_root, "mr"))
    writer = writer_cls(log_dir=dirs["tensorboard"])

    env_factory = lambda: TeleopEnv(env_mode=env_mode, master_input_mode=master_input_mode)
    mean_reward, tracking_rmse, transparency_rmse, history = _evaluate_mrac(
        env_factory,
        n_episodes=test_episodes,
        seed_offset=50_000 + seed,
    )

    writer.add_scalar("eval/mean_reward", mean_reward, 1)
    writer.add_scalar("eval/tracking_rmse_m", tracking_rmse, 1)
    writer.add_scalar("eval/transparency_rmse_w", transparency_rmse, 1)
    writer.flush()
    writer.close()

    _save_history_npz(history, os.path.join(dirs["episodes"], "test_episode.npz"))
    _save_common_visuals(history, dirs["plots"], "MRAC", env_switch_time=cfg.ENV_SWITCH_TIME)

    result = RunResult(
        label="mrac",
        family="mrac",
        mean_reward=mean_reward,
        tracking_rmse_m=tracking_rmse,
        transparency_rmse_w=transparency_rmse,
        history=history,
        out_dir=dirs["base"],
        tensorboard_dir=dirs["tensorboard"],
        model_path="",
    )
    _write_run_summary(
        dirs,
        result,
        extra={
            "env_mode": env_mode,
            "master_input_mode": master_input_mode,
            "test_episodes": test_episodes,
        },
    )
    return result


def _write_master_summary(study_root: str, results: list[RunResult]) -> str:
    out_path = os.path.join(study_root, "study_summary.csv")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(
            "label,family,mean_reward,tracking_rmse_mm,transparency_rmse_w,"
            "tensorboard_dir,model_path,out_dir\n"
        )
        for result in results:
            fh.write(
                f"{result.label},{result.family},{result.mean_reward:.6f},"
                f"{result.tracking_rmse_m * 1000.0:.6f},{result.transparency_rmse_w:.6f},"
                f"{result.tensorboard_dir},{result.model_path},{result.out_dir}\n"
            )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the long study: Q-learning, MRAC, and DQN reward ablations."
    )
    parser.add_argument(
        "--env-mode",
        choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING],
        default=cfg.ENV_MODE_CHANGING,
    )
    parser.add_argument(
        "--master-input-mode",
        choices=[cfg.MASTER_INPUT_REFERENCE, cfg.MASTER_INPUT_FORCE],
        default=cfg.DEFAULT_MASTER_INPUT_MODE,
    )
    parser.add_argument("--q-episodes", type=int, default=cfg.NUM_EPISODES)
    parser.add_argument("--dqn-episodes", type=int, default=cfg.DQN_NUM_EPISODES)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument(
        "--dqn-only",
        action="store_true",
        help="Run only the DQN reward-function study and skip Q-learning/MRAC.",
    )
    parser.add_argument(
        "--max-dqn-variants",
        type=int,
        default=10,
        help="Limit the number of DQN reward variants, useful for smoke tests.",
    )
    parser.add_argument(
        "--variant-name",
        action="append",
        default=None,
        help="Train only the named DQN reward variant. Repeat to select multiple variants.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Maximum number of parallel DQN worker processes.",
    )
    parser.add_argument(
        "--worker-torch-threads",
        type=int,
        default=1,
        help="Torch CPU threads per DQN worker process.",
    )
    args = parser.parse_args()

    study_name = args.study_name
    if args.dqn_only and study_name is None:
        study_name = "reward_function_study_dqn"

    study_root = _study_root(study_name)
    tb_study_root = _tensorboard_study_root(study_root)
    all_reward_variants = _build_reward_variants()
    reward_variants = all_reward_variants
    requested_variant_names = list(args.variant_name or [])
    if requested_variant_names:
        requested_set = set(requested_variant_names)
        known_names = {variant.name for variant in all_reward_variants}
        missing = [name for name in requested_variant_names if name not in known_names]
        if missing:
            raise ValueError(f"Unknown reward variant name(s): {', '.join(missing)}")
        reward_variants = [variant for variant in all_reward_variants if variant.name in requested_set]
    elif args.max_dqn_variants > 0:
        reward_variants = all_reward_variants[: args.max_dqn_variants]

    detected_cpus = os.cpu_count() or 1
    requested_workers = max(1, int(args.parallel_workers))
    parallel_workers = max(1, min(requested_workers, len(reward_variants))) if reward_variants else 1
    worker_torch_threads = _configure_process_env(int(args.worker_torch_threads))

    print(f"Study root: {study_root}")
    print(
        "Running long study with "
        f"env_mode={args.env_mode}, master_input_mode={args.master_input_mode}, "
        f"q_episodes={args.q_episodes}, dqn_episodes={args.dqn_episodes}, "
        f"test_episodes={args.test_episodes}, dqn_variants={len(reward_variants)}, "
        f"parallel_workers={parallel_workers}, worker_torch_threads={worker_torch_threads}"
    )

    manifest = {
        "study_root": study_root,
        "env_mode": args.env_mode,
        "master_input_mode": args.master_input_mode,
        "q_episodes": args.q_episodes,
        "dqn_episodes": args.dqn_episodes,
        "test_episodes": args.test_episodes,
        "seed": args.seed,
        "tensorboard_root": tb_study_root,
        "dqn_only": args.dqn_only,
        "variant_filter": requested_variant_names,
        "parallel_workers_requested": requested_workers,
        "parallel_workers_used": parallel_workers,
        "worker_torch_threads": worker_torch_threads,
        "reward_variants": [asdict(variant) for variant in reward_variants],
    }
    _save_json(os.path.join(study_root, "study_manifest.json"), manifest)

    q_result: RunResult | None = None
    mrac_result: RunResult | None = None

    if not args.dqn_only:
        print("1/3 Training Q-learning baseline...")
        q_result = _train_qlearning_study(
            study_root=study_root,
            env_mode=args.env_mode,
            master_input_mode=args.master_input_mode,
            total_episodes=args.q_episodes,
            test_episodes=args.test_episodes,
            seed=args.seed,
        )
        _write_master_summary(study_root, [q_result])

        print("2/3 Evaluating MRAC baseline...")
        mrac_result = _run_mrac_baseline(
            study_root=study_root,
            env_mode=args.env_mode,
            master_input_mode=args.master_input_mode,
            test_episodes=args.test_episodes,
            seed=args.seed,
        )
        _write_master_summary(study_root, [q_result, mrac_result])

        print("3/3 Running DQN reward ablations...")
    else:
        print("Running DQN-only reward-function study...")

    dqn_results: list[RunResult] = []
    if parallel_workers <= 1:
        _worker_runtime_init(worker_torch_threads)
        for idx, variant in enumerate(reward_variants, start=1):
            print(f"  Variant {idx}/{len(reward_variants)}: {variant.name}")
            result = _train_dqn_variant(
                study_root=study_root,
                env_mode=args.env_mode,
                master_input_mode=args.master_input_mode,
                total_episodes=args.dqn_episodes,
                test_episodes=args.test_episodes,
                seed=args.seed + idx,
                variant=variant,
                variant_index=idx,
            )
            dqn_results.append(result)
            partial_results = [
                *([q_result] if q_result is not None else []),
                *([mrac_result] if mrac_result is not None else []),
                *dqn_results,
            ]
            _write_master_summary(study_root, partial_results)
    else:
        print(
            f"Launching {len(reward_variants)} DQN reward variants with up to "
            f"{parallel_workers} worker processes "
            f"(detected CPUs: {detected_cpus}, per-worker torch threads: {worker_torch_threads})."
        )
        tasks = [
            {
                "order": idx,
                "study_root": study_root,
                "env_mode": args.env_mode,
                "master_input_mode": args.master_input_mode,
                "dqn_episodes": args.dqn_episodes,
                "test_episodes": args.test_episodes,
                "seed": args.seed + idx,
                "variant": asdict(variant),
                "variant_index": idx,
            }
            for idx, variant in enumerate(reward_variants, start=1)
        ]
        completed_results: dict[int, RunResult] = {}
        with cf.ProcessPoolExecutor(
            max_workers=parallel_workers,
            initializer=_worker_runtime_init,
            initargs=(worker_torch_threads,),
        ) as executor:
            future_map = {
                executor.submit(_train_dqn_variant_task, task): (
                    int(task["order"]),
                    str(task["variant"]["name"]),
                )
                for task in tasks
            }
            for future in cf.as_completed(future_map):
                order, variant_name = future_map[future]
                payload = future.result()
                completed_results[order] = _run_result_from_payload(dict(payload["result"]))
                print(
                    f"[done {order}/{len(reward_variants)}] {variant_name} -> "
                    f"{completed_results[order].out_dir}"
                )
                dqn_results = [completed_results[idx] for idx in sorted(completed_results)]
                partial_results = [
                    *([q_result] if q_result is not None else []),
                    *([mrac_result] if mrac_result is not None else []),
                    *dqn_results,
                ]
                _write_master_summary(study_root, partial_results)
        dqn_results = [completed_results[idx] for idx in sorted(completed_results)]

    all_results = [
        *([q_result] if q_result is not None else []),
        *([mrac_result] if mrac_result is not None else []),
        *dqn_results,
    ]
    summary_csv = _write_master_summary(study_root, all_results)
    with open(os.path.join(study_root, "tensorboard_command.txt"), "w", encoding="utf-8") as fh:
        fh.write(f'tensorboard --logdir "{tb_study_root}"\n')

    print("\nStudy complete.")
    print(f"Summary CSV: {summary_csv}")
    print(f"TensorBoard command saved to: {os.path.join(study_root, 'tensorboard_command.txt')}")


if __name__ == "__main__":
    main()
