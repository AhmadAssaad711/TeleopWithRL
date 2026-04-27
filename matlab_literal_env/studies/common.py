from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional until runtime
    SummaryWriter = None

from ... import config as cfg
from ...dqn_agent import DQNAgent
from ...q_learning_agent import QLearningAgent
from ..simuoriginal_replica import FE_MODE_GUI


DEFAULT_REPLICA_RESULTS_ROOT = "matlab_literal_env/results/studies"
FE_MODE_DIR_ALIASES = {
    "gui_skin_locked": "gui",
    "switched_dynamics": "dyn",
}


@dataclass
class RunResult:
    label: str
    family: str
    mean_reward: float
    tracking_rmse_m: float
    transparency_rmse_w: float
    pre_switch_tracking_rmse_m: float
    post_switch_tracking_rmse_m: float
    pre_switch_transparency_rmse_w: float
    post_switch_transparency_rmse_w: float
    invalid_episode_rate: float
    history: dict[str, Any]
    out_dir: str
    tensorboard_dir: str
    model_path: str
    reward_variant: str
    state_variant: str


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def results_root(fe_mode: str = FE_MODE_GUI) -> Path:
    fe_dir = FE_MODE_DIR_ALIASES.get(str(fe_mode), str(fe_mode))
    return package_root() / DEFAULT_REPLICA_RESULTS_ROOT / fe_dir


def timestamped_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def require_tensorboard():
    if SummaryWriter is None:
        raise ImportError(
            "TensorBoard logging requires 'tensorboard'. Install it with "
            "'pip install tensorboard' in the active environment."
        )
    return SummaryWriter


def moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x
    width = max(1, min(int(window), x.size))
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(x, kernel, mode="same")


def history_array(history: dict[str, Any], key: str, dtype=np.float64) -> np.ndarray:
    values = history.get(key, [])
    try:
        return np.asarray(values, dtype=dtype)
    except (TypeError, ValueError):
        return np.asarray(values, dtype=object)


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=json_default)


def save_history_npz(history: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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


def q_gap(q_values: np.ndarray) -> float:
    q_values = np.asarray(q_values, dtype=np.float64).reshape(-1)
    if q_values.size == 0:
        return 0.0
    if q_values.size == 1:
        return float(q_values[0])
    sorted_q = np.sort(q_values)
    return float(sorted_q[-1] - sorted_q[-2])


def resolve_action_levels(
    action_levels: list[float] | tuple[float, ...] | np.ndarray | None = None,
    *,
    expected_n_actions: int | None = None,
) -> np.ndarray:
    levels = np.asarray(cfg.V_LEVELS if action_levels is None else action_levels, dtype=np.float64).reshape(-1)
    if levels.size == 0:
        raise ValueError("action_levels must contain at least one voltage level")
    if expected_n_actions is not None and levels.size != int(expected_n_actions):
        raise ValueError(
            f"action_levels has {levels.size} entries, expected {int(expected_n_actions)}"
        )
    return levels.astype(np.float64, copy=True)


def greedy_q_action(q_values: np.ndarray, action_levels: list[float] | tuple[float, ...] | np.ndarray | None = None) -> int:
    q_values = np.asarray(q_values, dtype=np.float64).reshape(-1)
    levels = resolve_action_levels(action_levels, expected_n_actions=q_values.size)
    max_q = float(np.max(q_values))
    best = np.flatnonzero(q_values == max_q)
    zero_action = int(np.argmin(np.abs(levels)))
    return zero_action if zero_action in best else int(best[0])


def mk_run_dirs(base_dir: str | Path) -> dict[str, str]:
    base = Path(base_dir)
    rel = base.relative_to(package_root())
    tb_root = Path.home() / "AppData" / "Local" / "TeleopWithRL_tb" / rel
    paths = {
        "base": str(base),
        "models": str(base / "m"),
        "logs": str(base / "l"),
        "plots": str(base / "p"),
        "episodes": str(base / "e"),
        "tensorboard": str(tb_root),
    }
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return paths


def study_root(study_name: str | None, family_name: str) -> Path:
    root = results_root() / family_name / (study_name or timestamped_name(family_name))
    root.mkdir(parents=True, exist_ok=True)
    return root


def history_with_obs(history: dict[str, Any], obs_trace: list[np.ndarray]) -> dict[str, Any]:
    merged = dict(history)
    merged["obs"] = [np.asarray(obs, dtype=np.float32) for obs in obs_trace]
    return merged


def _pad_numeric_series(arrays: list[np.ndarray]) -> np.ndarray:
    max_len = max(arr.shape[0] for arr in arrays)
    tail_shape = arrays[0].shape[1:]
    stack = np.full((len(arrays), max_len, *tail_shape), np.nan, dtype=np.float64)
    for idx, arr in enumerate(arrays):
        cast = np.asarray(arr, dtype=np.float64)
        stack[idx, : cast.shape[0], ...] = cast
    return stack


def _aggregate_numeric_series(arrays: list[np.ndarray]) -> np.ndarray:
    return np.nanmean(_pad_numeric_series(arrays), axis=0)


def _aggregate_object_series(arrays: list[np.ndarray]) -> np.ndarray:
    max_len = max(arr.shape[0] for arr in arrays)
    merged: list[Any] = []
    for t_idx in range(max_len):
        votes: dict[Any, int] = {}
        for arr in arrays:
            if t_idx >= arr.shape[0]:
                continue
            value = arr[t_idx]
            key = value.item() if isinstance(value, np.generic) else value
            votes[key] = votes.get(key, 0) + 1
        if not votes:
            merged.append("")
            continue
        merged.append(max(votes.items(), key=lambda item: (item[1], str(item[0])))[0])
    return np.asarray(merged, dtype=object)


def aggregate_episode_histories(histories: list[dict[str, Any]]) -> dict[str, Any]:
    if not histories:
        return {}

    merged: dict[str, Any] = {
        "aggregated_episodes": int(len(histories)),
    }
    all_keys = sorted({key for history in histories for key in history.keys()})
    for key in all_keys:
        series = [np.asarray(history[key]) for history in histories if key in history]
        if not series:
            continue

        ref = series[0]
        if ref.ndim == 0:
            if np.issubdtype(ref.dtype, np.number) or ref.dtype == np.bool_:
                values = np.asarray([np.asarray(item, dtype=np.float64).item() for item in series], dtype=np.float64)
                merged[key] = float(np.mean(values))
            else:
                merged[key] = ref.item() if isinstance(ref, np.generic) else ref
            continue

        compatible = all(arr.ndim == ref.ndim and arr.shape[1:] == ref.shape[1:] for arr in series)
        if compatible and (np.issubdtype(ref.dtype, np.number) or ref.dtype == np.bool_):
            merged[key] = _aggregate_numeric_series([np.asarray(arr, dtype=np.float64) for arr in series])
            merged[f"{key}_all"] = np.concatenate([np.asarray(arr, dtype=np.float64) for arr in series], axis=0)
            continue

        if compatible:
            object_series = [np.asarray(arr, dtype=object) for arr in series]
            merged[key] = _aggregate_object_series(object_series)
            merged[f"{key}_all"] = np.concatenate(object_series, axis=0)
            continue

        merged[key] = ref

    return merged


def rollout_metrics(history: dict[str, Any], env_switch_time: float) -> dict[str, float]:
    rewards = history_array(history, "reward", dtype=np.float64)
    pos_error = history_array(history, "pos_error", dtype=np.float64)
    transparency_error = history_array(history, "transparency_error", dtype=np.float64)
    time_s = history_array(history, "time", dtype=np.float64)
    invalid = history_array(history, "invalid_state", dtype=np.float64)

    def _rmse(values: np.ndarray, mask: np.ndarray | None = None) -> float:
        if values.size == 0:
            return 0.0
        if mask is not None:
            values = values[mask]
        return float(np.sqrt(np.mean(values ** 2))) if values.size else 0.0

    pre_mask = time_s < float(env_switch_time)
    post_mask = time_s >= float(env_switch_time)
    return {
        "mean_reward": float(np.sum(rewards)) if rewards.size else 0.0,
        "tracking_rmse_m": _rmse(pos_error),
        "transparency_rmse_w": _rmse(transparency_error),
        "pre_switch_tracking_rmse_m": _rmse(pos_error, pre_mask),
        "post_switch_tracking_rmse_m": _rmse(pos_error, post_mask),
        "pre_switch_transparency_rmse_w": _rmse(transparency_error, pre_mask),
        "post_switch_transparency_rmse_w": _rmse(transparency_error, post_mask),
        "invalid_episode": float(np.any(invalid > 0.0)),
    }


def write_run_summary(dirs: dict[str, str], result: RunResult, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "label": result.label,
        "family": result.family,
        "mean_reward": result.mean_reward,
        "tracking_rmse_m": result.tracking_rmse_m,
        "transparency_rmse_w": result.transparency_rmse_w,
        "pre_switch_tracking_rmse_m": result.pre_switch_tracking_rmse_m,
        "post_switch_tracking_rmse_m": result.post_switch_tracking_rmse_m,
        "pre_switch_transparency_rmse_w": result.pre_switch_transparency_rmse_w,
        "post_switch_transparency_rmse_w": result.post_switch_transparency_rmse_w,
        "invalid_episode_rate": result.invalid_episode_rate,
        "out_dir": result.out_dir,
        "tensorboard_dir": result.tensorboard_dir,
        "model_path": result.model_path,
        "reward_variant": result.reward_variant,
        "state_variant": result.state_variant,
    }
    if extra:
        payload.update(extra)
    save_json(Path(dirs["logs"]) / "summary.json", payload)
    with open(Path(dirs["logs"]) / "summary.txt", "w", encoding="utf-8") as fh:
        for key, value in payload.items():
            fh.write(f"{key}={value}\n")
    return payload


def save_training_plot(
    returns: np.ndarray,
    tracking_rmse: np.ndarray,
    transparency_rmse: np.ndarray,
    out_path: str | Path,
    title: str,
    losses: np.ndarray | None = None,
    eval_payload: dict[str, np.ndarray] | None = None,
) -> None:
    n_rows = 4 if losses is not None and losses.size else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.5 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)
    episodes = np.arange(1, len(returns) + 1, dtype=np.int64)

    axes[0].plot(episodes, returns, lw=0.7, alpha=0.30, color="tab:blue")
    axes[0].plot(episodes, moving_avg(returns, 100), lw=1.8, color="tab:red")
    if eval_payload is not None and eval_payload.get("steps") is not None and eval_payload.get("mean_reward") is not None:
        axes[0].plot(
            np.asarray(eval_payload["steps"], dtype=np.float64),
            np.asarray(eval_payload["mean_reward"], dtype=np.float64),
            marker="o",
            lw=1.3,
            ms=3.5,
            color="black",
            label="Eval",
        )
    axes[0].set_ylabel("Return")
    axes[0].set_title(f"{title}: episode return")
    axes[0].grid(True, alpha=0.3)
    if eval_payload is not None and eval_payload.get("steps") is not None and eval_payload.get("mean_reward") is not None:
        axes[0].legend()

    axes[1].plot(episodes, tracking_rmse * 1000.0, lw=0.7, alpha=0.30, color="tab:green")
    axes[1].plot(episodes, moving_avg(tracking_rmse * 1000.0, 100), lw=1.8, color="tab:olive")
    if eval_payload is not None and eval_payload.get("steps") is not None and eval_payload.get("tracking_rmse_m") is not None:
        axes[1].plot(
            np.asarray(eval_payload["steps"], dtype=np.float64),
            np.asarray(eval_payload["tracking_rmse_m"], dtype=np.float64) * 1000.0,
            marker="o",
            lw=1.3,
            ms=3.5,
            color="black",
        )
    axes[1].set_ylabel("Track [mm]")
    axes[1].set_title(f"{title}: tracking RMSE")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(episodes, transparency_rmse, lw=0.7, alpha=0.30, color="tab:purple")
    axes[2].plot(episodes, moving_avg(transparency_rmse, 100), lw=1.8, color="tab:pink")
    if eval_payload is not None and eval_payload.get("steps") is not None and eval_payload.get("transparency_rmse_w") is not None:
        axes[2].plot(
            np.asarray(eval_payload["steps"], dtype=np.float64),
            np.asarray(eval_payload["transparency_rmse_w"], dtype=np.float64),
            marker="o",
            lw=1.3,
            ms=3.5,
            color="black",
        )
    axes[2].set_ylabel("Transp [W]")
    axes[2].set_title(f"{title}: transparency RMSE")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Episode")

    if n_rows == 4 and losses is not None:
        axes[3].plot(episodes, losses, lw=0.7, alpha=0.35, color="tab:orange")
        axes[3].plot(episodes, moving_avg(losses, 100), lw=1.8, color="tab:brown")
        axes[3].set_ylabel("Loss")
        axes[3].set_title(f"{title}: mean TD loss")
        axes[3].set_xlabel("Episode")
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _rollout_title_suffix(history: dict[str, Any]) -> str:
    eval_episodes = int(history.get("evaluation_episodes", history.get("aggregated_episodes", 0)) or 0)
    completed_episodes = int(history.get("completed_episodes", 0) or 0)
    if eval_episodes <= 0:
        return ""
    return f" | mean over {eval_episodes} eval eps | completed {completed_episodes}/{eval_episodes}"


def plot_average_core_rollout(
    history: dict[str, Any],
    out_path: str | Path,
    title: str,
    env_switch_time: float,
) -> None:
    t = history_array(history, "time", dtype=np.float64)
    if t.size == 0:
        return

    x_m = history_array(history, "x_m", dtype=np.float64) * 1000.0
    x_s = history_array(history, "x_s", dtype=np.float64) * 1000.0
    f_h = history_array(history, "F_h", dtype=np.float64)
    f_e = history_array(history, "F_e", dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, x_m, lw=1.8, color="tab:blue", label="x_m")
    axes[0].plot(t, x_s, lw=1.8, color="tab:orange", label="x_s")
    axes[0].set_ylabel("Pos [mm]")
    axes[0].set_title(f"{title}: average rollout{_rollout_title_suffix(history)}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, f_h, lw=1.7, color="tab:green", label="F_h")
    axes[1].plot(t, f_e, lw=1.7, color="tab:red", label="F_e")
    axes[1].set_ylabel("Force [N]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.axvline(env_switch_time, color="gray", lw=1.0, ls="--", alpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_dashboard(history: dict[str, Any], out_path: str | Path, title: str, env_switch_time: float) -> None:
    t = history_array(history, "time", dtype=np.float64)
    if t.size == 0:
        return

    x_m = history_array(history, "x_m", dtype=np.float64) * 1000.0
    x_s = history_array(history, "x_s", dtype=np.float64) * 1000.0
    f_h = history_array(history, "F_h", dtype=np.float64)
    f_e = history_array(history, "F_e", dtype=np.float64)
    pos_error = history_array(history, "pos_error", dtype=np.float64) * 1000.0
    transparency_error = history_array(history, "transparency_error", dtype=np.float64)
    u_v = history_array(history, "u_v", dtype=np.float64)
    reward = history_array(history, "reward", dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    axes[0].plot(t, x_m, lw=1.7, color="tab:blue", label="Master")
    axes[0].plot(t, x_s, lw=1.7, color="tab:orange", label="Slave")
    axes[0].set_ylabel("Pos [mm]")
    axes[0].set_title(f"{title}: rollout dashboard{_rollout_title_suffix(history)}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, f_h, lw=1.6, color="tab:green", label="F_h")
    axes[1].plot(t, f_e, lw=1.6, color="tab:red", label="F_e")
    axes[1].set_ylabel("Force [N]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    track_line = axes[2].plot(t, pos_error, lw=1.5, color="tab:purple", label="Tracking error")[0]
    twin = axes[2].twinx()
    transp_line = twin.plot(
        t,
        transparency_error,
        lw=1.2,
        color="tab:brown",
        alpha=0.85,
        label="Transparency error",
    )[0]
    axes[2].set_ylabel("Track [mm]", color="tab:purple")
    twin.set_ylabel("Transp [W]", color="tab:brown")
    axes[2].tick_params(axis="y", colors="tab:purple")
    twin.tick_params(axis="y", colors="tab:brown")
    axes[2].set_title("Tracking error (purple) and transparency error (brown)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(
        [track_line, transp_line],
        ["Tracking error", "Transparency error"],
        loc="upper left",
        framealpha=0.95,
        facecolor="white",
    )

    action_line = axes[3].plot(t, u_v, lw=1.5, color="tab:cyan", label="Action voltage")[0]
    twin2 = axes[3].twinx()
    reward_line = twin2.plot(t, reward, lw=1.2, color="tab:gray", alpha=0.85, label="Reward")[0]
    axes[3].set_ylabel("u_v [V]", color="tab:cyan")
    twin2.set_ylabel("Reward", color="tab:gray")
    axes[3].tick_params(axis="y", colors="tab:cyan")
    twin2.tick_params(axis="y", colors="tab:gray")
    axes[3].set_title("Action voltage (cyan) and reward (gray)")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(
        [action_line, reward_line],
        ["Action voltage", "Reward"],
        loc="upper left",
        framealpha=0.95,
        facecolor="white",
    )

    for ax in axes:
        ax.axvline(env_switch_time, color="gray", lw=1.0, ls="--", alpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_action_usage(
    history: dict[str, Any],
    out_path: str | Path,
    title: str,
    action_levels: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> None:
    actions = history_array(history, "u_v_all" if "u_v_all" in history else "u_v", dtype=np.float64)
    labels = history_array(history, "env_label_all" if "env_label_all" in history else "env_label", dtype=object)
    if actions.size == 0:
        return

    levels = resolve_action_levels(action_levels)
    envs = [("skin", "tab:blue"), ("fat", "tab:orange")]
    x = np.arange(levels.size)
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, (env_name, color) in enumerate(envs):
        mask = labels == env_name
        counts = np.array([(actions[mask] == level).sum() for level in levels], dtype=np.float64)
        total = max(1.0, counts.sum())
        ax.bar(x + ((idx - 0.5) * width), counts / total, width=width, color=color, alpha=0.85, label=env_name)

    ax.set_xticks(x, [f"{v:.0f}" for v in levels])
    ax.set_xlabel("Voltage action [V]")
    ax.set_ylabel("Action frequency")
    ax.set_title(f"{title}: action usage by environment")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_state_trajectory(history: dict[str, Any], out_path: str | Path, title: str) -> None:
    tracking_error = history_array(history, "pos_error", dtype=np.float64)
    v_m = history_array(history, "v_m", dtype=np.float64)
    v_s = history_array(history, "v_s", dtype=np.float64)
    t = history_array(history, "time", dtype=np.float64)
    if tracking_error.size == 0 or v_m.size == 0 or v_s.size == 0:
        return

    velocity_error = v_m - v_s
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.plot(tracking_error * 1000.0, velocity_error, lw=0.9, alpha=0.35, color="tab:gray")
    sc = ax.scatter(tracking_error * 1000.0, velocity_error, c=t, s=16, cmap="viridis", alpha=0.90, edgecolor="none")
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


def save_common_visuals(
    history: dict[str, Any],
    plots_dir: str | Path,
    title: str,
    env_switch_time: float,
    action_levels: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> None:
    plots_dir = Path(plots_dir)
    plot_average_core_rollout(history, plots_dir / "avg_roll.png", title, env_switch_time)
    plot_rollout_dashboard(history, plots_dir / "roll.png", title, env_switch_time)
    plot_action_usage(history, plots_dir / "act.png", title, action_levels=action_levels)
    plot_state_trajectory(history, plots_dir / "traj.png", title)


def _slice_index(dim_size: int) -> int:
    return max(0, min(dim_size - 1, dim_size // 2))


def plot_qlearning_policy_maps(
    agent: QLearningAgent,
    feature_names: tuple[str, ...],
    out_path: str | Path,
    action_levels: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> None:
    if len(agent.state_dims) < 2:
        return
    levels = resolve_action_levels(action_levels, expected_n_actions=agent.n_actions)
    x_dim, y_dim = agent.state_dims[0], agent.state_dims[1]
    extra_dims = tuple(_slice_index(dim) for dim in agent.state_dims[2:])
    action_map = np.zeros((y_dim, x_dim), dtype=np.float64)
    visit_map = np.zeros((y_dim, x_dim), dtype=np.float64)
    gap_map = np.zeros((y_dim, x_dim), dtype=np.float64)
    zero_visits = np.zeros(agent.n_actions, dtype=np.int64)

    for ix in range(x_dim):
        for iy in range(y_dim):
            state = (ix, iy, *extra_dims)
            q_values = agent.q_values(state)
            visits = agent.visit_count.get(state, zero_visits)
            action_map[iy, ix] = levels[greedy_q_action(q_values, action_levels=levels)]
            visit_map[iy, ix] = float(np.sum(visits))
            gap_map[iy, ix] = q_gap(q_values)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    im0 = axes[0].imshow(action_map, origin="lower", aspect="auto", cmap="coolwarm")
    axes[0].set_title("Action map")
    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])
    plt.colorbar(im0, ax=axes[0], label="Voltage [V]")

    im1 = axes[1].imshow(np.log1p(visit_map), origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title("Visitation trust")
    axes[1].set_xlabel(feature_names[0])
    axes[1].set_ylabel(feature_names[1])
    plt.colorbar(im1, ax=axes[1], label="log(1 + visits)")

    im2 = axes[2].imshow(gap_map, origin="lower", aspect="auto", cmap="viridis")
    axes[2].set_title("Q-gap confidence")
    axes[2].set_xlabel(feature_names[0])
    axes[2].set_ylabel(feature_names[1])
    plt.colorbar(im2, ax=axes[2], label="Q gap")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qlearning_state_visit_heatmap(agent: QLearningAgent, feature_names: tuple[str, ...], out_path: str | Path) -> None:
    if len(agent.state_dims) < 2:
        return
    x_dim, y_dim = agent.state_dims[0], agent.state_dims[1]
    heatmap = np.zeros((y_dim, x_dim), dtype=np.float64)
    for state, visits in agent.visit_count.items():
        if len(state) < 2:
            continue
        heatmap[state[1], state[0]] += float(np.sum(visits))

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(np.log1p(heatmap), origin="lower", aspect="auto", cmap="magma")
    ax.set_title("State visitation heatmap")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    plt.colorbar(im, ax=ax, label="log(1 + visits)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def default_context_obs() -> np.ndarray:
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


def dqn_slice_templates(history: dict[str, Any]) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    obs = history_array(history, "obs", dtype=np.float32)
    labels = history_array(history, "env_label", dtype=object)
    f_h = history_array(history, "F_h", dtype=np.float64)
    f_e = history_array(history, "F_e", dtype=np.float64)
    if obs.size == 0:
        return [("overall", default_context_obs(), {"F_h": 0.0, "F_e": 0.0})]

    if obs.ndim == 1:
        obs = obs.reshape(1, -1)

    templates: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for label in ("skin", "fat"):
        mask = labels == label
        if np.any(mask):
            templates.append(
                (
                    label,
                    np.median(obs[mask], axis=0).astype(np.float32),
                    {
                        "F_h": float(np.median(f_h[mask])) if f_h.size else 0.0,
                        "F_e": float(np.median(f_e[mask])) if f_e.size else 0.0,
                    },
                )
            )
    templates.append(
        (
            "overall",
            np.median(obs, axis=0).astype(np.float32),
            {
                "F_h": float(np.median(f_h)) if f_h.size else 0.0,
                "F_e": float(np.median(f_e)) if f_e.size else 0.0,
            },
        )
    )
    return templates


def plot_dqn_policy_slices(agent: DQNAgent, history: dict[str, Any], state_variant, out_path: str | Path) -> None:
    obs = history_array(history, "obs", dtype=np.float32)
    labels = history_array(history, "env_label", dtype=object)
    if obs.size == 0 or getattr(state_variant, "obs_dim", 0) < 2:
        return
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)

    x_idx, y_idx = 0, 1
    x_name = state_variant.feature_names[x_idx]
    y_name = state_variant.feature_names[y_idx]

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
        action_map = np.zeros((y_centers.size, x_centers.size), dtype=np.float64)
        maxq_map = np.zeros_like(action_map)
        gap_map = np.zeros_like(action_map)

        for x_plot_idx, x_value in enumerate(x_centers):
            for y_plot_idx, y_value in enumerate(y_centers):
                variant_obs = np.array(context_obs, dtype=np.float32, copy=True)
                variant_obs[x_idx] = float(x_value)
                variant_obs[y_idx] = float(y_value)
                q_values = agent.q_values(variant_obs)
                action_map[y_plot_idx, x_plot_idx] = cfg.V_LEVELS[int(np.argmax(q_values))]
                maxq_map[y_plot_idx, x_plot_idx] = float(np.max(q_values))
                gap_map[y_plot_idx, x_plot_idx] = q_gap(q_values)

        extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]
        im0 = axes[row_idx, 0].imshow(action_map, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
        axes[row_idx, 0].set_title(f"Action slice | {label}")
        plt.colorbar(im0, ax=axes[row_idx, 0], label="Voltage [V]")

        im1 = axes[row_idx, 1].imshow(maxq_map, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        axes[row_idx, 1].set_title(f"Max-Q slice | {label}")
        plt.colorbar(im1, ax=axes[row_idx, 1], label="max Q")

        im2 = axes[row_idx, 2].imshow(gap_map, origin="lower", aspect="auto", extent=extent, cmap="plasma")
        axes[row_idx, 2].set_title(f"Q-gap confidence | {label}")
        plt.colorbar(im2, ax=axes[row_idx, 2], label="Q gap")

        for col in range(3):
            axes[row_idx, col].set_xlabel(f"{x_name} [scaled]")
            axes[row_idx, col].set_ylabel(f"{y_name} [scaled]")
            axes[row_idx, col].grid(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def action_fraction(
    rows: list[dict[str, Any]],
    env_name: str | None = None,
    action_levels: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> dict[str, float]:
    levels = resolve_action_levels(action_levels)
    filtered = rows if env_name is None else [row for row in rows if row["env_label"] == env_name]
    if not filtered:
        return {f"{voltage:.1f}": 0.0 for voltage in levels}
    counts = {f"{voltage:.1f}": 0.0 for voltage in levels}
    for row in filtered:
        key = f"{row['u_v']:.1f}"
        counts[key] += 1.0
    total = float(len(filtered))
    return {key: value / total for key, value in counts.items()}


def policy_summary(
    policy_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    action_levels: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> dict[str, Any]:
    levels = resolve_action_levels(action_levels)
    q_gap_arr = np.asarray([row["q_gap"] for row in policy_rows], dtype=np.float64)
    max_q = np.asarray([row["max_q"] for row in policy_rows], dtype=np.float64)
    chosen_q = np.asarray([row["chosen_q"] for row in policy_rows], dtype=np.float64)
    action_voltage = np.asarray([row["u_v"] for row in policy_rows], dtype=np.float64)
    env_rows = {env_name: [row for row in policy_rows if row["env_label"] == env_name] for env_name in cfg.ENV_LABELS}
    scenario_names = sorted({row["scenario_name"] for row in policy_rows})
    return {
        "total_steps": int(len(policy_rows)),
        "mean_q_gap": float(np.mean(q_gap_arr)) if q_gap_arr.size else 0.0,
        "std_q_gap": float(np.std(q_gap_arr)) if q_gap_arr.size else 0.0,
        "mean_max_q": float(np.mean(max_q)) if max_q.size else 0.0,
        "std_max_q": float(np.std(max_q)) if max_q.size else 0.0,
        "mean_chosen_q": float(np.mean(chosen_q)) if chosen_q.size else 0.0,
        "std_chosen_q": float(np.std(chosen_q)) if chosen_q.size else 0.0,
        "mean_action_voltage_v": float(np.mean(action_voltage)) if action_voltage.size else 0.0,
        "mean_abs_action_voltage_v": float(np.mean(np.abs(action_voltage))) if action_voltage.size else 0.0,
        "action_levels": levels.tolist(),
        "action_usage_fraction": action_fraction(policy_rows, action_levels=levels),
        "action_usage_by_env_fraction": {
            env_name: action_fraction(policy_rows, env_name=env_name, action_levels=levels)
            for env_name in cfg.ENV_LABELS
        },
        "env_policy_metrics": {
            env_name: {
                "steps": int(len(rows)),
                "mean_q_gap": float(np.mean(np.asarray([row["q_gap"] for row in rows], dtype=np.float64))) if rows else 0.0,
                "mean_max_q": float(np.mean(np.asarray([row["max_q"] for row in rows], dtype=np.float64))) if rows else 0.0,
                "mean_abs_action_voltage_v": float(np.mean(np.abs(np.asarray([row["u_v"] for row in rows], dtype=np.float64)))) if rows else 0.0,
            }
            for env_name, rows in env_rows.items()
        },
        "scenario_metrics": {
            name: {
                "tracking_rmse_mm": float(next(row["tracking_rmse_mm"] for row in episode_rows if row["scenario_name"] == name)),
                "transparency_rmse_w": float(next(row["transparency_rmse_w"] for row in episode_rows if row["scenario_name"] == name)),
                "episode_return": float(next(row["episode_return"] for row in episode_rows if row["scenario_name"] == name)),
                "mean_q_gap": float(np.mean(np.asarray([row["q_gap"] for row in policy_rows if row["scenario_name"] == name], dtype=np.float64))),
                "mean_abs_action_voltage_v": float(np.mean(np.abs(np.asarray([row["u_v"] for row in policy_rows if row["scenario_name"] == name], dtype=np.float64)))),
            }
            for name in scenario_names
        },
    }


def plot_episode_metrics(rows: list[dict[str, Any]], out_path: Path) -> None:
    episodes = np.asarray([row["episode"] for row in rows], dtype=np.int64)
    returns = np.asarray([row["episode_return"] for row in rows], dtype=np.float64)
    tracking = np.asarray([row["tracking_rmse_mm"] for row in rows], dtype=np.float64)
    transparency = np.asarray([row["transparency_rmse_w"] for row in rows], dtype=np.float64)
    q_gap_arr = np.asarray([row["mean_q_gap"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(episodes, returns, marker="o", lw=1.5, color="tab:blue")
    axes[0, 0].set_title("Episode return")
    axes[0, 1].plot(episodes, tracking, marker="o", lw=1.5, color="tab:orange")
    axes[0, 1].set_title("Tracking RMSE")
    axes[1, 0].plot(episodes, transparency, marker="o", lw=1.5, color="tab:green")
    axes[1, 0].set_title("Transparency RMSE")
    axes[1, 1].plot(episodes, q_gap_arr, marker="o", lw=1.5, color="tab:red")
    axes[1, 1].set_title("Mean policy Q-gap")
    for ax in axes.ravel():
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("Return")
    axes[0, 1].set_ylabel("RMSE [mm]")
    axes[1, 0].set_ylabel("RMSE [W]")
    axes[1, 1].set_ylabel("Q-gap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(summary: dict[str, Any], out_path: Path) -> None:
    labels = ["Return", "Track RMSE [mm]", "Transp RMSE [W]", "Mean |u_v| [V]", "Mean Q-gap"]
    values = [summary["mean_return"], summary["mean_tracking_rmse_mm"], summary["mean_transparency_rmse_w"], summary["mean_abs_u_v"], summary["mean_q_gap"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])
    ax.set_title("Greedy evaluation summary")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_policy_dashboard(
    policy_rows: list[dict[str, Any]],
    out_path: Path,
    env_switch_time: float,
    action_levels: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> None:
    if not policy_rows:
        return
    action_levels = resolve_action_levels(action_levels)
    overall = action_fraction(policy_rows, action_levels=action_levels)
    skin = action_fraction(policy_rows, env_name="skin", action_levels=action_levels)
    fat = action_fraction(policy_rows, env_name="fat", action_levels=action_levels)
    first_episode = [row for row in policy_rows if row["episode"] == 1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].bar([f"{v:.0f}" for v in action_levels], [overall[f"{v:.1f}"] for v in action_levels], color="tab:blue", alpha=0.85)
    axes[0, 0].set_title("Action usage across all test steps")
    axes[0, 0].set_xlabel("Voltage action [V]")
    axes[0, 0].set_ylabel("Fraction")
    x = np.arange(action_levels.size)
    width = 0.38
    axes[0, 1].bar(x - width / 2.0, [skin[f"{v:.1f}"] for v in action_levels], width=width, color="tab:blue", alpha=0.85, label="skin")
    axes[0, 1].bar(x + width / 2.0, [fat[f"{v:.1f}"] for v in action_levels], width=width, color="tab:orange", alpha=0.85, label="fat")
    axes[0, 1].set_xticks(x, [f"{v:.0f}" for v in action_levels])
    axes[0, 1].set_title("Action usage by environment")
    axes[0, 1].set_xlabel("Voltage action [V]")
    axes[0, 1].set_ylabel("Fraction")
    axes[0, 1].legend()
    t = np.asarray([row["time"] for row in first_episode], dtype=np.float64)
    u_v = np.asarray([row["u_v"] for row in first_episode], dtype=np.float64)
    q_gap_arr = np.asarray([row["q_gap"] for row in first_episode], dtype=np.float64)
    max_q = np.asarray([row["max_q"] for row in first_episode], dtype=np.float64)
    axes[1, 0].plot(t, u_v, lw=1.6, color="tab:red")
    axes[1, 0].axvline(env_switch_time, color="0.35", ls="--", lw=1.1)
    axes[1, 0].set_title("Greedy action over time (episode 1)")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Voltage [V]")
    axes[1, 1].plot(t, q_gap_arr, lw=1.6, color="tab:purple", label="Q-gap")
    axes[1, 1].plot(t, max_q, lw=1.2, color="tab:green", alpha=0.8, label="max Q")
    axes[1, 1].axvline(env_switch_time, color="0.35", ls="--", lw=1.1)
    axes[1, 1].set_title("Policy confidence over time (episode 1)")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_input_signal_dashboard(policy_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]], out_path: Path, env_switch_time: float) -> None:
    scenario_names = [row["scenario_name"] for row in episode_rows]
    if not scenario_names:
        return
    n_items = len(scenario_names)
    n_cols = 2
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.1 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for idx, scenario_name in enumerate(scenario_names):
        ax = axes[idx // n_cols, idx % n_cols]
        episode_row = episode_rows[idx]
        scenario_rows = [row for row in policy_rows if row["scenario_name"] == scenario_name]
        t = np.asarray([row["time"] for row in scenario_rows], dtype=np.float64)
        force_input = np.asarray([row["F_h"] for row in scenario_rows], dtype=np.float64)
        force_nominal = np.asarray([row["F_h_nominal"] for row in scenario_rows], dtype=np.float64)
        force_noise = np.asarray([row["F_h_noise"] for row in scenario_rows], dtype=np.float64)
        ax.plot(t, force_input, lw=1.4, color="tab:blue", label="Noisy input")
        if np.any(np.abs(force_nominal) > 1e-12):
            ax.plot(t, force_nominal, lw=1.0, ls="--", color="0.35", alpha=0.9, label="Nominal")
        if np.any(np.abs(force_noise) > 1e-12):
            ax.plot(t, force_noise, lw=0.9, color="tab:orange", alpha=0.85, label="Noise")
        ax.axvline(env_switch_time, color="0.35", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(
            f"{scenario_name} | track {episode_row['tracking_rmse_mm']:.2f} mm | "
            f"transp {episode_row['transparency_rmse_w']:.3f} W"
        )
        ax.set_ylabel("Force [N]")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)
    for idx in range(n_items, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_dashboard(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows or len({row["scenario_name"] for row in rows}) <= 1:
        return
    labels = [row["scenario_name"] for row in rows]
    tracking = np.asarray([row["tracking_rmse_mm"] for row in rows], dtype=np.float64)
    transparency = np.asarray([row["transparency_rmse_w"] for row in rows], dtype=np.float64)
    q_gap_arr = np.asarray([row["mean_q_gap"] for row in rows], dtype=np.float64)
    returns = np.asarray([row["episode_return"] for row in rows], dtype=np.float64)
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0, 0].bar(x, tracking, color="tab:orange")
    axes[0, 0].set_title("Tracking RMSE by scenario")
    axes[0, 0].set_ylabel("RMSE [mm]")
    axes[0, 1].bar(x, transparency, color="tab:green")
    axes[0, 1].set_title("Transparency RMSE by scenario")
    axes[0, 1].set_ylabel("RMSE [W]")
    axes[1, 0].bar(x, q_gap_arr, color="tab:purple")
    axes[1, 0].set_title("Mean Q-gap by scenario")
    axes[1, 0].set_ylabel("Q-gap")
    axes[1, 1].bar(x, returns, color="tab:blue")
    axes[1, 1].set_title("Return by scenario")
    axes[1, 1].set_ylabel("Return")
    for ax in axes.ravel():
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def scenario_plan(scenario_set: str | None, noise_std: float) -> list[dict[str, Any]] | None:
    amp = float(cfg.FORCE_INPUT_AMP)
    phase = float(cfg.FORCE_INPUT_PHASE)
    if scenario_set is None:
        return None
    if scenario_set == "force_generalization_10":
        return [
            {"name": "sine_0p25hz", "reset_options": {"force_amp": amp, "force_freq": 0.25, "force_phase": phase, "force_waveform": "sine"}},
            {"name": "sine_0p35hz", "reset_options": {"force_amp": amp, "force_freq": 0.35, "force_phase": phase, "force_waveform": "sine"}},
            {"name": "sine_0p50hz", "reset_options": {"force_amp": amp, "force_freq": 0.50, "force_phase": phase, "force_waveform": "sine"}},
            {"name": "sine_0p75hz", "reset_options": {"force_amp": amp, "force_freq": 0.75, "force_phase": phase, "force_waveform": "sine"}},
            {"name": "sine_1p00hz", "reset_options": {"force_amp": amp, "force_freq": 1.00, "force_phase": phase, "force_waveform": "sine"}},
            {"name": "cosine_0p50hz", "reset_options": {"force_amp": amp, "force_freq": 0.50, "force_phase": phase, "force_waveform": "cosine"}},
            {"name": "cosine_0p75hz", "reset_options": {"force_amp": amp, "force_freq": 0.75, "force_phase": phase, "force_waveform": "cosine"}},
            {"name": "multisine_0p50hz", "reset_options": {"force_amp": amp, "force_freq": 0.50, "force_phase": phase, "force_waveform": "multisine"}},
            {"name": "multisine_0p75hz", "reset_options": {"force_amp": amp, "force_freq": 0.75, "force_phase": phase, "force_waveform": "multisine"}},
            {"name": "multisine_1p00hz", "reset_options": {"force_amp": amp, "force_freq": 1.00, "force_phase": phase, "force_waveform": "multisine"}},
        ]
    if scenario_set == "force_square_10":
        square_freqs = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10]
        return [{"name": f"square_{freq:0.2f}hz".replace(".", "p"), "reset_options": {"force_amp": amp, "force_freq": float(freq), "force_phase": phase, "force_waveform": "square"}} for freq in square_freqs]
    if scenario_set == "force_noise_10":
        return [
            {
                "name": f"noisy_sine_{idx + 1:02d}",
                "reset_options": {
                    "force_amp": amp,
                    "force_freq": float(cfg.FORCE_INPUT_FREQ),
                    "force_phase": phase,
                    "force_waveform": "sine",
                    "force_noise_std": float(noise_std),
                    "force_noise_seed": 17_100 + idx,
                },
            }
            for idx in range(10)
        ]
    raise ValueError(f"Unknown scenario set: {scenario_set}")


def stage_completed(stage_dir: str | Path) -> bool:
    return (Path(stage_dir) / "study_summary.csv").exists()


def stage_summary_rows_to_csv(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    import csv

    if not rows:
        return
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_summary_json(summary_path: str | Path) -> dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
