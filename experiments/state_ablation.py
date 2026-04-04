"""Parallel DQN state-ablation study runner."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import numpy as np

from .. import config as cfg
from ..dqn_agent import DQNAgent
from ..state_variants import StateVariant, StateVariantEnv, build_state_variants, get_state_variant
from ..teleop_env import TeleopEnv
from .long_study import (
    RewardAblationEnv,
    RewardVariant,
    RunResult,
    _compute_eval_metrics,
    _evaluate_dqn,
    _mk_dirs,
    _require_tensorboard,
    _save_training_plot,
    _write_run_summary,
)
from .saved_dqn_eval import (
    _evaluate_saved_dqn,
    _plot_input_signal_dashboard,
    _plot_scenario_dashboard,
    _resolve_model_path,
)


DEFAULT_BASELINE_MODEL = Path(__file__).resolve().parents[1] / cfg.RESULTS_ROOT_DIR / "reward_function_study_dqn" / "dqn" / "v01" / "m" / "dqn_model.pt"


def _fs_path(path: str | Path) -> str:
    resolved = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def _study_root(study_name: str | None) -> str:
    base = Path(__file__).resolve().parents[1] / cfg.RESULTS_ROOT_DIR / "ablation studies" / "state ablation study"
    name = study_name or f"state_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = base / name
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(_fs_path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with open(_fs_path(path), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _build_reward_variant_from_summary(summary: dict[str, Any]) -> RewardVariant:
    reward_cfg = summary.get("reward_variant")
    if not reward_cfg:
        raise ValueError("Baseline summary does not include reward_variant; cannot freeze the DQN v01 reward.")
    return RewardVariant(
        name=str(reward_cfg["name"]),
        tracking_weight=float(reward_cfg["tracking_weight"]),
        transparency_weight=float(reward_cfg["transparency_weight"]),
        jerk_weight=float(reward_cfg["jerk_weight"]),
        use_jerk=bool(reward_cfg["use_jerk"]),
    )


def _transfer_prefix(scenario_set: str, noise_std: float) -> str:
    prefix = scenario_set
    if scenario_set == "force_noise_10":
        noise_label = f"{noise_std:.2f}".replace(".", "p")
        prefix = f"{prefix}_n{noise_label}"
    return prefix


def _variant_model_path(result: RunResult) -> str:
    return str(result.model_path)


def _variant_run_id(state_variant_name: str) -> str:
    return state_variant_name.split("_", 1)[0]


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


def _save_transfer_bundle(
    run_dir: str,
    model_path: str,
    scenario_set: str,
    noise_std: float,
    seed: int,
    disable_jerk: bool = False,
) -> dict[str, Any]:
    os.makedirs(run_dir, exist_ok=True)
    rows, summary, policy_rows, policy_summary, env_switch_time = _evaluate_saved_dqn(
        model_path=Path(model_path),
        n_episodes=10,
        seed=seed,
        scenario_set=scenario_set,
        noise_std=noise_std,
        disable_jerk=disable_jerk,
    )
    dirs = _mk_dirs(run_dir)
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    prefix = _transfer_prefix(scenario_set, noise_std)
    metrics_csv_path = os.path.join(dirs["logs"], f"{prefix}_metrics.csv")
    summary_json_path = os.path.join(dirs["logs"], f"{prefix}_summary.json")
    scenario_plot_path = os.path.join(dirs["plots"], f"{prefix}_scenario_dashboard.png")
    input_plot_path = os.path.join(dirs["plots"], f"{prefix}_input_signal_dashboard.png")

    with open(_fs_path(metrics_csv_path), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _save_json(summary_json_path, summary)
    _plot_scenario_dashboard(rows, Path(scenario_plot_path))
    _plot_input_signal_dashboard(policy_rows, rows, Path(input_plot_path), env_switch_time)

    return {
        "summary": summary,
        "policy_summary": policy_summary,
        "metrics_csv": metrics_csv_path,
        "summary_json": summary_json_path,
        "scenario_dashboard": scenario_plot_path,
        "input_signal_dashboard": input_plot_path,
    }


def _reward_variant_from_dict(payload: dict[str, Any]) -> RewardVariant:
    return RewardVariant(
        name=str(payload["name"]),
        tracking_weight=float(payload["tracking_weight"]),
        transparency_weight=float(payload["transparency_weight"]),
        jerk_weight=float(payload["jerk_weight"]),
        use_jerk=bool(payload["use_jerk"]),
    )


def _train_state_variant(
    study_root: str,
    env_mode: str,
    master_input_mode: str,
    total_episodes: int,
    test_episodes: int,
    seed: int,
    reward_variant: RewardVariant,
    state_variant: StateVariant,
) -> RunResult:
    writer_cls = _require_tensorboard()
    run_id = _variant_run_id(state_variant.name)
    dirs = _mk_dirs(os.path.join(study_root, run_id))
    writer = writer_cls(log_dir=dirs["tensorboard"])

    env_factory = lambda: StateVariantEnv(
        RewardAblationEnv(
            TeleopEnv(env_mode=env_mode, master_input_mode=master_input_mode),
            reward_variant,
        ),
        state_variant,
    )

    train_env = env_factory()
    agent = DQNAgent(obs_dim=state_variant.obs_dim, n_actions=cfg.N_ACTIONS, seed=seed)

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
                f"[{state_variant.name}] ep {step}/{total_episodes} | "
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
        state_variant.name,
        losses=ep_loss,
    )

    mean_reward, tracking_rmse, transparency_rmse, history = _evaluate_dqn(
        agent,
        env_factory,
        n_episodes=test_episodes,
        seed_offset=40_000,
    )

    result = RunResult(
        label=state_variant.name,
        family="dqn_state_ablation",
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
            "reward_variant": asdict(reward_variant),
            "state_variant_name": state_variant.name,
            "feature_names": list(state_variant.feature_names),
            "state_variant_description": state_variant.description,
            "obs_dim": state_variant.obs_dim,
        },
    )
    return result


def _train_and_evaluate_variant_task(task: dict[str, Any]) -> dict[str, Any]:
    state_variant = get_state_variant(str(task["state_variant_name"]))
    reward_variant = _reward_variant_from_dict(dict(task["reward_variant"]))
    result = _train_state_variant(
        study_root=str(task["study_root"]),
        env_mode=str(task["env_mode"]),
        master_input_mode=str(task["master_input_mode"]),
        total_episodes=int(task["episodes"]),
        test_episodes=int(task["test_episodes"]),
        seed=int(task["seed"]),
        reward_variant=reward_variant,
        state_variant=state_variant,
    )
    transfer = _save_transfer_bundle(
        run_dir=result.out_dir,
        model_path=_variant_model_path(result),
        scenario_set=str(task["scenario_set"]),
        noise_std=float(task["noise_std"]),
        seed=int(task["transfer_seed"]),
        disable_jerk=bool(task.get("disable_jerk", False)),
    )
    return {
        "order": int(task["order"]),
        "label": state_variant.name,
        "out_dir": result.out_dir,
        "summary_row": _trained_variant_row(result, state_variant, transfer["summary"]),
    }


def _baseline_reference_row(
    baseline_summary: dict[str, Any],
    transfer_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": "S0_baseline_full10",
        "kind": "baseline_reference",
        "obs_dim": 10,
        "feature_names": "x_s_eq|x_m_eq|v_s|v_m|P_s1|P_s2|P_m1|P_m2|mdot_L1|mdot_L2",
        "standard_mean_reward": float(baseline_summary["mean_reward"]),
        "standard_tracking_rmse_mm": float(baseline_summary["tracking_rmse_m"]) * 1000.0,
        "standard_transparency_rmse_w": float(baseline_summary["transparency_rmse_w"]),
        "transfer_mean_return": float(transfer_summary["mean_return"]),
        "transfer_tracking_rmse_mm": float(transfer_summary["mean_tracking_rmse_mm"]),
        "transfer_transparency_rmse_w": float(transfer_summary["mean_transparency_rmse_w"]),
        "transfer_mean_abs_u_v": float(transfer_summary["mean_abs_u_v"]),
        "transfer_mean_abs_delta_u_v": float(transfer_summary["mean_abs_delta_u_v"]),
        "transfer_mean_q_gap": float(transfer_summary["mean_q_gap"]),
        "transfer_mean_max_q": float(transfer_summary["mean_max_q"]),
        "model_path": str(baseline_summary["model_path"]),
    }


def _trained_variant_row(
    result: RunResult,
    state_variant: StateVariant,
    transfer_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": state_variant.name,
        "kind": "trained_variant",
        "obs_dim": state_variant.obs_dim,
        "feature_names": "|".join(state_variant.feature_names),
        "standard_mean_reward": float(result.mean_reward),
        "standard_tracking_rmse_mm": float(result.tracking_rmse_m) * 1000.0,
        "standard_transparency_rmse_w": float(result.transparency_rmse_w),
        "transfer_mean_return": float(transfer_summary["mean_return"]),
        "transfer_tracking_rmse_mm": float(transfer_summary["mean_tracking_rmse_mm"]),
        "transfer_transparency_rmse_w": float(transfer_summary["mean_transparency_rmse_w"]),
        "transfer_mean_abs_u_v": float(transfer_summary["mean_abs_u_v"]),
        "transfer_mean_abs_delta_u_v": float(transfer_summary["mean_abs_delta_u_v"]),
        "transfer_mean_q_gap": float(transfer_summary["mean_q_gap"]),
        "transfer_mean_max_q": float(transfer_summary["mean_max_q"]),
        "model_path": str(result.model_path),
    }


def _write_study_summary(study_root: str, rows: list[dict[str, Any]]) -> str:
    out_path = os.path.join(study_root, "study_summary.csv")
    with open(_fs_path(out_path), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN state-ablation variants and evaluate them with parallel workers.")
    parser.add_argument("--study-name", default=None, help="Optional study folder name under results_fh/ablation studies/state ablation study.")
    parser.add_argument("--baseline-model-path", default=str(DEFAULT_BASELINE_MODEL), help="Saved baseline DQN model folder or dqn_model.pt path.")
    parser.add_argument("--episodes", type=int, default=cfg.DQN_NUM_EPISODES, help="Training episodes for each new state variant.")
    parser.add_argument("--test-episodes", type=int, default=100, help="Greedy test episodes for the standard post-training summary.")
    parser.add_argument(
        "--scenario-set",
        choices=["force_generalization_10", "force_square_10", "force_noise_10"],
        default="force_generalization_10",
        help="10-input transfer-evaluation suite to run for the baseline and each trained variant.",
    )
    parser.add_argument("--noise-std", type=float, default=0.5, help="Noise std in Newtons when --scenario-set force_noise_10 is used.")
    parser.add_argument("--seed", type=int, default=80_000, help="Base seed for training and evaluation.")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=min(24, os.cpu_count() or 1),
        help="Maximum number of parallel state-variant worker processes.",
    )
    parser.add_argument(
        "--worker-torch-threads",
        type=int,
        default=1,
        help="Torch CPU thread count inside each worker process to avoid oversubscription.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Optional subset of state variant names to train. Baseline is always evaluated without retraining.",
    )
    args = parser.parse_args()

    study_root = _study_root(args.study_name)
    baseline_model_path = _resolve_model_path(args.baseline_model_path)
    baseline_summary = _load_json(baseline_model_path.parent.parent / "l" / "summary.json")
    reward_variant = _build_reward_variant_from_summary(baseline_summary)

    all_variants = build_state_variants()
    train_variants = [variant for variant in all_variants if variant.name != "S0_baseline_full10"]
    if args.variants:
        selected = {name.strip() for name in args.variants}
        train_variants = [variant for variant in train_variants if variant.name in selected]
        missing = sorted(selected.difference({variant.name for variant in train_variants}))
        if missing:
            raise ValueError(f"Unknown --variants entries: {missing}")

    detected_cpus = max(1, int(os.cpu_count() or 1))
    requested_workers = max(1, int(args.parallel_workers))
    parallel_workers = max(1, min(requested_workers, len(train_variants))) if train_variants else 1
    worker_torch_threads = _configure_process_env(int(args.worker_torch_threads))

    manifest = {
        "study_root": study_root,
        "baseline_model_path": str(baseline_model_path),
        "env_mode": str(baseline_summary["env_mode"]),
        "master_input_mode": str(baseline_summary["master_input_mode"]),
        "episodes": int(args.episodes),
        "test_episodes": int(args.test_episodes),
        "scenario_set": args.scenario_set,
        "noise_std": float(args.noise_std),
        "cpu_count_detected": detected_cpus,
        "parallel_workers_requested": requested_workers,
        "parallel_workers_used": parallel_workers,
        "worker_torch_threads": worker_torch_threads,
        "reward_variant": asdict(reward_variant),
        "train_variants": [
            {
                "name": variant.name,
                "obs_dim": variant.obs_dim,
                "feature_names": list(variant.feature_names),
                "description": variant.description,
            }
            for variant in train_variants
        ],
    }
    _save_json(os.path.join(study_root, "study_manifest.json"), manifest)

    print(f"Study root: {study_root}")
    print("Evaluating frozen baseline DQN v01 on the transfer suite...")
    baseline_transfer = _save_transfer_bundle(
        run_dir=os.path.join(study_root, "S0"),
        model_path=str(baseline_model_path),
        scenario_set=args.scenario_set,
        noise_std=args.noise_std,
        seed=args.seed + 1_000,
    )

    baseline_row = _baseline_reference_row(baseline_summary, baseline_transfer["summary"])
    summary_rows: list[dict[str, Any]] = [baseline_row]

    if train_variants:
        print(
            f"Launching {len(train_variants)} trainable variants with up to "
            f"{parallel_workers} worker processes "
            f"(detected CPUs: {detected_cpus}, per-worker torch threads: {worker_torch_threads})."
        )

    if parallel_workers <= 1:
        for idx, state_variant in enumerate(train_variants, start=1):
            print("")
            print(f"[{idx}/{len(train_variants)}] Training {state_variant.name}...")
            payload = _train_and_evaluate_variant_task(
                {
                    "order": idx,
                    "study_root": study_root,
                    "env_mode": str(baseline_summary["env_mode"]),
                    "master_input_mode": str(baseline_summary["master_input_mode"]),
                    "episodes": int(args.episodes),
                    "test_episodes": int(args.test_episodes),
                    "seed": args.seed + (idx * 10_000),
                    "reward_variant": asdict(reward_variant),
                    "state_variant_name": state_variant.name,
                    "scenario_set": args.scenario_set,
                    "noise_std": float(args.noise_std),
                    "transfer_seed": args.seed + 500_000 + (idx * 10_000),
                    "disable_jerk": False,
                }
            )
            summary_rows.append(payload["summary_row"])
            _write_study_summary(study_root, summary_rows)
    else:
        tasks = [
            {
                "order": idx,
                "study_root": study_root,
                "env_mode": str(baseline_summary["env_mode"]),
                "master_input_mode": str(baseline_summary["master_input_mode"]),
                "episodes": int(args.episodes),
                "test_episodes": int(args.test_episodes),
                "seed": args.seed + (idx * 10_000),
                "reward_variant": asdict(reward_variant),
                "state_variant_name": state_variant.name,
                "scenario_set": args.scenario_set,
                "noise_std": float(args.noise_std),
                "transfer_seed": args.seed + 500_000 + (idx * 10_000),
                "disable_jerk": False,
            }
            for idx, state_variant in enumerate(train_variants, start=1)
        ]
        completed_rows: dict[int, dict[str, Any]] = {}
        with cf.ProcessPoolExecutor(
            max_workers=parallel_workers,
            initializer=_worker_runtime_init,
            initargs=(worker_torch_threads,),
        ) as executor:
            future_map = {
                executor.submit(_train_and_evaluate_variant_task, task): (
                    int(task["order"]),
                    str(task["state_variant_name"]),
                )
                for task in tasks
            }
            for future in cf.as_completed(future_map):
                order, name = future_map[future]
                payload = future.result()
                completed_rows[order] = payload["summary_row"]
                print(f"[done {order}/{len(train_variants)}] {name} -> {payload['out_dir']}")
                summary_rows = [baseline_row] + [completed_rows[idx] for idx in sorted(completed_rows)]
                _write_study_summary(study_root, summary_rows)

        summary_rows = [baseline_row] + [completed_rows[idx] for idx in sorted(completed_rows)]

    summary_csv = _write_study_summary(study_root, summary_rows)
    print("")
    print(f"State ablation study complete. Summary: {summary_csv}")


if __name__ == "__main__":
    main()
