from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
from pathlib import Path
from typing import Any
from datetime import datetime

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL import config as cfg
    from TeleopWithRL.matlab_literal_env.simuoriginal_replica import FE_MODE_CHOICES, FE_MODE_GUI
    from TeleopWithRL.matlab_literal_env.scripts._common import replica_env_kwargs_from_args
    from TeleopWithRL.matlab_literal_env.studies.common import (
        results_root,
        save_json,
        stage_completed,
        stage_summary_rows_to_csv,
        study_root,
    )
    from TeleopWithRL.matlab_literal_env.studies.dqn import train_dqn_variant
    from TeleopWithRL.matlab_literal_env.studies.dqn_state_variants import build_dqn_state_variants, get_dqn_state_variant
    from TeleopWithRL.matlab_literal_env.studies.qlearning import train_qlearning_variant
    from TeleopWithRL.matlab_literal_env.studies.ql_state_variants import build_ql_state_variants, get_ql_state_variant
    from TeleopWithRL.matlab_literal_env.studies.rewarding import (
        baseline_reward_variant,
        build_core_reward_variants,
        build_full_reward_variants,
        reward_variant_from_name,
    )
    from TeleopWithRL.matlab_literal_env.studies.saved_policy_eval import (
        evaluate_saved_policy,
        save_evaluation_bundle,
    )
else:
    from ... import config as cfg
    from ..simuoriginal_replica import FE_MODE_CHOICES, FE_MODE_GUI
    from ._common import replica_env_kwargs_from_args
    from ..studies.common import results_root, save_json, stage_completed, stage_summary_rows_to_csv, study_root
    from ..studies.dqn import train_dqn_variant
    from ..studies.dqn_state_variants import build_dqn_state_variants, get_dqn_state_variant
    from ..studies.qlearning import train_qlearning_variant
    from ..studies.ql_state_variants import build_ql_state_variants, get_ql_state_variant
    from ..studies.rewarding import baseline_reward_variant, build_core_reward_variants, build_full_reward_variants
    from ..studies.rewarding import reward_variant_from_name
    from ..studies.saved_policy_eval import evaluate_saved_policy, save_evaluation_bundle


DEFAULT_SWEEP_Q_EPISODES = 5_000
DEFAULT_SWEEP_DQN_EPISODES = 2_500
DEFAULT_PARALLEL_WORKERS = max(1, min(8, os.cpu_count() or 1))


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "l" / "summary.json"


def _log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    with open(_summary_path(run_dir), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_or_load(run_dir: Path, resume: bool, trainer, **kwargs) -> dict[str, Any]:
    summary_path = _summary_path(run_dir)
    if resume and summary_path.exists():
        _log(f"resume {run_dir}")
        return _load_summary(run_dir)
    _log(f"run {run_dir}")
    trainer(out_dir=run_dir, **kwargs)
    _log(f"done {run_dir}")
    return _load_summary(run_dir)


def _row_from_summary(agent: str, study_family: str, variant_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent": agent,
        "study_family": study_family,
        "variant_name": variant_name,
        "state_variant": str(summary["state_variant"]),
        "reward_variant": str(summary["reward_variant"]),
        "tracking_rmse_m": float(summary["tracking_rmse_m"]),
        "transparency_rmse_w": float(summary["transparency_rmse_w"]),
        "pre_switch_tracking_rmse_m": float(summary["pre_switch_tracking_rmse_m"]),
        "post_switch_tracking_rmse_m": float(summary["post_switch_tracking_rmse_m"]),
        "pre_switch_transparency_rmse_w": float(summary["pre_switch_transparency_rmse_w"]),
        "post_switch_transparency_rmse_w": float(summary["post_switch_transparency_rmse_w"]),
        "mean_reward": float(summary["mean_reward"]),
        "invalid_episode_rate": float(summary["invalid_episode_rate"]),
        "model_path": str(summary["model_path"]),
        "out_dir": str(summary["out_dir"]),
    }


def _select_best(rows: list[dict[str, Any]], baseline_tracking_rmse_m: float) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if float(row["tracking_rmse_m"]) <= (1.25 * baseline_tracking_rmse_m)
        and float(row["invalid_episode_rate"]) <= 0.05
    ]
    candidates = eligible if eligible else rows
    return sorted(
        candidates,
        key=lambda row: (
            float(row["transparency_rmse_w"]),
            float(row["tracking_rmse_m"]),
            -float(row["mean_reward"]),
        ),
    )[0]


def _write_stage_outputs(stage_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    save_json(stage_dir / "study_manifest.json", manifest)
    stage_summary_rows_to_csv(rows, stage_dir / "study_summary.csv")


def _short_variant_dir(name: str) -> str:
    head = name.split("_", 1)[0]
    return head or name


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
    threads = _configure_process_env(int(torch_threads))
    try:
        import torch

        torch.set_num_threads(threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(threads, 2)))
    except Exception:
        pass


def _train_stage_task(task: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(str(task["out_dir"]))
    if bool(task.get("resume")) and _summary_path(out_dir).exists():
        summary = _load_summary(out_dir)
        return {"order": int(task["order"]), "name": str(task["name"]), "summary": summary}

    family = str(task["family"])
    reward_variant = reward_variant_from_name(str(task["reward_variant"]))
    env_kwargs = dict(task["env_kwargs"])
    common = dict(
        out_dir=out_dir,
        env_mode=str(task["env_mode"]),
        env_kwargs=env_kwargs,
        reward_variant=reward_variant,
        total_episodes=int(task["total_episodes"]),
        test_episodes=int(task["test_episodes"]),
        seed=int(task["seed"]),
        label=str(task["label"]),
    )
    if family == "ql":
        train_qlearning_variant(
            state_variant=get_ql_state_variant(str(task["state_variant"])),
            **common,
        )
    elif family == "dqn":
        train_dqn_variant(
            state_variant=get_dqn_state_variant(str(task["state_variant"])),
            parallel_envs=int(task.get("parallel_envs", 1)),
            **common,
        )
    else:
        raise ValueError(f"Unknown training family: {family}")

    summary = _load_summary(out_dir)
    return {"order": int(task["order"]), "name": str(task["name"]), "summary": summary}


def _run_stage_train_tasks(
    *,
    stage_name: str,
    tasks: list[dict[str, Any]],
    parallel_workers: int,
    worker_torch_threads: int,
) -> list[dict[str, Any]]:
    if not tasks:
        return []

    max_workers = max(1, min(int(parallel_workers), len(tasks)))
    if max_workers <= 1:
        _worker_runtime_init(worker_torch_threads)
        completed: list[dict[str, Any]] = []
        for idx, task in enumerate(tasks, start=1):
            _log(f"{stage_name} {idx}/{len(tasks)} | {task['name']}")
            payload = _train_stage_task(task)
            completed.append(payload)
            _log(f"[done {idx}/{len(tasks)}] {task['name']}")
        return sorted(completed, key=lambda item: int(item["order"]))

    detected_cpus = os.cpu_count() or 1
    _log(
        f"{stage_name} | launching {len(tasks)} jobs with up to {max_workers} worker processes "
        f"(detected CPUs: {detected_cpus}, per-worker torch threads: {worker_torch_threads})"
    )
    completed_map: dict[int, dict[str, Any]] = {}
    with cf.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_runtime_init,
        initargs=(int(worker_torch_threads),),
    ) as executor:
        future_map = {
            executor.submit(_train_stage_task, task): (int(task["order"]), str(task["name"]))
            for task in tasks
        }
        for done_idx, future in enumerate(cf.as_completed(future_map), start=1):
            order, task_name = future_map[future]
            payload = future.result()
            completed_map[order] = payload
            _log(f"[done {done_idx}/{len(tasks)}] {task_name}")
    return [completed_map[idx] for idx in sorted(completed_map)]


def _saved_eval_task(task: dict[str, Any]) -> dict[str, Any]:
    best_row = dict(task["best_row"])
    agent_name = str(task["agent_name"])
    out_dir = Path(str(task["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_rows, summary, policy_rows, policy_summary_payload, env_switch_time = evaluate_saved_policy(
        model_path=str(best_row["model_path"]),
        episodes=int(task["test_episodes"]),
        seed=int(task["seed"]),
        scenario_set=None,
        noise_std=float(task["noise_std"]),
    )
    save_evaluation_bundle(
        out_dir=out_dir,
        prefix="canon",
        episode_rows=episode_rows,
        summary=summary,
        policy_rows=policy_rows,
        policy_summary_payload=policy_summary_payload,
        env_switch_time=env_switch_time,
    )
    transfer_outputs = {}
    for scenario_set, prefix in (
        ("force_generalization_10", "gen10"),
        ("force_square_10", "square10"),
        ("force_noise_10", "noise10"),
    ):
        ep_rows, bundle_summary, bundle_policy_rows, bundle_policy_summary, switch_time = evaluate_saved_policy(
            model_path=str(best_row["model_path"]),
            episodes=int(task["test_episodes"]),
            seed=int(task["seed"]) + 10_000,
            scenario_set=scenario_set,
            noise_std=float(task["noise_std"]),
        )
        save_evaluation_bundle(
            out_dir=out_dir,
            prefix=prefix,
            episode_rows=ep_rows,
            summary=bundle_summary,
            policy_rows=bundle_policy_rows,
            policy_summary_payload=bundle_policy_summary,
            env_switch_time=switch_time,
        )
        transfer_outputs[scenario_set] = bundle_summary
    return {
        "agent_name": agent_name,
        "best_row": best_row,
        "canonical_summary": summary,
        "transfer_summaries": transfer_outputs,
        "out_dir": str(out_dir),
    }


def _run_stage_eval_tasks(
    *,
    stage_name: str,
    tasks: list[dict[str, Any]],
    parallel_workers: int,
    worker_torch_threads: int,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    max_workers = max(1, min(int(parallel_workers), len(tasks)))
    if max_workers <= 1:
        _worker_runtime_init(worker_torch_threads)
        results = []
        for idx, task in enumerate(tasks, start=1):
            _log(f"{stage_name} {idx}/{len(tasks)} | {task['agent_name']}")
            payload = _saved_eval_task(task)
            results.append(payload)
            _log(f"[done {idx}/{len(tasks)}] eval {task['agent_name']}")
        return results

    _log(f"{stage_name} | launching {len(tasks)} eval jobs with up to {max_workers} worker processes")
    results: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_runtime_init,
        initargs=(int(worker_torch_threads),),
    ) as executor:
        future_map = {
            executor.submit(_saved_eval_task, task): str(task["agent_name"])
            for task in tasks
        }
        for done_idx, future in enumerate(cf.as_completed(future_map), start=1):
            agent_name = future_map[future]
            payload = future.result()
            results.append(payload)
            _log(f"[done {done_idx}/{len(tasks)}] eval {agent_name}")
    return results


def _canonical_env_kwargs(args) -> dict[str, Any]:
    return replica_env_kwargs_from_args(
        args,
        episode_duration=float(cfg.PAPER_EPISODE_DURATION),
        env_switch_time=float(cfg.PAPER_ENV_SWITCH_TIME),
    )


def _run_baselines(suite_root: Path, args, env_kwargs: dict[str, Any]) -> dict[str, Any]:
    stage_dir = suite_root / "00b"
    _log("stage baselines")
    if args.skip_existing and stage_completed(stage_dir):
        _log(f"skip stage baselines: {stage_dir}")
        with open(stage_dir / "study_manifest.json", "r", encoding="utf-8") as fh:
            return json.load(fh)

    ql_variant = get_ql_state_variant("Q0_baseline_reduced4")
    dqn_variant = get_dqn_state_variant("S0_baseline_full10")
    reward_variant = baseline_reward_variant()

    tasks = [
        {
            "order": 1,
            "name": "QL_baseline_replica_60s30s",
            "family": "ql",
            "out_dir": str(stage_dir / "ql"),
            "env_mode": args.env_mode,
            "env_kwargs": env_kwargs,
            "state_variant": ql_variant.name,
            "reward_variant": reward_variant.name,
            "total_episodes": args.q_episodes,
            "test_episodes": args.test_episodes,
            "seed": args.seed,
            "label": "QL_baseline_replica_60s30s",
            "resume": args.resume,
        },
        {
            "order": 2,
            "name": "DQN_baseline_replica_60s30s",
            "family": "dqn",
            "out_dir": str(stage_dir / "dqn"),
            "env_mode": args.env_mode,
            "env_kwargs": env_kwargs,
            "state_variant": dqn_variant.name,
            "reward_variant": reward_variant.name,
            "total_episodes": args.dqn_episodes,
            "test_episodes": args.test_episodes,
            "seed": args.seed,
            "label": "DQN_baseline_replica_60s30s",
            "resume": args.resume,
            "parallel_envs": int(args.dqn_parallel_envs),
        },
    ]
    payloads = _run_stage_train_tasks(
        stage_name="baselines",
        tasks=tasks,
        parallel_workers=min(int(args.parallel_workers), 2),
        worker_torch_threads=int(args.worker_torch_threads),
    )
    ql_summary = payloads[0]["summary"]
    dqn_summary = payloads[1]["summary"]
    rows = [
        _row_from_summary("ql", "baselines", "QL_baseline_replica_60s30s", ql_summary),
        _row_from_summary("dqn", "baselines", "DQN_baseline_replica_60s30s", dqn_summary),
    ]
    manifest = {
        "stage": "baselines",
        "rows": rows,
        "ql_baseline": rows[0],
        "dqn_baseline": rows[1],
    }
    _write_stage_outputs(stage_dir, manifest, rows)
    _log(
        "baselines complete | "
        f"QL transp={rows[0]['transparency_rmse_w']:.4f}, "
        f"DQN transp={rows[1]['transparency_rmse_w']:.4f}"
    )
    return manifest


def _run_ql_state_study(suite_root: Path, args, env_kwargs: dict[str, Any], ql_baseline: dict[str, Any]) -> dict[str, Any]:
    stage_dir = suite_root / "10qs"
    _log("stage ql_state")
    if args.skip_existing and stage_completed(stage_dir):
        _log(f"skip stage ql_state: {stage_dir}")
        with open(stage_dir / "study_manifest.json", "r", encoding="utf-8") as fh:
            return json.load(fh)
    reward_variant = baseline_reward_variant()
    tasks = [
        {
            "order": idx,
            "name": variant.name,
            "family": "ql",
            "out_dir": str(stage_dir / _short_variant_dir(variant.name)),
            "env_mode": args.env_mode,
            "env_kwargs": env_kwargs,
            "state_variant": variant.name,
            "reward_variant": reward_variant.name,
            "total_episodes": args.q_episodes,
            "test_episodes": args.test_episodes,
            "seed": args.seed,
            "label": variant.name,
            "resume": args.resume,
        }
        for idx, variant in enumerate(build_ql_state_variants(), start=1)
    ]
    payloads = _run_stage_train_tasks(
        stage_name="ql_state",
        tasks=tasks,
        parallel_workers=int(args.parallel_workers),
        worker_torch_threads=int(args.worker_torch_threads),
    )
    rows = [_row_from_summary("ql", "ql_state_rescue", payload["name"], payload["summary"]) for payload in payloads]
    best = _select_best(rows, baseline_tracking_rmse_m=float(ql_baseline["tracking_rmse_m"]))
    manifest = {"stage": "ql_state_rescue", "rows": rows, "best": best}
    _write_stage_outputs(stage_dir, manifest, rows)
    _log(
        "ql_state complete | "
        f"best={best['state_variant']} transp={best['transparency_rmse_w']:.4f} "
        f"track={best['tracking_rmse_m']:.4f}"
    )
    return manifest


def _run_ql_reward_study(suite_root: Path, args, env_kwargs: dict[str, Any], ql_baseline: dict[str, Any], best_state_name: str) -> dict[str, Any]:
    stage_dir = suite_root / "20qr"
    _log(f"stage ql_reward | state={best_state_name}")
    if args.skip_existing and stage_completed(stage_dir):
        _log(f"skip stage ql_reward: {stage_dir}")
        with open(stage_dir / "study_manifest.json", "r", encoding="utf-8") as fh:
            return json.load(fh)
    state_variant = get_ql_state_variant(best_state_name)
    reward_variants = build_full_reward_variants() if args.full_grid else build_core_reward_variants()
    tasks = [
        {
            "order": idx,
            "name": reward_variant.name,
            "family": "ql",
            "out_dir": str(stage_dir / _short_variant_dir(reward_variant.name)),
            "env_mode": args.env_mode,
            "env_kwargs": env_kwargs,
            "state_variant": state_variant.name,
            "reward_variant": reward_variant.name,
            "total_episodes": args.q_episodes,
            "test_episodes": args.test_episodes,
            "seed": args.seed,
            "label": f"{state_variant.name}_{reward_variant.name}",
            "resume": args.resume,
        }
        for idx, reward_variant in enumerate(reward_variants, start=1)
    ]
    payloads = _run_stage_train_tasks(
        stage_name="ql_reward",
        tasks=tasks,
        parallel_workers=int(args.parallel_workers),
        worker_torch_threads=int(args.worker_torch_threads),
    )
    rows = [_row_from_summary("ql", "ql_reward_study", payload["name"], payload["summary"]) for payload in payloads]
    best = _select_best(rows, baseline_tracking_rmse_m=float(ql_baseline["tracking_rmse_m"]))
    manifest = {"stage": "ql_reward_study", "rows": rows, "best": best, "state_variant": best_state_name}
    _write_stage_outputs(stage_dir, manifest, rows)
    _log(
        "ql_reward complete | "
        f"best={best['reward_variant']} transp={best['transparency_rmse_w']:.4f} "
        f"track={best['tracking_rmse_m']:.4f}"
    )
    return manifest


def _run_dqn_reward_study(suite_root: Path, args, env_kwargs: dict[str, Any], dqn_baseline: dict[str, Any]) -> dict[str, Any]:
    stage_dir = suite_root / "30dr"
    _log("stage dqn_reward")
    if args.skip_existing and stage_completed(stage_dir):
        _log(f"skip stage dqn_reward: {stage_dir}")
        with open(stage_dir / "study_manifest.json", "r", encoding="utf-8") as fh:
            return json.load(fh)
    state_variant = get_dqn_state_variant("S0_baseline_full10")
    reward_variants = build_full_reward_variants() if args.full_grid else build_core_reward_variants()
    tasks = [
        {
            "order": idx,
            "name": reward_variant.name,
            "family": "dqn",
            "out_dir": str(stage_dir / _short_variant_dir(reward_variant.name)),
            "env_mode": args.env_mode,
            "env_kwargs": env_kwargs,
            "state_variant": state_variant.name,
            "reward_variant": reward_variant.name,
            "total_episodes": args.dqn_episodes,
            "test_episodes": args.test_episodes,
            "seed": args.seed,
            "label": f"{state_variant.name}_{reward_variant.name}",
            "resume": args.resume,
            "parallel_envs": int(args.dqn_parallel_envs),
        }
        for idx, reward_variant in enumerate(reward_variants, start=1)
    ]
    payloads = _run_stage_train_tasks(
        stage_name="dqn_reward",
        tasks=tasks,
        parallel_workers=int(args.parallel_workers),
        worker_torch_threads=int(args.worker_torch_threads),
    )
    rows = [_row_from_summary("dqn", "dqn_reward_study", payload["name"], payload["summary"]) for payload in payloads]
    best = _select_best(rows, baseline_tracking_rmse_m=float(dqn_baseline["tracking_rmse_m"]))
    manifest = {"stage": "dqn_reward_study", "rows": rows, "best": best, "state_variant": state_variant.name}
    _write_stage_outputs(stage_dir, manifest, rows)
    _log(
        "dqn_reward complete | "
        f"best={best['reward_variant']} transp={best['transparency_rmse_w']:.4f} "
        f"track={best['tracking_rmse_m']:.4f}"
    )
    return manifest


def _run_dqn_state_study(suite_root: Path, args, env_kwargs: dict[str, Any], dqn_baseline: dict[str, Any], best_reward_name: str) -> dict[str, Any]:
    stage_dir = suite_root / "40ds"
    _log(f"stage dqn_state | reward={best_reward_name}")
    if args.skip_existing and stage_completed(stage_dir):
        _log(f"skip stage dqn_state: {stage_dir}")
        with open(stage_dir / "study_manifest.json", "r", encoding="utf-8") as fh:
            return json.load(fh)
    reward_variant = next(variant for variant in build_full_reward_variants() if variant.name == best_reward_name)
    tasks = [
        {
            "order": idx,
            "name": state_variant.name,
            "family": "dqn",
            "out_dir": str(stage_dir / _short_variant_dir(state_variant.name)),
            "env_mode": args.env_mode,
            "env_kwargs": env_kwargs,
            "state_variant": state_variant.name,
            "reward_variant": reward_variant.name,
            "total_episodes": args.dqn_episodes,
            "test_episodes": args.test_episodes,
            "seed": args.seed,
            "label": f"{state_variant.name}_{reward_variant.name}",
            "resume": args.resume,
            "parallel_envs": int(args.dqn_parallel_envs),
        }
        for idx, state_variant in enumerate(build_dqn_state_variants(), start=1)
    ]
    payloads = _run_stage_train_tasks(
        stage_name="dqn_state",
        tasks=tasks,
        parallel_workers=int(args.parallel_workers),
        worker_torch_threads=int(args.worker_torch_threads),
    )
    rows = [_row_from_summary("dqn", "dqn_state_ablation", payload["name"], payload["summary"]) for payload in payloads]
    best = _select_best(rows, baseline_tracking_rmse_m=float(dqn_baseline["tracking_rmse_m"]))
    manifest = {"stage": "dqn_state_ablation", "rows": rows, "best": best, "reward_variant": best_reward_name}
    _write_stage_outputs(stage_dir, manifest, rows)
    _log(
        "dqn_state complete | "
        f"best={best['state_variant']} transp={best['transparency_rmse_w']:.4f} "
        f"track={best['tracking_rmse_m']:.4f}"
    )
    return manifest


def _run_saved_policy_evals(suite_root: Path, args, ql_best: dict[str, Any], dqn_best: dict[str, Any]) -> dict[str, Any]:
    stage_dir = suite_root / "50ev"
    _log("stage eval")
    if args.skip_existing and stage_completed(stage_dir):
        _log(f"skip stage eval: {stage_dir}")
        with open(stage_dir / "study_manifest.json", "r", encoding="utf-8") as fh:
            return json.load(fh)

    tasks = [
        {
            "agent_name": "ql",
            "best_row": ql_best,
            "out_dir": str(stage_dir / "ql"),
            "test_episodes": args.test_episodes,
            "seed": args.seed + 40_000,
            "noise_std": args.noise_std,
        },
        {
            "agent_name": "dqn",
            "best_row": dqn_best,
            "out_dir": str(stage_dir / "dqn"),
            "test_episodes": args.test_episodes,
            "seed": args.seed + 50_000,
            "noise_std": args.noise_std,
        },
    ]
    payloads = _run_stage_eval_tasks(
        stage_name="eval",
        tasks=tasks,
        parallel_workers=min(int(args.parallel_workers), 2),
        worker_torch_threads=int(args.worker_torch_threads),
    )
    outputs: dict[str, Any] = {str(payload["agent_name"]): payload for payload in payloads}

    manifest = {"stage": "best_policy_evals", "agents": outputs}
    save_json(stage_dir / "study_manifest.json", manifest)
    stage_summary_rows_to_csv(
        [
            {
                "agent": agent_name,
                "study_family": "best_policy_evals",
                "variant_name": outputs[agent_name]["best_row"]["variant_name"],
                "state_variant": outputs[agent_name]["best_row"]["state_variant"],
                "reward_variant": outputs[agent_name]["best_row"]["reward_variant"],
                "tracking_rmse_m": outputs[agent_name]["best_row"]["tracking_rmse_m"],
                "transparency_rmse_w": outputs[agent_name]["best_row"]["transparency_rmse_w"],
                "pre_switch_tracking_rmse_m": outputs[agent_name]["best_row"]["pre_switch_tracking_rmse_m"],
                "post_switch_tracking_rmse_m": outputs[agent_name]["best_row"]["post_switch_tracking_rmse_m"],
                "pre_switch_transparency_rmse_w": outputs[agent_name]["best_row"]["pre_switch_transparency_rmse_w"],
                "post_switch_transparency_rmse_w": outputs[agent_name]["best_row"]["post_switch_transparency_rmse_w"],
                "mean_reward": outputs[agent_name]["best_row"]["mean_reward"],
                "invalid_episode_rate": outputs[agent_name]["best_row"]["invalid_episode_rate"],
                "model_path": outputs[agent_name]["best_row"]["model_path"],
                "out_dir": outputs[agent_name]["out_dir"],
            }
            for agent_name in ("ql", "dqn")
        ],
        stage_dir / "study_summary.csv",
    )
    _log("eval complete")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run replica-only RL baselines, ablations, and saved-policy evals in series.")
    parser.add_argument("--study-name", default="replica_rl")
    parser.add_argument("--stage", choices=["all", "baselines", "ql_state", "ql_reward", "dqn_reward", "dqn_state", "eval"], default="all")
    parser.add_argument("--env-mode", choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING], default=cfg.ENV_MODE_CHANGING)
    parser.add_argument("--episode-duration", type=float, default=float(cfg.PAPER_EPISODE_DURATION))
    parser.add_argument("--env-switch-time", type=float, default=float(cfg.PAPER_ENV_SWITCH_TIME))
    parser.add_argument("--force-amp", type=float, default=5.0)
    parser.add_argument("--force-bias", type=float, default=5.0)
    parser.add_argument("--force-freq", type=float, default=cfg.FORCE_INPUT_FREQ)
    parser.add_argument("--force-freq-rad", type=float, default=0.5)
    parser.add_argument("--force-phase", type=float, default=0.0)
    parser.add_argument("--force-waveform", choices=["sine", "cosine", "square", "ramp", "multisine"], default="sine")
    parser.add_argument("--fe-mode", choices=FE_MODE_CHOICES, default=FE_MODE_GUI)
    parser.add_argument("--q-episodes", type=int, default=DEFAULT_SWEEP_Q_EPISODES)
    parser.add_argument("--dqn-episodes", type=int, default=DEFAULT_SWEEP_DQN_EPISODES)
    parser.add_argument("--dqn-parallel-envs", type=int, default=1)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    args = parser.parse_args()
    worker_torch_threads = _configure_process_env(int(args.worker_torch_threads))

    suite_root = results_root(args.fe_mode) / args.study_name
    suite_root.mkdir(parents=True, exist_ok=True)
    env_kwargs = _canonical_env_kwargs(args)
    _log(f"study_root={suite_root}")
    _log(f"canonical_env={env_kwargs}")
    _log(
        f"parallel_workers={int(args.parallel_workers)} | "
        f"worker_torch_threads={worker_torch_threads}"
    )
    save_json(
        suite_root / "suite_manifest.json",
        {
            "study_name": args.study_name,
            "stage": args.stage,
            "env_kwargs": env_kwargs,
            "q_episodes": args.q_episodes,
            "dqn_episodes": args.dqn_episodes,
            "dqn_parallel_envs": int(args.dqn_parallel_envs),
            "test_episodes": args.test_episodes,
            "seed": args.seed,
            "full_grid": bool(args.full_grid),
            "fe_mode": str(args.fe_mode),
            "parallel_workers": int(args.parallel_workers),
            "worker_torch_threads": int(worker_torch_threads),
        },
    )

    baselines = _run_baselines(suite_root, args, env_kwargs)
    if args.stage == "baselines":
        return
    ql_state = _run_ql_state_study(suite_root, args, env_kwargs, baselines["ql_baseline"])
    if args.stage == "ql_state":
        return
    ql_reward = _run_ql_reward_study(suite_root, args, env_kwargs, baselines["ql_baseline"], ql_state["best"]["state_variant"])
    if args.stage == "ql_reward":
        return
    dqn_reward = _run_dqn_reward_study(suite_root, args, env_kwargs, baselines["dqn_baseline"])
    if args.stage == "dqn_reward":
        return
    dqn_state = _run_dqn_state_study(suite_root, args, env_kwargs, baselines["dqn_baseline"], dqn_reward["best"]["reward_variant"])
    if args.stage == "dqn_state":
        return
    _run_saved_policy_evals(suite_root, args, ql_reward["best"], dqn_state["best"])


if __name__ == "__main__":
    main()
