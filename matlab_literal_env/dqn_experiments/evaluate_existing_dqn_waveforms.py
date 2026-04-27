from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL import config as cfg
    from TeleopWithRL.dqn_agent import DQNAgent
    from TeleopWithRL.matlab_literal_env.studies.common import (
        save_common_visuals,
        save_history_npz,
        save_json,
        stage_summary_rows_to_csv,
    )
    from TeleopWithRL.matlab_literal_env.studies.dqn import build_dqn_env_factory, evaluate_dqn
    from TeleopWithRL.matlab_literal_env.studies.dqn_state_variants import get_dqn_state_variant
    from TeleopWithRL.matlab_literal_env.studies.rewarding import reward_variant_from_name
    from TeleopWithRL.matlab_literal_env.studies.saved_policy_eval import resolve_model_path
    from TeleopWithRL.matlab_literal_env.dqn_experiments.waveform_suite import parse_waveform_forms, suite_reset_options
else:
    from ... import config as cfg
    from ...dqn_agent import DQNAgent
    from ..studies.common import save_common_visuals, save_history_npz, save_json, stage_summary_rows_to_csv
    from ..studies.dqn import build_dqn_env_factory, evaluate_dqn
    from ..studies.dqn_state_variants import get_dqn_state_variant
    from ..studies.rewarding import reward_variant_from_name
    from ..studies.saved_policy_eval import resolve_model_path
    from .waveform_suite import parse_waveform_forms, suite_reset_options


def _load_summary(model_path: Path) -> dict:
    summary_path = model_path.parent.parent / "l" / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing DQN model on a suite of waveform forms."
    )
    parser.add_argument("--run-dir", required=True, help="Path to the DQN run directory or model file.")
    parser.add_argument("--waveforms", default="sine,cosine,square,ramp,multisine")
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--parallel-envs", type=int, default=8)
    parser.add_argument("--force-amp", type=float, default=5.0)
    parser.add_argument("--force-bias", type=float, default=5.0)
    parser.add_argument("--force-freq-rad", type=float, default=0.5)
    parser.add_argument("--force-phase", type=float, default=0.0)
    parser.add_argument("--seed-offset", type=int, default=30_000)
    args = parser.parse_args()

    model_path = resolve_model_path(Path(args.run_dir))
    summary = _load_summary(model_path)
    if str(summary.get("family")) != "dqn":
        raise ValueError(f"Expected a DQN run, got family={summary.get('family')!r}")

    reward_variant = reward_variant_from_name(str(summary["reward_variant"]))
    state_variant = get_dqn_state_variant(str(summary["state_variant"]))
    env_kwargs = {
        "episode_duration": float(summary["episode_duration"]),
        "env_switch_time": float(summary["env_switch_time"]),
        "terminate_on_error": bool(summary["terminate_on_error"]),
        "reset_options": dict(summary.get("reset_options", {})),
    }

    agent = DQNAgent(obs_dim=state_variant.obs_dim, n_actions=cfg.N_ACTIONS, seed=42)
    agent.load(str(model_path))

    waveforms = parse_waveform_forms(args.waveforms)
    suite = suite_reset_options(
        waveforms=waveforms,
        force_amp=float(args.force_amp),
        force_bias=float(args.force_bias),
        force_freq_rad=float(args.force_freq_rad),
        force_phase=float(args.force_phase),
    )

    run_dir = model_path.parent.parent
    out_root = run_dir / "wv"
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for idx, item in enumerate(suite):
        waveform = str(item["name"])
        waveform_dir_name = str(item.get("dir_name", waveform))
        reset_options = dict(env_kwargs["reset_options"])
        reset_options.update(dict(item["reset_options"]))
        waveform_env_kwargs = dict(env_kwargs)
        waveform_env_kwargs["reset_options"] = reset_options
        env_factory = build_dqn_env_factory(
            env_mode=str(summary["env_mode"]),
            env_kwargs=waveform_env_kwargs,
            reward_variant=reward_variant,
            state_variant=state_variant,
        )
        metrics, history = evaluate_dqn(
            agent,
            env_factory,
            n_episodes=int(args.test_episodes),
            seed_offset=int(args.seed_offset) + (1_000 * idx),
            parallel_envs=max(1, min(int(args.parallel_envs), int(args.test_episodes))),
        )
        waveform_dir = out_root / waveform_dir_name
        waveform_dir.mkdir(parents=True, exist_ok=True)
        save_history_npz(history, waveform_dir / "test_episode.npz")
        save_common_visuals(history, waveform_dir, f"{summary['label']}_{waveform}", env_switch_time=float(summary["env_switch_time"]))
        payload = {
            "waveform": waveform,
            "waveform_dir": waveform_dir_name,
            "model_path": str(model_path),
            "label": str(summary["label"]),
            "test_episodes": int(args.test_episodes),
            "force_amp": float(args.force_amp),
            "force_bias": float(args.force_bias),
            "force_freq_rad": float(args.force_freq_rad),
            "force_phase": float(args.force_phase),
            **metrics,
        }
        save_json(waveform_dir / "summary.json", payload)
        rows.append(payload)

    stage_summary_rows_to_csv(rows, out_root / "waveform_summary.csv")
    save_json(
        out_root / "waveform_manifest.json",
        {
            "run_dir": str(run_dir),
            "model_path": str(model_path),
            "waveforms": waveforms,
            "waveform_dirs": {str(item["name"]): str(item.get("dir_name", item["name"])) for item in suite},
            "rows": rows,
        },
    )
    print(f"waveform_eval_root={out_root}")
    print(f"waveform_summary={out_root / 'waveform_summary.csv'}")


if __name__ == "__main__":
    main()
