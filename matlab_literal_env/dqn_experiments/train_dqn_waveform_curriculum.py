from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL import config as cfg
    from TeleopWithRL.dqn_agent import DQNAgent
    from TeleopWithRL.matlab_literal_env.simuoriginal_replica import FE_MODE_CHOICES, FE_MODE_DYNAMICS
    from TeleopWithRL.matlab_literal_env.scripts import run_replica_studies as runner
    from TeleopWithRL.matlab_literal_env.studies.common import (
        save_common_visuals,
        save_history_npz,
        save_json,
        stage_summary_rows_to_csv,
    )
    from TeleopWithRL.matlab_literal_env.studies.dqn import build_dqn_env_factory, evaluate_dqn, train_dqn_variant
    from TeleopWithRL.matlab_literal_env.studies.dqn_state_variants import get_dqn_state_variant
    from TeleopWithRL.matlab_literal_env.studies.rewarding import reward_variant_from_name
    from TeleopWithRL.matlab_literal_env.studies.saved_policy_eval import resolve_model_path
    from TeleopWithRL.matlab_literal_env.dqn_experiments.waveform_suite import (
        curriculum_schedule,
        parse_waveform_forms,
        parse_waveform_stages,
        suite_reset_options,
    )
else:
    from ... import config as cfg
    from ...dqn_agent import DQNAgent
    from ..simuoriginal_replica import FE_MODE_CHOICES, FE_MODE_DYNAMICS
    from ..scripts import run_replica_studies as runner
    from ..studies.common import save_common_visuals, save_history_npz, save_json, stage_summary_rows_to_csv
    from ..studies.dqn import build_dqn_env_factory, evaluate_dqn, train_dqn_variant
    from ..studies.dqn_state_variants import get_dqn_state_variant
    from ..studies.rewarding import reward_variant_from_name
    from ..studies.saved_policy_eval import resolve_model_path
    from .waveform_suite import curriculum_schedule, parse_waveform_forms, parse_waveform_stages, suite_reset_options


DEFAULT_DQN_PARALLEL_ENVS = max(1, min(8, 8))
DEFAULT_EPISODE_DURATION = 20.0
DEFAULT_ENV_SWITCH_TIME = 10.0


def _suite_root(fe_mode: str, study_name: str) -> Path:
    base_dir = Path(__file__).resolve().parent / "results"
    fe_dir = "dyn" if str(fe_mode) == FE_MODE_DYNAMICS else "gui"
    return base_dir / fe_dir / study_name


def _load_summary(model_path: Path) -> dict:
    import json

    summary_path = model_path.parent.parent / "l" / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a DQN on a waveform curriculum and evaluate it on the same waveform suite."
    )
    parser.add_argument("--study-name", default="dqn_waveform_curriculum_01")
    parser.add_argument("--env-mode", choices=[cfg.ENV_MODE_CONSTANT, cfg.ENV_MODE_CHANGING], default=cfg.ENV_MODE_CHANGING)
    parser.add_argument("--episode-duration", type=float, default=DEFAULT_EPISODE_DURATION)
    parser.add_argument("--env-switch-time", type=float, default=DEFAULT_ENV_SWITCH_TIME)
    parser.add_argument("--force-amp", type=float, default=5.0)
    parser.add_argument("--force-bias", type=float, default=5.0)
    parser.add_argument("--force-freq-rad", type=float, default=0.5)
    parser.add_argument("--force-phase", type=float, default=0.0)
    parser.add_argument("--fe-mode", choices=FE_MODE_CHOICES, default=FE_MODE_DYNAMICS)
    parser.add_argument("--reward-variant", default="baseline_cfg")
    parser.add_argument("--state-variant", default="S0_baseline_full10")
    parser.add_argument("--dqn-episodes", type=int, default=2500)
    parser.add_argument("--dqn-parallel-envs", type=int, default=DEFAULT_DQN_PARALLEL_ENVS)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--train-waveforms", default="sine,cosine,square,ramp,multisine")
    parser.add_argument("--train-waveform-stages", default="sine,cosine;multisine;square,ramp")
    parser.add_argument("--eval-waveforms", default="sine,cosine,square,ramp,multisine")
    parser.add_argument("--init-model", default=None, help="Warm-start from an existing DQN run directory or model file.")
    parser.add_argument("--curriculum-replay-min-size", type=int, default=4000)
    parser.add_argument("--curriculum-epsilon", type=float, default=0.20)
    parser.add_argument("--disable-terminate-on-error", action="store_true")
    args = parser.parse_args()

    args.legacy_baseline_env = True
    args.force_waveform = "sine"
    runner._configure_process_env(int(args.worker_torch_threads))

    suite_root = _suite_root(str(args.fe_mode), str(args.study_name))
    suite_root.mkdir(parents=True, exist_ok=True)
    env_kwargs = runner._canonical_env_kwargs(args)
    reward_variant = reward_variant_from_name(str(args.reward_variant))
    state_variant = get_dqn_state_variant(str(args.state_variant))

    train_waveforms = parse_waveform_forms(args.train_waveforms)
    train_stages = parse_waveform_stages(args.train_waveform_stages)
    eval_waveforms = parse_waveform_forms(args.eval_waveforms)
    curriculum_pool = [
        item["reset_options"]
        for item in suite_reset_options(
            waveforms=train_waveforms,
            force_amp=float(args.force_amp),
            force_bias=float(args.force_bias),
            force_freq_rad=float(args.force_freq_rad),
            force_phase=float(args.force_phase),
        )
    ]
    schedule = curriculum_schedule(
        total_episodes=int(args.dqn_episodes),
        stages=train_stages,
        force_amp=float(args.force_amp),
        force_bias=float(args.force_bias),
        force_freq_rad=float(args.force_freq_rad),
        force_phase=float(args.force_phase),
        rng_seed=int(args.seed) + 9_999,
    )
    init_model_path = None if args.init_model in (None, "") else resolve_model_path(Path(args.init_model))

    train_stage_dir = suite_root / "0t"
    train_dir = train_stage_dir / "dqn"
    result = train_dqn_variant(
        out_dir=train_dir,
        env_mode=str(args.env_mode),
        env_kwargs=env_kwargs,
        state_variant=state_variant,
        reward_variant=reward_variant,
        total_episodes=int(args.dqn_episodes),
        test_episodes=int(args.test_episodes),
        seed=int(args.seed),
        label=f"DQN_curriculum_{state_variant.name}_{reward_variant.name}",
        parallel_envs=int(args.dqn_parallel_envs),
        train_reset_options_pool=curriculum_pool,
        train_reset_options_schedule=schedule,
        init_model_path=init_model_path,
        replay_min_size_override=int(args.curriculum_replay_min_size),
        epsilon_after_load=float(args.curriculum_epsilon) if init_model_path is not None else None,
        decay_epsilon_on_learning_only=True,
    )

    train_summary = _load_summary(Path(result.model_path))
    train_row = runner._row_from_summary(
        "dqn",
        "waveform_curriculum_train",
        f"{state_variant.name}_{reward_variant.name}",
        train_summary,
    )
    stage_summary_rows_to_csv([train_row], train_stage_dir / "study_summary.csv")
    save_json(
        train_stage_dir / "study_manifest.json",
        {
            "stage": "waveform_curriculum_train",
            "train_waveforms": train_waveforms,
            "train_waveform_stages": train_stages,
            "init_model_path": None if init_model_path is None else str(init_model_path),
            "rows": [train_row],
        },
    )

    agent = DQNAgent(obs_dim=state_variant.obs_dim, n_actions=cfg.N_ACTIONS, seed=42)
    agent.load(str(result.model_path))
    wave_rows: list[dict[str, float | str]] = []
    waveforms_dir = suite_root / "1w"
    waveforms_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(
        suite_reset_options(
            waveforms=eval_waveforms,
            force_amp=float(args.force_amp),
            force_bias=float(args.force_bias),
            force_freq_rad=float(args.force_freq_rad),
            force_phase=float(args.force_phase),
        )
    ):
        waveform = str(item["name"])
        waveform_dir_name = str(item.get("dir_name", waveform))
        reset_options = dict(env_kwargs["reset_options"])
        reset_options.update(dict(item["reset_options"]))
        waveform_env_kwargs = dict(env_kwargs)
        waveform_env_kwargs["reset_options"] = reset_options
        env_factory = build_dqn_env_factory(
            env_mode=str(args.env_mode),
            env_kwargs=waveform_env_kwargs,
            reward_variant=reward_variant,
            state_variant=state_variant,
        )
        metrics, history = evaluate_dqn(
            agent,
            env_factory,
            n_episodes=int(args.test_episodes),
            seed_offset=40_000 + (1_000 * idx),
            parallel_envs=max(1, min(int(args.dqn_parallel_envs), int(args.test_episodes))),
        )
        waveform_dir = waveforms_dir / waveform_dir_name
        waveform_dir.mkdir(parents=True, exist_ok=True)
        save_history_npz(history, waveform_dir / "test_episode.npz")
        save_common_visuals(history, waveform_dir, f"{train_summary['label']}_{waveform}", env_switch_time=float(args.env_switch_time))
        payload = {
            "waveform": waveform,
            "waveform_dir": waveform_dir_name,
            "model_path": str(result.model_path),
            "label": str(train_summary["label"]),
            "train_waveforms": list(train_waveforms),
            "test_episodes": int(args.test_episodes),
            "force_amp": float(args.force_amp),
            "force_bias": float(args.force_bias),
            "force_freq_rad": float(args.force_freq_rad),
            "force_phase": float(args.force_phase),
            **metrics,
        }
        save_json(waveform_dir / "summary.json", payload)
        wave_rows.append(payload)

    stage_summary_rows_to_csv(wave_rows, waveforms_dir / "waveform_summary.csv")
    save_json(
        waveforms_dir / "waveform_manifest.json",
        {
            "stage": "waveform_curriculum_eval",
            "train_waveforms": train_waveforms,
            "eval_waveforms": eval_waveforms,
            "waveform_dirs": {str(item["name"]): str(item.get("dir_name", item["name"])) for item in suite_reset_options(
                waveforms=eval_waveforms,
                force_amp=float(args.force_amp),
                force_bias=float(args.force_bias),
                force_freq_rad=float(args.force_freq_rad),
                force_phase=float(args.force_phase),
            )},
            "rows": wave_rows,
        },
    )
    print(f"train_root={train_stage_dir}")
    print(f"waveform_eval_root={waveforms_dir}")
    print(f"waveform_summary={waveforms_dir / 'waveform_summary.csv'}")


if __name__ == "__main__":
    main()
