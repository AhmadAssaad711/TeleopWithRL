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
        plot_dqn_policy_slices,
    )
    from TeleopWithRL.matlab_literal_env.studies.dqn import build_dqn_env_factory, evaluate_dqn
    from TeleopWithRL.matlab_literal_env.studies.dqn_state_variants import get_dqn_state_variant
    from TeleopWithRL.matlab_literal_env.studies.rewarding import reward_variant_from_name
    from TeleopWithRL.matlab_literal_env.studies.saved_policy_eval import resolve_model_path
else:
    from ... import config as cfg
    from ...dqn_agent import DQNAgent
    from ..studies.common import save_common_visuals, save_history_npz, save_json, plot_dqn_policy_slices
    from ..studies.dqn import build_dqn_env_factory, evaluate_dqn
    from ..studies.dqn_state_variants import get_dqn_state_variant
    from ..studies.rewarding import reward_variant_from_name
    from ..studies.saved_policy_eval import resolve_model_path


def _load_summary(model_path: Path) -> dict:
    summary_path = model_path.parent.parent / "l" / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reevaluate an existing DQN run and regenerate averaged evaluation artifacts without retraining."
    )
    parser.add_argument("--run-dir", required=True, help="Path to the DQN run directory or model file.")
    parser.add_argument("--test-episodes", type=int, default=None, help="Number of evaluation episodes. Defaults to the run summary value.")
    parser.add_argument("--parallel-envs", type=int, default=8, help="Parallel env workers for reevaluation.")
    parser.add_argument("--seed-offset", type=int, default=20_000, help="Seed offset for evaluation resets.")
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
    env_factory = build_dqn_env_factory(
        env_mode=str(summary["env_mode"]),
        env_kwargs=env_kwargs,
        reward_variant=reward_variant,
        state_variant=state_variant,
    )

    agent = DQNAgent(obs_dim=state_variant.obs_dim, n_actions=cfg.N_ACTIONS, seed=42)
    agent.load(str(model_path))

    test_episodes = int(summary.get("test_episodes", 100) if args.test_episodes is None else args.test_episodes)
    parallel_envs = max(1, min(int(args.parallel_envs), test_episodes))

    eval_metrics, history = evaluate_dqn(
        agent,
        env_factory,
        n_episodes=test_episodes,
        seed_offset=int(args.seed_offset),
        parallel_envs=parallel_envs,
    )

    run_dir = model_path.parent.parent
    plots_dir = run_dir / "p"
    episodes_dir = run_dir / "e"
    logs_dir = run_dir / "l"

    save_history_npz(history, episodes_dir / "test_episode.npz")
    save_common_visuals(history, plots_dir, str(summary["label"]), env_switch_time=float(summary["env_switch_time"]))
    plot_dqn_policy_slices(agent, history, state_variant, plots_dir / "slices.png")

    payload = {
        "model_path": str(model_path),
        "label": str(summary["label"]),
        "reward_variant": str(summary["reward_variant"]),
        "state_variant": str(summary["state_variant"]),
        "test_episodes": int(test_episodes),
        "parallel_envs": int(parallel_envs),
        "evaluation_history_mode": "mean_over_test_episodes",
        **eval_metrics,
    }
    save_json(logs_dir / "reeval_summary.json", payload)
    with open(logs_dir / "reeval_summary.txt", "w", encoding="utf-8") as fh:
        for key, value in payload.items():
            fh.write(f"{key}={value}\n")

    print(f"reevaluated_model={model_path}")
    print(f"updated_history={episodes_dir / 'test_episode.npz'}")
    print(f"updated_plots={plots_dir}")
    print(f"reeval_summary={logs_dir / 'reeval_summary.json'}")


if __name__ == "__main__":
    main()
