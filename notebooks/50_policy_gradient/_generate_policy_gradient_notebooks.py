from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


ALGO_SPECS = [
    {
        "filename": "51_ppo_continuous_baseline.ipynb",
        "algo_key": "ppo_continuous",
        "algo_label": "PPO Continuous",
        "algo_tag": "ppo",
        "run_dir": "ppo",
        "parallel_envs": 8,
    },
    {
        "filename": "52_td3_baseline.ipynb",
        "algo_key": "td3",
        "algo_label": "TD3",
        "algo_tag": "td3",
        "run_dir": "td3",
        "parallel_envs": 1,
    },
    {
        "filename": "53_sac_baseline.ipynb",
        "algo_key": "sac",
        "algo_label": "SAC",
        "algo_tag": "sac",
        "run_dir": "sac",
        "parallel_envs": 1,
    },
    {
        "filename": "54_ppo_discrete_baseline.ipynb",
        "algo_key": "ppo_discrete",
        "algo_label": "PPO Discrete",
        "algo_tag": "ppod",
        "run_dir": "ppod",
        "parallel_envs": 8,
    },
]


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else f"{line}\n" for line in text.splitlines()],
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else f"{line}\n" for line in text.splitlines()],
    }


def build_notebook(spec: dict) -> dict:
    algo_key = spec["algo_key"]
    algo_label = spec["algo_label"]
    algo_tag = spec["algo_tag"]
    run_dir = spec["run_dir"]
    parallel_envs = spec["parallel_envs"]

    cells = [
        md_cell(
            f"""# {algo_label} Baseline Runner

This notebook launches fresh `{algo_label}` baselines on `matlab_literal_env` using the **exact same teleoperation formulation** as the DQN notebook:

- `30 s` episode duration
- `10 s` skin-to-fat switch
- midpoint reset
- force input `5 N` amplitude, `15 N` bias, `6 rad/s`
- stroke limit handled with mechanical clamp mode
- state variant `S0_baseline_full10`
- reward variant `eqgrad_t40_tr40_nojerk`
- both FE modes: `switched_dynamics` and `gui_skin_locked`
"""
        ),
        md_cell("## Baseline Config"),
        code_cell(
            f"""import subprocess
import sys
from pathlib import Path

for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
    for root in (candidate, candidate / 'TeleopWithRL'):
        if (root / 'matlab_literal_env').exists() and (root / 'notebooks' / '_teleop_nb.py').exists():
            for path_to_add in (root.parent, root):
                if str(path_to_add) not in sys.path:
                    sys.path.insert(0, str(path_to_add))
            break
    else:
        continue
    break
else:
    raise RuntimeError('Could not find TeleopWithRL notebook root.')

from notebooks._teleop_nb import load_json, project_python_executable, repo_paths, show_image, show_rows
from TeleopWithRL.matlab_literal_env.policy_gradient_experiments.paths import suite_root as policy_gradient_suite_root

P = repo_paths()
REPO = P['repo']
WORKSPACE = REPO.parent
PYTHON = project_python_executable(REPO)
PG_RESULTS = REPO / 'matlab_literal_env' / 'policy_gradient_experiments' / 'results'

ALGO_KEY = '{algo_key}'
ALGO_LABEL = '{algo_label}'
ALGO_TAG = '{algo_tag}'
RUN_DIR = '{run_dir}'

CFG = {{
    'study_name': f'{{ALGO_TAG}}_eqg40_b01',
    'env_mode': 'changing_skin_fat',
    'episode_duration_s': 30.0,
    'env_switch_time_s': 10.0,
    'reset_position_mode': 'midpoint',
    'stroke_limit_mode': 'clamp',
    'force_amp_N': 5.0,
    'force_bias_N': 15.0,
    'force_freq_rad_s': 6.0,
    'force_phase_rad': 0.0,
    'force_waveform': 'sine',
    'reward_variant': 'eqgrad_t40_tr40_nojerk',
    'state_variant': 'S0_baseline_full10',
    'train_episodes': 2500,
    'parallel_envs': {parallel_envs},
    'eval_every_episodes': 100,
    'test_episodes': 100,
    'seed': 42,
    'parallel_workers': 1,
    'worker_torch_threads': 1,
    'skip_existing': True,
}}

MODE_LABELS = {{
    'dyn': 'switched_dynamics',
    'gui': 'gui_skin_locked',
}}

RUN_ROOTS = {{
    mode: policy_gradient_suite_root(MODE_LABELS[mode], f"{{CFG['study_name']}}_{{mode}}") / '00b' / RUN_DIR
    for mode in MODE_LABELS
}}

CMD = [
    str(PYTHON),
    '-m',
    'TeleopWithRL.matlab_literal_env.policy_gradient_experiments.run_policy_gradient_baselines_both_fe',
    '--algo', ALGO_KEY,
    '--study-name', CFG['study_name'],
    '--env-mode', CFG['env_mode'],
    '--episode-duration', str(CFG['episode_duration_s']),
    '--env-switch-time', str(CFG['env_switch_time_s']),
    '--reset-position-mode', CFG['reset_position_mode'],
    '--stroke-limit-mode', CFG['stroke_limit_mode'],
    '--force-amp', str(CFG['force_amp_N']),
    '--force-bias', str(CFG['force_bias_N']),
    '--force-freq-rad', str(CFG['force_freq_rad_s']),
    '--force-phase', str(CFG['force_phase_rad']),
    '--force-waveform', CFG['force_waveform'],
    '--reward-variant', CFG['reward_variant'],
    '--state-variant', CFG['state_variant'],
    '--train-episodes', str(CFG['train_episodes']),
    '--parallel-envs', str(CFG['parallel_envs']),
    '--eval-every-episodes', str(CFG['eval_every_episodes']),
    '--test-episodes', str(CFG['test_episodes']),
    '--seed', str(CFG['seed']),
    '--parallel-workers', str(CFG['parallel_workers']),
    '--worker-torch-threads', str(CFG['worker_torch_threads']),
]
if CFG['skip_existing']:
    CMD.append('--skip-existing')

show_rows(
    [{{
        'algo': ALGO_LABEL,
        'reward_variant': CFG['reward_variant'],
        'state_variant': CFG['state_variant'],
        'episode_duration_s': CFG['episode_duration_s'],
        'env_switch_time_s': CFG['env_switch_time_s'],
        'stroke_limit_mode': CFG['stroke_limit_mode'],
        'force_amp_N': CFG['force_amp_N'],
        'force_bias_N': CFG['force_bias_N'],
        'force_freq_rad_s': CFG['force_freq_rad_s'],
        'train_episodes': CFG['train_episodes'],
        'parallel_envs': CFG['parallel_envs'],
        'test_episodes': CFG['test_episodes'],
        'python_executable': str(PYTHON),
        'dyn_run_root': str(RUN_ROOTS['dyn']),
        'gui_run_root': str(RUN_ROOTS['gui']),
        'command': subprocess.list2cmdline(CMD),
    }}],
    title=f'{{ALGO_LABEL}} baseline run config',
    max_rows=10,
)"""
        ),
        code_cell(
            """print(subprocess.list2cmdline(CMD))
completed = subprocess.run(CMD, cwd=str(WORKSPACE), check=True)
print(f'Completed with return code {completed.returncode}.')"""
        ),
        md_cell("## Numeric Summary"),
        code_cell(
            """summary_rows = []
metric_rows = []
artifact_rows = []
for mode, run_root in RUN_ROOTS.items():
    summary_path = run_root / 'l' / 'summary.json'
    plots_dir = run_root / 'p'
    artifact_rows.append({'fe_mode': MODE_LABELS[mode], 'summary_json': str(summary_path), 'plots_dir': str(plots_dir)})
    if not summary_path.exists():
        summary_rows.append({'fe_mode': MODE_LABELS[mode], 'status': 'missing', 'run_root': str(run_root)})
        continue
    data = load_json(summary_path)
    reset_options = dict(data.get('reset_options', {}))
    summary_rows.append({
        'fe_mode': MODE_LABELS[mode],
        'algo': data.get('algo_display_name', data.get('algo')),
        'label': data.get('label'),
        'tracking_rmse_m': data.get('tracking_rmse_m'),
        'transparency_rmse_w': data.get('transparency_rmse_w'),
        'pre_switch_tracking_rmse_m': data.get('pre_switch_tracking_rmse_m'),
        'post_switch_tracking_rmse_m': data.get('post_switch_tracking_rmse_m'),
        'pre_switch_transparency_rmse_w': data.get('pre_switch_transparency_rmse_w'),
        'post_switch_transparency_rmse_w': data.get('post_switch_transparency_rmse_w'),
        'invalid_episode_rate': data.get('invalid_episode_rate'),
        'episode_duration': data.get('episode_duration'),
        'env_switch_time': data.get('env_switch_time'),
        'force_amp': reset_options.get('force_amp'),
        'force_bias': reset_options.get('force_bias'),
        'force_freq_rad': reset_options.get('force_freq_rad'),
        'reset_position_mode': reset_options.get('reset_position_mode'),
        'state_variant': data.get('state_variant'),
        'reward_variant': data.get('reward_variant'),
        'out_dir': data.get('out_dir'),
    })
    metric_rows.append({
        'fe_mode': MODE_LABELS[mode],
        'track_rmse_mm': round(1000.0 * float(data.get('tracking_rmse_m') or 0.0), 3),
        'transp_rmse_w': round(float(data.get('transparency_rmse_w') or 0.0), 3),
        'pre_track_rmse_mm': round(1000.0 * float(data.get('pre_switch_tracking_rmse_m') or 0.0), 3),
        'post_track_rmse_mm': round(1000.0 * float(data.get('post_switch_tracking_rmse_m') or 0.0), 3),
        'pre_transp_rmse_w': round(float(data.get('pre_switch_transparency_rmse_w') or 0.0), 3),
        'post_transp_rmse_w': round(float(data.get('post_switch_transparency_rmse_w') or 0.0), 3),
        'mean_reward': round(float(data.get('mean_reward') or 0.0), 3),
    })

show_rows(metric_rows, title=f'{ALGO_LABEL} rollout RMSE metrics', max_rows=10)
show_rows(summary_rows, title=f'{ALGO_LABEL} baseline summaries', max_rows=10)
show_rows(artifact_rows, title=f'{ALGO_LABEL} artifact locations', max_rows=10)"""
        ),
        md_cell("## Plot Gallery"),
        md_cell("### switched_dynamics"),
        code_cell(
            """plot_specs = [
    ('train.png', 'Training metrics'),
    ('roll.png', 'Evaluation roll plot'),
    ('act.png', 'Evaluation action plot'),
    ('traj.png', 'Evaluation trajectory plot'),
    ('slices.png', 'Policy slices'),
]

run_root = RUN_ROOTS['dyn']
for filename, title in plot_specs:
    show_image(run_root / 'p' / filename, title=f"{MODE_LABELS['dyn']}: {title}")"""
        ),
        md_cell("### gui_skin_locked"),
        code_cell(
            """run_root = RUN_ROOTS['gui']
for filename, title in plot_specs:
    show_image(run_root / 'p' / filename, title=f"{MODE_LABELS['gui']}: {title}")"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for spec in ALGO_SPECS:
        notebook = build_notebook(spec)
        out_path = ROOT / spec["filename"]
        out_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
