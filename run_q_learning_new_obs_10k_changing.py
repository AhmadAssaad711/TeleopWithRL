"""Run Q-learning training on the changing environment."""

from __future__ import annotations

import os

import config as cfg
from run_q_learning_new_obs_10k import TOTAL_TRAIN_EPISODES, run_q_learning_new_obs_10k


if __name__ == "__main__":
    run_q_learning_new_obs_10k(
        total_episodes=TOTAL_TRAIN_EPISODES,
        env_mode=cfg.ENV_MODE_CHANGING,
        out_dir_name=os.path.join(cfg.RL_CHANGING_DIR, "rl_new_obs_10k_episodes"),
    )
