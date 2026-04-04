"""Basic regression tests for environment reset, stepping, and state bins."""

from __future__ import annotations

import numpy as np

from TeleopWithRL import config as cfg
from TeleopWithRL.teleop_env import TeleopEnv


def test_env_reset_and_zero_voltage_step() -> None:
    env = TeleopEnv(master_input_mode=cfg.DEFAULT_MASTER_INPUT_MODE)
    obs, info = env.reset(seed=0)

    assert obs.shape == (10,)
    assert np.isfinite(obs).all()
    assert info["time"] == 0.0
    assert info["master_input_mode"] == cfg.DEFAULT_MASTER_INPUT_MODE

    next_obs, reward, terminated, truncated, next_info = env.step_voltage(0.0)

    assert next_obs.shape == (10,)
    assert np.isfinite(next_obs).all()
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert next_info["u_v"] == 0.0
    assert next_info["time"] > 0.0


def test_reduced_state_discretisation_shape() -> None:
    env = TeleopEnv(master_input_mode=cfg.DEFAULT_MASTER_INPUT_MODE)
    obs, _ = env.reset(seed=1)

    state = env.discretise_obs_reduced(obs)
    dims = env.get_state_dims_reduced()

    assert len(state) == 4
    assert len(dims) == 4
    assert all(isinstance(index, int) for index in state)
    assert all(size > 1 for size in dims)
