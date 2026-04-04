"""Regression checks for passive tube coupling in a deterministic skin case."""

from __future__ import annotations

import numpy as np

from TeleopWithRL import config as cfg
from TeleopWithRL.teleop_env import TeleopEnv


def test_zero_voltage_passive_coupling_moves_slave_without_instability() -> None:
    env = TeleopEnv(
        env_mode=cfg.ENV_MODE_CONSTANT,
        master_input_mode=cfg.MASTER_INPUT_FORCE,
        episode_duration=2.0,
        terminate_on_error=False,
    )
    env.reset(
        seed=0,
        options={
            "force_amp": 10.0,
            "force_freq": 0.5,
            "force_phase": 0.0,
            "force_waveform": "sine",
            "force_noise_std": 0.0,
        },
    )

    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step_voltage(0.0)
        done = terminated or truncated

    history = env.render() or {}
    x_m = np.asarray(history["x_m"], dtype=np.float64)
    x_s = np.asarray(history["x_s"], dtype=np.float64)
    pos_error = np.asarray(history["pos_error"], dtype=np.float64)
    p_m1 = np.asarray(history["P_m1"], dtype=np.float64)
    p_s1 = np.asarray(history["P_s1"], dtype=np.float64)

    assert x_m.size > 0
    assert x_s.size == x_m.size
    assert np.ptp(x_m) > 1e-3
    assert np.ptp(x_s) > 1e-4
    assert np.isfinite(pos_error).all()
    assert np.sqrt(np.mean(pos_error ** 2)) < 0.05
    assert p_m1.min() > 0.0
    assert p_s1.min() > 0.0
