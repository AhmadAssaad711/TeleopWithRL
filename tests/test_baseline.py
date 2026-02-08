"""Verify passive tube coupling and estimate training speed."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from teleop_env import TeleopEnv
import config as cfg

env = TeleopEnv()
obs, _ = env.reset()

# Force a deterministic environment for this test
env.fh_amp   = 10.0
env.fh_freq  = 0.5
env.fh_phase = 0.0
env.Be, env.Ke = cfg.SKIN_BE, cfg.SKIN_KE

# Run full episode with u_v = 0 (baseline — pure tube coupling)
zero_act = int(np.argmin(np.abs(env._action_table)))  # action index for 0V
print(f"Zero-voltage action index: {zero_act}  → u_v = {cfg.V_LEVELS[zero_act]} V")

t0 = time.perf_counter()
done = False
while not done:
    _, _, terminated, truncated, _ = env.step(zero_act)
    done = terminated or truncated
elapsed = time.perf_counter() - t0

h = env.render()
x_m = np.array(h["x_m"]) * 1000  # mm
x_s = np.array(h["x_s"]) * 1000
pe  = np.array(h["pos_error"]) * 1000
t   = np.array(h["time"])

print(f"\n=== Baseline Episode (u_v = 0, skin environment) ===")
print(f"  Steps: {len(t)},  Duration: {t[-1]:.1f}s")
print(f"  Wall time: {elapsed:.2f}s")
print(f"  x_m range: [{x_m.min():.1f}, {x_m.max():.1f}] mm")
print(f"  x_s range: [{x_s.min():.1f}, {x_s.max():.1f}] mm")
print(f"  Tracking RMSE: {np.sqrt(np.mean(pe**2)):.2f} mm")
print(f"  Max |error|:   {np.max(np.abs(pe)):.2f} mm")
print(f"  Correlation x_m vs x_s: {np.corrcoef(x_m, x_s)[0,1]:.4f}")

# Check pressures
P_m1 = np.array(h["P_m1"]) / 1000
P_s1 = np.array(h["P_s1"]) / 1000
print(f"  P_m1 range: [{P_m1.min():.1f}, {P_m1.max():.1f}] kPa")
print(f"  P_s1 range: [{P_s1.min():.1f}, {P_s1.max():.1f}] kPa")

# Estimate training time
ep_time = elapsed
print(f"\n=== Training Time Estimates ===")
for n_ep in [1000, 3000, 5000, 10000]:
    hrs = n_ep * ep_time / 3600
    print(f"  {n_ep:>6} episodes: {hrs:.1f} hours")

# Shorter episode estimate
print(f"\n  (With 10s episodes instead of {cfg.EPISODE_DURATION:.0f}s:)")
for n_ep in [1000, 3000, 5000]:
    hrs = n_ep * ep_time * (10 / cfg.EPISODE_DURATION) / 3600
    print(f"  {n_ep:>6} episodes: {hrs:.1f} hours")
