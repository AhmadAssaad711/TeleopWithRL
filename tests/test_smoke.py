"""Quick smoke test for pneumatic simulation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from teleop_env import TeleopEnv

env = TeleopEnv()
obs, info = env.reset()
print(f"Reset OK  —  obs: {obs}")
print(f"Info: {info}")

# Single step
obs2, r, term, trunc, info2 = env.step(4)  # u_v = 0
print(f"\nStep 1 (u_v=0V)  —  obs: {obs2}  reward: {r:.6f}")

# Run 50 RL steps, measure wall time
t0 = time.perf_counter()
total_r = 0.0
for i in range(50):
    obs, r, term, trunc, _ = env.step(np.random.randint(9))
    total_r += r
elapsed = time.perf_counter() - t0
print(f"\n50 RL steps in {elapsed:.3f}s  ({elapsed/50*1000:.1f} ms/step)")
print(f"  Total reward: {total_r:.4f}")
print(f"  Final state:  x_m={env.state[0]*1000:.2f}mm  x_s={env.state[2]*1000:.2f}mm")
print(f"  Pressures:    P_m1={env.state[4]/1000:.1f}kPa  P_s1={env.state[6]/1000:.1f}kPa")

# Full episode speed estimate
steps_per_sec = 50 / elapsed
est_ep = 1500 / steps_per_sec
print(f"\n  Estimated episode time: {est_ep:.1f}s")
print(f"  Estimated 10k episodes: {est_ep * 10000 / 3600:.1f} hours")
print("\nSmoke test PASSED")
