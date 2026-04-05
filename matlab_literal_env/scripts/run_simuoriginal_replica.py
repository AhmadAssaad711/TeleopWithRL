from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from TeleopWithRL.matlab_literal_env.simuoriginal_replica import (
    ParmsOriginal,
    SimuOriginalProfile,
    saved_force_input,
    simulate_simuoriginal_replica,
    write_simuoriginal_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a standalone nonlinear Python replica of SimuOriginal.slx."
    )
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("TeleopWithRL/matlab_literal_env/results/simuoriginal_replica_saved_open_loop"),
    )
    parser.add_argument("--force-amp", type=float, default=10.0)
    parser.add_argument("--force-bias", type=float, default=5.0)
    parser.add_argument("--force-freq-rad", type=float, default=0.5)
    parser.add_argument("--u-constant", type=float, default=0.0)
    parser.add_argument(
        "--valve-pressure-source",
        choices=["atm", "chamber"],
        default="atm",
        help="Mirror the saved manual-switch setting (`atm`) or use chamber pressure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parms = ParmsOriginal()
    profile = SimuOriginalProfile(
        fixed_step=parms.Ts,
        force_amplitude=args.force_amp,
        force_bias=args.force_bias,
        force_frequency_rad=args.force_freq_rad,
        valve_branch_1_pressure_source=args.valve_pressure_source,
        valve_branch_2_pressure_source=args.valve_pressure_source,
    )

    force_fn = lambda t: saved_force_input(t, profile)
    control_fn = lambda _t: args.u_constant

    result = simulate_simuoriginal_replica(
        duration=args.duration,
        parms=parms,
        profile=profile,
        F_h_fn=force_fn,
        u_fn=control_fn,
    )
    write_simuoriginal_result(result, args.out_dir)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(result.time, result.F_h, label="F_h")
    axes[0].plot(result.time, result.u, label="u")
    axes[0].legend(loc="best")
    axes[0].set_ylabel("Input")

    axes[1].plot(result.time, result.x_m, label="x_m")
    axes[1].plot(result.time, result.x_s, label="x_s")
    axes[1].legend(loc="best")
    axes[1].set_ylabel("Position [m]")

    axes[2].plot(result.time, result.Pm1, label="Pm1")
    axes[2].plot(result.time, result.Pm2, label="Pm2")
    axes[2].plot(result.time, result.Ps1, label="Ps1")
    axes[2].plot(result.time, result.Ps2, label="Ps2")
    axes[2].legend(loc="best", ncol=2)
    axes[2].set_ylabel("Pressure [Pa]")

    axes[3].plot(result.time, result.Fe, label="Fe")
    axes[3].plot(result.time, result.xm_dot, label="xm_dot")
    axes[3].plot(result.time, result.xs_dot, label="xs_dot")
    axes[3].legend(loc="best")
    axes[3].set_ylabel("Derived")
    axes[3].set_xlabel("Time [s]")

    fig.suptitle("SimuOriginal nonlinear Python replica")
    fig.tight_layout()
    fig.savefig(args.out_dir / "simuoriginal_replica_plot.png", dpi=150)
    plt.close(fig)

    print(f"Wrote results to {args.out_dir}")
    print(f"Exported duration: {result.time[-1]:.6f} s")
    print(f"Singularity time: {result.singularity_time}")


if __name__ == "__main__":
    main()
