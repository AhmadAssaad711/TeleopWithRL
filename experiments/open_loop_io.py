"""Analyze open-loop plant I/O for the pneumatic teleoperation environment.

Default case:
  F_h(t) = 5 + 10 * sin(0.5*t) N
  u(t)   = 0 V
  environment switches from skin to fat at t = 30 s
  total duration = 180 s

Outputs:
  - plant_io.csv
  - plant_io_debug_state.npz
  - plant_io_scopes.png
  - plant_io_error_scopes.png
  - plant_io_switch_zoom.png
  - plant_io_summary.txt

Position convention:
  x_m and x_s are exported as displacement from equilibrium so the Python
  traces share the MATLAB GUI's zero-centered position origin. Raw absolute
  positions are still preserved in the debug NPZ.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import config as cfg
from ..teleop_env import TeleopEnv


CSV_COLUMNS = (
    "t",
    "F_h",
    "u",
    "x_m",
    "x_s",
    "e",
    "Fe",
    "transparency_error",
    "x_mdot",
    "x_sdot",
    "env_label",
)


def _default_out_dir() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    return package_root / cfg.RESULTS_ROOT_DIR / "plant_io_analysis"


def _clip_voltage(env: TeleopEnv, u_v: float) -> float:
    low = float(env.action_space.low[0])
    high = float(env.action_space.high[0])
    return float(np.clip(float(u_v), low, high))


def _sample_row(env: TeleopEnv, obs: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
    state = env.state
    x_eq = float(info["x_eq"])
    x_m_abs = float(state[env.IX_XM])
    x_s_abs = float(state[env.IX_XS])
    x_m = float(info["x_m_centered"])
    x_s = float(info["x_s_centered"])
    v_m = float(state[env.IX_VM])
    v_s = float(state[env.IX_VS])
    e = x_m - x_s
    fe = float(info["F_e"])
    fh = float(info["F_h"])
    return {
        "t": float(info["time"]),
        "F_h": fh,
        "u": float(info["u_v"]),
        "x_m": x_m,
        "x_s": x_s,
        "e": e,
        "Fe": fe,
        "transparency_error": (fe * v_m) - (fh * v_s),
        "x_mdot": v_m,
        "x_sdot": v_s,
        "env_label": str(info["env_label"]),
        "P_m1": float(state[env.IX_PM1]),
        "P_m2": float(state[env.IX_PM2]),
        "P_s1": float(state[env.IX_PS1]),
        "P_s2": float(state[env.IX_PS2]),
        "x_eq": x_eq,
        "x_m_abs": x_m_abs,
        "x_s_abs": x_s_abs,
        "mdot_L1": float(state[env.IX_ML1]),
        "mdot_L2": float(state[env.IX_ML2]),
        "x_v": float(state[env.IX_XV]),
        "v_m": v_m,
        "v_s": v_s,
        "v_v": float(state[env.IX_VV]),
        "observation": np.asarray(obs, dtype=np.float32).copy(),
        "state": np.asarray(state, dtype=np.float64).copy(),
    }


def _rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key == "env_label":
            arrays[key] = np.asarray(values, dtype="<U16")
        elif key in {"observation", "state"}:
            arrays[key] = np.stack(values, axis=0)
        else:
            arrays[key] = np.asarray(values, dtype=np.float64)
    return arrays


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    f"{float(row['t']):.16g}",
                    f"{float(row['F_h']):.16g}",
                    f"{float(row['u']):.16g}",
                    f"{float(row['x_m']):.16g}",
                    f"{float(row['x_s']):.16g}",
                    f"{float(row['e']):.16g}",
                    f"{float(row['Fe']):.16g}",
                    f"{float(row['transparency_error']):.16g}",
                    f"{float(row['x_mdot']):.16g}",
                    f"{float(row['x_sdot']):.16g}",
                    row["env_label"],
                ]
            )


def _signal_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rms = float(np.sqrt(np.mean(values ** 2)))
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "rms": rms,
        "peak_to_peak": float(np.max(values) - np.min(values)),
    }


def _write_summary(
    out_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    duration: float,
    switch_time: float,
    force_amp: float,
    force_bias: float,
    force_freq: float,
    force_phase: float,
    u_voltage: float,
) -> None:
    segments = {
        "full_run": np.ones_like(arrays["t"], dtype=bool),
        "skin_segment": arrays["env_label"] == "skin",
        "fat_segment": arrays["env_label"] == "fat",
    }
    signals = ("x_m", "x_s", "e", "x_mdot", "x_sdot", "Fe", "transparency_error", "F_h")

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("Open-loop plant I/O analysis\n")
        fh.write("============================\n")
        fh.write(f"duration_s={duration:.6f}\n")
        fh.write(f"sample_time_s={cfg.RL_DT:.6f}\n")
        fh.write("position_origin=equilibrium_centered\n")
        fh.write(f"equilibrium_position_m={float(arrays['x_eq'][0]):.6f}\n")
        fh.write(f"switch_time_s={switch_time:.6f}\n")
        fh.write("force_waveform=sine\n")
        fh.write(f"force_amp_n={force_amp:.6f}\n")
        fh.write(f"force_bias_n={force_bias:.6f}\n")
        fh.write(f"force_freq_hz={force_freq:.6f}\n")
        fh.write(f"force_phase_rad={force_phase:.6f}\n")
        fh.write(f"u_voltage_v={u_voltage:.6f}\n\n")

        for segment_name, mask in segments.items():
            fh.write(f"[{segment_name}]\n")
            fh.write(f"samples={int(np.sum(mask))}\n")
            if not np.any(mask):
                fh.write("empty=true\n\n")
                continue
            fh.write(
                f"time_start_s={float(arrays['t'][mask][0]):.6f}\n"
                f"time_end_s={float(arrays['t'][mask][-1]):.6f}\n"
            )
            for signal in signals:
                stats = _signal_stats(arrays[signal][mask])
                fh.write(
                    f"{signal}: min={stats['min']:.9f}, max={stats['max']:.9f}, "
                    f"mean={stats['mean']:.9f}, rms={stats['rms']:.9f}, "
                    f"peak_to_peak={stats['peak_to_peak']:.9f}\n"
                )
            fh.write("\n")


def _plot_scopes(
    out_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    switch_time: float,
) -> None:
    t = arrays["t"]
    series = (
        ("F_h", "Input force [N]"),
        ("x_m", "Master displacement [m]"),
        ("x_s", "Slave displacement [m]"),
        ("e", "Tracking error [m]"),
        ("x_mdot", "Master velocity [m/s]"),
        ("x_sdot", "Slave velocity [m/s]"),
        ("Fe", "Environment force [N]"),
        ("transparency_error", "Transparency error [W]"),
    )
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 15), sharex=True)
    for ax, (key, ylabel) in zip(axes, series):
        ax.plot(t, arrays[key], color="tab:blue", lw=1.4)
        ax.axvline(switch_time, color="gray", lw=1.0, ls="--")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Open-Loop Plant I/O Scopes")
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_switch_zoom(
    out_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    switch_time: float,
    half_window: float = 8.0,
) -> None:
    t = arrays["t"]
    mask = (t >= max(0.0, switch_time - half_window)) & (t <= switch_time + half_window)
    series = (
        ("x_m", "Master displacement [m]"),
        ("x_s", "Slave displacement [m]"),
        ("e", "Tracking error [m]"),
        ("x_mdot", "Master velocity [m/s]"),
        ("x_sdot", "Slave velocity [m/s]"),
        ("Fe", "Environment force [N]"),
        ("transparency_error", "Transparency error [W]"),
    )
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 13), sharex=True)
    for ax, (key, ylabel) in zip(axes, series):
        ax.plot(t[mask], arrays[key][mask], color="tab:green", lw=1.5)
        ax.axvline(switch_time, color="gray", lw=1.0, ls="--")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Open-Loop Plant I/O Around Skin-To-Fat Switch")
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_error_scopes(
    out_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    switch_time: float,
) -> None:
    t = arrays["t"]
    series = (
        ("e", "Tracking error [m]"),
        ("transparency_error", "Transparency error [W]"),
    )
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 6), sharex=True)
    for ax, (key, ylabel) in zip(axes, series):
        ax.plot(t, arrays[key], color="tab:red", lw=1.5)
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.7)
        ax.axvline(switch_time, color="gray", lw=1.0, ls="--")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Tracking And Transparency Error Scopes")
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_open_loop_plant_io_analysis(
    *,
    duration: float = 180.0,
    switch_time: float = 30.0,
    force_amp: float = cfg.FORCE_INPUT_AMP,
    force_bias: float = cfg.MATLAB_REFERENCE_FORCE_BIAS,
    force_freq: float = cfg.MATLAB_REFERENCE_FORCE_FREQ_HZ,
    force_phase: float = 0.0,
    u_voltage: float = 0.0,
    out_dir: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    out_path = Path(out_dir) if out_dir is not None else _default_out_dir()
    out_path.mkdir(parents=True, exist_ok=True)

    env = TeleopEnv(
        env_mode=cfg.ENV_MODE_CHANGING,
        master_input_mode=cfg.MASTER_INPUT_FORCE,
        episode_duration=float(duration),
        env_switch_time=float(switch_time),
        terminate_on_error=False,
    )

    reset_options = {
        "force_amp": float(force_amp),
        "force_bias": float(force_bias),
        "force_freq": float(force_freq),
        "force_phase": float(force_phase),
        "force_waveform": "sine",
        "force_noise_std": 0.0,
        "force_noise_seed": 0,
    }
    obs, info = env.reset(seed=0, options=reset_options)
    applied_u = _clip_voltage(env, u_voltage)

    initial_row = _sample_row(env, obs, info)
    initial_row["u"] = applied_u
    rows: list[dict[str, Any]] = [initial_row]

    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step_voltage(applied_u)
        rows.append(_sample_row(env, obs, info))
        done = terminated or truncated

    expected_samples = env.max_steps + 1
    if len(rows) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} samples (including t=0), got {len(rows)}."
        )

    arrays = _rows_to_arrays(rows)
    csv_path = out_path / "plant_io.csv"
    debug_path = out_path / "plant_io_debug_state.npz"
    scopes_path = out_path / "plant_io_scopes.png"
    error_scopes_path = out_path / "plant_io_error_scopes.png"
    switch_zoom_path = out_path / "plant_io_switch_zoom.png"
    summary_path = out_path / "plant_io_summary.txt"

    _write_csv(rows, csv_path)
    np.savez(
        debug_path,
        t=arrays["t"],
        F_h=arrays["F_h"],
        u=arrays["u"],
        x_m=arrays["x_m"],
        x_s=arrays["x_s"],
        x_m_abs=arrays["x_m_abs"],
        x_s_abs=arrays["x_s_abs"],
        x_eq=arrays["x_eq"],
        e=arrays["e"],
        Fe=arrays["Fe"],
        transparency_error=arrays["transparency_error"],
        x_mdot=arrays["x_mdot"],
        x_sdot=arrays["x_sdot"],
        env_label=arrays["env_label"],
        P_m1=arrays["P_m1"],
        P_m2=arrays["P_m2"],
        P_s1=arrays["P_s1"],
        P_s2=arrays["P_s2"],
        mdot_L1=arrays["mdot_L1"],
        mdot_L2=arrays["mdot_L2"],
        x_v=arrays["x_v"],
        v_m=arrays["v_m"],
        v_s=arrays["v_s"],
        v_v=arrays["v_v"],
        observation=arrays["observation"],
        state=arrays["state"],
        sample_time_s=np.asarray([cfg.RL_DT], dtype=np.float64),
        duration_s=np.asarray([float(duration)], dtype=np.float64),
        switch_time_s=np.asarray([float(switch_time)], dtype=np.float64),
        force_amp=np.asarray([float(force_amp)], dtype=np.float64),
        force_bias=np.asarray([float(force_bias)], dtype=np.float64),
        force_freq=np.asarray([float(force_freq)], dtype=np.float64),
        force_phase=np.asarray([float(force_phase)], dtype=np.float64),
        u_voltage=np.asarray([applied_u], dtype=np.float64),
    )
    _plot_scopes(scopes_path, arrays, switch_time=float(switch_time))
    _plot_error_scopes(error_scopes_path, arrays, switch_time=float(switch_time))
    _plot_switch_zoom(switch_zoom_path, arrays, switch_time=float(switch_time))
    _write_summary(
        summary_path,
        arrays,
        duration=float(duration),
        switch_time=float(switch_time),
        force_amp=float(force_amp),
        force_bias=float(force_bias),
        force_freq=float(force_freq),
        force_phase=float(force_phase),
        u_voltage=float(applied_u),
    )

    print("Open-loop plant I/O analysis complete.")
    print(f"Samples: {len(rows)}")
    print(f"Sample time: {cfg.RL_DT:.6f} s")
    print(f"Duration: {rows[-1]['t']:.2f} s")
    print(f"Applied valve voltage: {applied_u:.3f} V")
    print(f"CSV: {csv_path}")
    print(f"Debug NPZ: {debug_path}")
    print(f"Scopes: {scopes_path}")
    print(f"Error scopes: {error_scopes_path}")
    print(f"Switch zoom: {switch_zoom_path}")
    print(f"Summary: {summary_path}")

    return {
        "out_dir": str(out_path),
        "csv_path": str(csv_path),
        "debug_path": str(debug_path),
        "scopes_path": str(scopes_path),
        "error_scopes_path": str(error_scopes_path),
        "switch_zoom_path": str(switch_zoom_path),
        "summary_path": str(summary_path),
    }


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze open-loop plant I/O scopes for the Python environment."
    )
    parser.add_argument("--duration", type=float, default=180.0, help="Episode duration in seconds.")
    parser.add_argument(
        "--switch-time",
        type=float,
        default=30.0,
        help="Time in seconds when the environment switches from skin to fat.",
    )
    parser.add_argument("--force-amp", type=float, default=cfg.FORCE_INPUT_AMP, help="Master force amplitude in N.")
    parser.add_argument(
        "--force-bias",
        type=float,
        default=cfg.MATLAB_REFERENCE_FORCE_BIAS,
        help="Master force bias in N.",
    )
    parser.add_argument(
        "--force-freq",
        type=float,
        default=cfg.MATLAB_REFERENCE_FORCE_FREQ_HZ,
        help="Master force frequency in Hz (MATLAB-aligned default).",
    )
    parser.add_argument("--force-phase", type=float, default=0.0, help="Master force phase in rad.")
    parser.add_argument(
        "--u-voltage",
        type=float,
        default=0.0,
        help="Open-loop servo valve voltage in V.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_default_out_dir()),
        help="Directory where the plant I/O analysis files are written.",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    run_open_loop_plant_io_analysis(
        duration=args.duration,
        switch_time=args.switch_time,
        force_amp=args.force_amp,
        force_bias=args.force_bias,
        force_freq=args.force_freq,
        force_phase=args.force_phase,
        u_voltage=args.u_voltage,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
