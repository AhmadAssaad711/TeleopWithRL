from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from TeleopWithRL.matlab_literal_env.studies.focused_evaluation import run_focused_evaluation
else:
    from ..studies.focused_evaluation import run_focused_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the focused unified evaluation battery for a continuous policy-gradient model."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a policy-gradient run dir, model zip, or summary-adjacent model path.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic policy actions.")
    parser.add_argument("--skip-bode", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir) if args.out_dir else (
        model_path if model_path.is_dir() else model_path.parent.parent
    ) / "focused_eval"
    result = run_focused_evaluation(
        model_path=model_path,
        out_dir=out_dir,
        seed=int(args.seed),
        deterministic=not bool(args.stochastic),
        include_bode=not bool(args.skip_bode),
        save_plots=not bool(args.no_plots),
    )
    print(f"focused_eval_dir={out_dir}")
    print(f"normal_scenarios={len(result['metrics'])}")
    print(f"bode_scenarios={len(result['bode'])}")


if __name__ == "__main__":
    main()
