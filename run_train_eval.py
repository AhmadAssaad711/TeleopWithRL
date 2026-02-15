"""Run full pipeline: training then evaluation plots."""
import os
import subprocess
import sys

WD = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    print("[1/2] Running training...")
    subprocess.run([sys.executable, os.path.join(WD, "run_train.py")], check=True)

    print("[2/2] Generating evaluation plots...")
    subprocess.run([sys.executable, os.path.join(WD, "plot_results.py")], check=True)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
