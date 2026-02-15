"""Run benchmark for MRAC, RL-constant, and RL-changing agents."""

import os
import sys


class Tee:
    def __init__(self, filename: str, stream):
        self.file = open(filename, "w", encoding="utf-8")
        self.stream = stream

    def write(self, data: str) -> None:
        self.file.write(data)
        self.file.flush()
        try:
            self.stream.write(data)
            self.stream.flush()
        except Exception:
            pass

    def flush(self) -> None:
        self.file.flush()
        try:
            self.stream.flush()
        except Exception:
            pass


def main() -> None:
    root = os.path.dirname(__file__)
    log_dir = os.path.join(root, "results", "comparisons")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "benchmark_output.log")

    sys.stdout = Tee(log_path, sys.stdout)
    sys.stderr = Tee(log_path, sys.stderr)

    from benchmark_agents import run_full_benchmark

    run_full_benchmark()


if __name__ == "__main__":
    main()
