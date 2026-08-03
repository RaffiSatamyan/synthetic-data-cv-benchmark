from __future__ import annotations

import argparse
from pathlib import Path

from synthbench.evaluation import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one completed run")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluate_run(args.run_dir)


if __name__ == "__main__":
    main()
