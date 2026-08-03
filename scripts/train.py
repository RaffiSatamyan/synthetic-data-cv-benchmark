from __future__ import annotations

import argparse
from pathlib import Path

from synthbench.config import load_yaml
from synthbench.experiments import build_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve and run one experiment")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = build_grid(load_yaml(args.experiment))
    if args.index < 0 or args.index >= len(rows):
        raise IndexError(f"index must be between 0 and {len(rows) - 1}")

    run = rows[args.index]
    print(run)
    if args.dry_run:
        return

    raise NotImplementedError(
        "Resolve dataset/generator/model/training YAML for this row, then call "
        "synthbench.training.train_experiment(config, outputs/run_id)."
    )


if __name__ == "__main__":
    main()
