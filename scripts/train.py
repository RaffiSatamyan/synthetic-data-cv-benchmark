from __future__ import annotations

import argparse
from pathlib import Path

from synthbench.config import load_yaml
from synthbench.experiments import build_grid

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve and run one experiment")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--configs-dir", type=Path, default=REPO_ROOT / "configs")
    parser.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--generation-id",
        default=None,
        help="Synthetic generation id to train on (defaults to newest).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = build_grid(load_yaml(args.experiment))
    if args.index < 0 or args.index >= len(rows):
        raise IndexError(f"index must be between 0 and {len(rows) - 1}")

    run = rows[args.index]
    print(run)
    if args.dry_run:
        return

    # Import lazily so --dry-run works without the training extras installed.
    from synthbench.training import resolve_experiment_config, train_experiment

    config = resolve_experiment_config(run, args.configs_dir, args.generation_id)
    run_dir = train_experiment(config, args.outputs / run["run_id"])
    print(f"Completed run: {run_dir}")


if __name__ == "__main__":
    main()
