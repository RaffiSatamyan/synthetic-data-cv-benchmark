"""Generate figures for one run or for the whole experiment sweep.

Per-run:  python scripts/plot.py --run outputs/<run_id>
Sweep:    python scripts/plot.py --aggregate outputs --metric test_accuracy
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synthbench.plots import aggregate_runs, plot_data_scaling, plot_run

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build run or sweep figures")
    parser.add_argument("--run", type=Path, default=None, help="A single run directory")
    parser.add_argument("--aggregate", type=Path, default=None, help="Outputs dir of many runs")
    parser.add_argument("--metric", default="test_accuracy")
    parser.add_argument(
        "--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures"
    )
    args = parser.parse_args()

    if not args.run and not args.aggregate:
        parser.error("pass --run and/or --aggregate")

    if args.run:
        outputs = plot_run(args.run)
        for path in outputs:
            print(f"wrote {path}")

    if args.aggregate:
        frame = aggregate_runs(args.aggregate)
        if frame.empty:
            print(f"No completed runs found under {args.aggregate}")
            return
        summary = args.figures_dir / "summary.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(summary, index=False)
        figure = plot_data_scaling(
            frame, args.figures_dir / f"data_scaling_{args.metric}.png", metric=args.metric
        )
        print(f"wrote {summary}")
        print(f"wrote {figure}")


if __name__ == "__main__":
    main()
