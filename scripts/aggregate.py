from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed run metrics")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(args.outputs.glob("*/metrics.json")):
        with metrics_path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        row.setdefault("run_id", metrics_path.parent.name)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {args.outputs}")

    args.results.mkdir(parents=True, exist_ok=True)
    all_runs = pd.json_normalize(rows)
    all_runs.to_csv(args.results / "all_runs.csv", index=False)

    grouping = [
        column
        for column in ("dataset", "model", "generator", "real_fraction", "synthetic_ratio")
        if column in all_runs
    ]
    metric_columns = [
        column
        for column in all_runs.select_dtypes("number").columns
        if column not in {"seed", "real_fraction", "synthetic_ratio"}
    ]

    if grouping and metric_columns:
        summary = all_runs.groupby(grouping, dropna=False)[metric_columns].agg(["mean", "std"])
        summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
        summary.reset_index().to_csv(args.results / "summary.csv", index=False)
    else:
        all_runs.to_csv(args.results / "summary.csv", index=False)

    print(f"Aggregated {len(all_runs)} runs into {args.results}")


if __name__ == "__main__":
    main()
