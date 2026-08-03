from __future__ import annotations

import argparse
import re
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_yaml
from .utils import stable_hash


def build_grid(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for dataset, real_fraction, seed in product(
        experiment["datasets"],
        experiment["real_fractions"],
        experiment["seeds"],
    ):
        common = {
            "dataset": dataset["name"],
            "model": dataset["model"],
            "real_fraction": float(real_fraction),
            "seed": int(seed),
        }

        for baseline in experiment.get("baselines", []):
            rows.append(
                _finalize(
                    {
                        **common,
                        "generator": baseline,
                        "synthetic_ratio": 0.0,
                    }
                )
            )

        synthetic = experiment.get("synthetic", {})
        combinations = product(
            synthetic.get("generators", []),
            synthetic.get("ratios", []),
        )
        for generator, ratio in combinations:
            rows.append(
                _finalize(
                    {
                        **common,
                        "generator": generator,
                        "synthetic_ratio": float(ratio),
                    }
                )
            )

    unique = {row["run_id"]: row for row in rows}
    return list(unique.values())


def _finalize(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["config_hash"] = stable_hash(row)
    result["run_id"] = make_run_id(result)
    result["status"] = "pending"
    return result


def make_run_id(row: dict[str, Any]) -> str:
    real = round(float(row["real_fraction"]) * 1000)
    synthetic = round(float(row["synthetic_ratio"]) * 100)
    pieces = [
        row["dataset"],
        row["model"],
        row["generator"],
        f"rf{real:03d}",
        f"sr{synthetic:03d}",
        f"seed{row['seed']}",
        row["config_hash"],
    ]
    return "-".join(_slug(piece) for piece in pieces)


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def write_grid(experiment_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    frame = pd.DataFrame(build_grid(load_yaml(experiment_path)))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deduplicated experiment manifest")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = write_grid(args.experiment, args.output)
    print(f"Wrote {len(frame)} experiments to {args.output}")


if __name__ == "__main__":
    main()
