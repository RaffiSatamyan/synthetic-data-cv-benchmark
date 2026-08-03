from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthbench import build_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["pilot", "full"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = build_grid(args.preset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"{args.preset}: {len(rows)} experiments -> {args.output}")


if __name__ == "__main__":
    main()
