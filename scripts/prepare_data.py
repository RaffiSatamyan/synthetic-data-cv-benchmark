from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthbench.datasets import create_nested_subsets


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed dataset split manifests")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--label-column")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.source_manifest)
    missing = {"sample_id", "image_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")

    shuffled = frame.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_size = round(len(shuffled) * args.test_fraction)
    validation_size = round(len(shuffled) * args.validation_fraction)

    test = shuffled.iloc[:test_size]
    validation = shuffled.iloc[test_size : test_size + validation_size]
    train = shuffled.iloc[test_size + validation_size :]

    split_dir = args.dataset_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(split_dir / "val.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)

    subsets = create_nested_subsets(
        train,
        fractions=(0.01, 0.03, 0.10, 1.0),
        seed=args.seed,
        label_column=args.label_column,
    )
    names = {0.01: "1pct", 0.03: "3pct", 0.10: "10pct", 1.0: "full"}
    for fraction, subset in subsets.items():
        destination = split_dir / names[fraction] / "train.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(destination, index=False)

    print(f"Prepared {args.dataset_dir}: train={len(train)}, val={len(validation)}, test={len(test)}")


if __name__ == "__main__":
    main()
