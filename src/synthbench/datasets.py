from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import math

REQUIRED_COLUMNS = {"sample_id", "image_path"}


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample IDs in {path}: {duplicates[:5]}")
    return frame


def validate_split_disjointness(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> None:
    groups = {
        "train": set(train["sample_id"]),
        "validation": set(validation["sample_id"]),
        "test": set(test["sample_id"]),
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        overlap = groups[left] & groups[right]
        if overlap:
            raise ValueError(f"{left}/{right} leakage: {sorted(overlap)[:5]}")


def create_nested_subsets(
    train: pd.DataFrame,
    fractions: Iterable[float] = (
        0.01,
        0.03,
        0.05,
        0.10,
        0.25,
        0.50,
        1.0,
    ),
    seed: int = 42,
    label_column: str | None = None,
) -> dict[float, pd.DataFrame]:

    fractions = sorted({float(value) for value in fractions})

    if not fractions or fractions[-1] != 1.0:
        raise ValueError("fractions must include 1.0")

    if fractions[0] <= 0 or fractions[-1] > 1:
        raise ValueError("fractions must be in (0, 1]")

    if train.empty:
        raise ValueError("train dataset cannot be empty")

    subsets: dict[float, pd.DataFrame] = {}

    if label_column and label_column in train.columns:
        shuffled_groups = []

        for _, group in train.groupby(label_column, sort=True):
            shuffled_group = group.sample(
                frac=1.0,
                random_state=seed,
            ).reset_index(drop=True)

            shuffled_groups.append(shuffled_group)

        for fraction in fractions:
            selected_groups = []

            for group in shuffled_groups:
                if fraction == 1.0:
                    size = len(group)
                else:
                    size = max(
                        1,
                        math.ceil(len(group) * fraction),
                    )

                selected_groups.append(group.iloc[:size])

            subset = pd.concat(
                selected_groups,
                ignore_index=True,
            )

            subsets[fraction] = subset.sample(
                frac=1.0,
                random_state=seed,
            ).reset_index(drop=True)

    else:
        shuffled = train.sample(
            frac=1.0,
            random_state=seed,
        ).reset_index(drop=True)

        for fraction in fractions:
            if fraction == 1.0:
                size = len(shuffled)
            else:
                size = max(
                    1,
                    math.ceil(len(shuffled) * fraction),
                )

            subsets[fraction] = shuffled.iloc[:size].copy()

    return subsets

def build_mixed_manifest(
    real: pd.DataFrame,
    synthetic: pd.DataFrame | None,
    synthetic_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Mix data using synthetic_ratio = selected synthetic / selected real samples."""
    if synthetic_ratio < 0:
        raise ValueError("synthetic_ratio cannot be negative")

    real_part = real.copy()
    real_part["source"] = "real"

    if synthetic is None or synthetic_ratio == 0:
        return real_part.reset_index(drop=True)

    synthetic_part = synthetic.copy()
    if "accepted" in synthetic_part:
        synthetic_part = synthetic_part[synthetic_part["accepted"].astype(bool)]

    requested = round(len(real_part) * synthetic_ratio)
    if requested == 0:
        return real_part.reset_index(drop=True)
    if synthetic_part.empty:
        raise ValueError("No accepted synthetic samples are available")

    sampled = synthetic_part.sample(
        n=requested,
        replace=requested > len(synthetic_part),
        random_state=seed,
    ).copy()
    sampled["source"] = "synthetic"

    return pd.concat([real_part, sampled], ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)


def resolve_paths(frame: pd.DataFrame, dataset_root: str | Path) -> pd.DataFrame:
    result = frame.copy()
    root = Path(dataset_root)
    for column in ("image_path", "annotation_path"):
        if column in result:
            result[column] = result[column].map(
                lambda value: str(root / value) if isinstance(value, str) and value else value
            )
    return result
