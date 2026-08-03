from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

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
    fractions: Iterable[float] = (0.01, 0.03, 0.10, 1.0),
    seed: int = 42,
    label_column: str | None = None,
) -> dict[float, pd.DataFrame]:
    """Create deterministic nested subsets; stratify approximately when labels exist."""
    fractions = sorted({float(value) for value in fractions})
    if not fractions or fractions[-1] != 1.0:
        raise ValueError("fractions must include 1.0")
    if fractions[0] <= 0 or fractions[-1] > 1:
        raise ValueError("fractions must be in (0, 1]")

    shuffled = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if label_column and label_column in train.columns:
        parts = [
            group.sample(frac=1.0, random_state=seed)
            for _, group in train.groupby(label_column, sort=True)
        ]
        shuffled = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    subsets: dict[float, pd.DataFrame] = {}
    for fraction in fractions:
        size = len(shuffled) if fraction == 1.0 else max(1, round(len(shuffled) * fraction))
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
