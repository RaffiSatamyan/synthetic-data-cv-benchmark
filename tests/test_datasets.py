import pandas as pd
import pytest

from synthbench.datasets import (
    build_mixed_manifest,
    create_nested_subsets,
    validate_split_disjointness,
)


def make_frame(size: int, prefix: str = "sample"):
    return pd.DataFrame(
        {
            "sample_id": [f"{prefix}_{index}" for index in range(size)],
            "image_path": [f"images/{index}.png" for index in range(size)],
            "label": [index % 2 for index in range(size)],
        }
    )


def test_subsets_are_nested():
    subsets = create_nested_subsets(
        make_frame(1000), fractions=(0.01, 0.03, 0.10, 1.0), seed=1, label_column="label"
    )
    assert set(subsets[0.01].sample_id) <= set(subsets[0.03].sample_id)
    assert set(subsets[0.03].sample_id) <= set(subsets[0.10].sample_id)
    assert set(subsets[0.10].sample_id) <= set(subsets[1.0].sample_id)


def test_synthetic_ratio_has_one_definition():
    mixed = build_mixed_manifest(make_frame(10), make_frame(20, "syn"), 1.5, seed=0)
    assert (mixed.source == "real").sum() == 10
    assert (mixed.source == "synthetic").sum() == 15


def test_split_leakage_is_detected():
    train = make_frame(5)
    validation = make_frame(2)
    test = make_frame(2, "test")
    with pytest.raises(ValueError, match="leakage"):
        validate_split_disjointness(train, validation, test)
