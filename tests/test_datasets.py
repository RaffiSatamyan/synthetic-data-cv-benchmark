import pandas as pd
import pytest

from synthbench.datasets import (
    build_mixed_manifest,
    create_nested_subsets,
    validate_split_disjointness,
)


REQUIRED_FRACTIONS = (
    0.01,
    0.03,
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
)


def make_frame(
    size: int,
    prefix: str = "sample",
    num_classes: int = 2,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [
                f"{prefix}_{index}"
                for index in range(size)
            ],
            "image_path": [
                f"images/{prefix}_{index}.png"
                for index in range(size)
            ],
            "label": [
                index % num_classes
                for index in range(size)
            ],
        }
    )


def test_all_required_fractions_are_created():
    subsets = create_nested_subsets(
        make_frame(
            size=10_000,
            num_classes=100,
        ),
        fractions=REQUIRED_FRACTIONS,
        seed=42,
        label_column="label",
    )

    assert set(subsets) == set(REQUIRED_FRACTIONS)


def test_subsets_are_nested():
    subsets = create_nested_subsets(
        make_frame(
            size=10_000,
            num_classes=100,
        ),
        fractions=REQUIRED_FRACTIONS,
        seed=42,
        label_column="label",
    )

    previous_ids: set[str] = set()

    for fraction in REQUIRED_FRACTIONS:
        current_ids = set(
            subsets[fraction]["sample_id"]
        )

        assert previous_ids <= current_ids
        previous_ids = current_ids


def test_every_subset_contains_every_class():
    subsets = create_nested_subsets(
        make_frame(
            size=10_000,
            num_classes=100,
        ),
        fractions=REQUIRED_FRACTIONS,
        seed=42,
        label_column="label",
    )

    expected_labels = set(range(100))

    for subset in subsets.values():
        assert set(subset["label"]) == expected_labels


def test_subsets_are_deterministic():
    frame = make_frame(
        size=10_000,
        num_classes=100,
    )

    first = create_nested_subsets(
        frame,
        fractions=REQUIRED_FRACTIONS,
        seed=42,
        label_column="label",
    )

    second = create_nested_subsets(
        frame,
        fractions=REQUIRED_FRACTIONS,
        seed=42,
        label_column="label",
    )

    for fraction in REQUIRED_FRACTIONS:
        assert first[fraction]["sample_id"].tolist() == (
            second[fraction]["sample_id"].tolist()
        )


@pytest.mark.parametrize(
    "ratio, expected_synthetic_count",
    [
        (0.5, 5),
        (1.0, 10),
        (2.0, 20),
        (5.0, 50),
        (10.0, 100),
    ],
)
def test_required_synthetic_ratios(
    ratio: float,
    expected_synthetic_count: int,
):
    real = make_frame(
        size=10,
        prefix="real",
    )

    synthetic = make_frame(
        size=100,
        prefix="synthetic",
    )

    mixed = build_mixed_manifest(
        real,
        synthetic,
        ratio,
        seed=42,
    )

    assert (mixed["source"] == "real").sum() == 10

    assert (
        mixed["source"] == "synthetic"
    ).sum() == expected_synthetic_count


def test_split_leakage_is_detected():
    train = make_frame(5)
    validation = make_frame(2)
    test = make_frame(2, prefix="test")

    with pytest.raises(
        ValueError,
        match="leakage",
    ):
        validate_split_disjointness(
            train,
            validation,
            test,
        )
