import pytest

from synthbench import build_grid


def test_pilot_count() -> None:
    assert len(build_grid("pilot")) == 18


def test_full_count() -> None:
    assert len(build_grid("full")) == 864


def test_invalid_preset() -> None:
    with pytest.raises(ValueError):
        build_grid("unknown")


def test_rows_are_unique() -> None:
    for preset in ("pilot", "full"):
        rows = build_grid(preset)
        keys = [tuple(sorted(row.items())) for row in rows]
        assert len(keys) == len(set(keys))
