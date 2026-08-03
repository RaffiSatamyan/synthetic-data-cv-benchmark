import pytest

from synthbench.config import deep_merge, validate_run_config


def test_deep_merge_preserves_nested_values():
    result = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
    assert result == {"a": {"b": 3, "c": 2}}


def test_task_mismatch_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        validate_run_config(
            {
                "dataset": {"task": "classification"},
                "generator": {"name": "real_only"},
                "model": {"task": "detection"},
                "training": {},
            }
        )
