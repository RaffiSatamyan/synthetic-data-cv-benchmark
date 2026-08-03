from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]+))?\}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return _resolve_environment(value)


def _resolve_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_environment(item) for item in value]
    if isinstance(value, str):
        match = _ENV_PATTERN.fullmatch(value)
        if match:
            import os

            name, default = match.groups()
            resolved = os.getenv(name, default)
            if resolved is None:
                raise ValueError(f"Environment variable {name} is required")
            return resolved
    return value


def deep_merge(*configs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for config in configs:
        result = _merge_two(result, config)
    return result


def _merge_two(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_two(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_run_config(
    dataset_path: str | Path,
    generator_path: str | Path,
    model_path: str | Path,
    training_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "dataset": load_yaml(dataset_path),
        "generator": load_yaml(generator_path),
        "model": load_yaml(model_path),
        "training": load_yaml(training_path),
    }
    if overrides:
        config = deep_merge(config, overrides)
    validate_run_config(config)
    return config


def validate_run_config(config: dict[str, Any]) -> None:
    for section in ("dataset", "generator", "model", "training"):
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")

    dataset_task = config["dataset"].get("task")
    model_task = config["model"].get("task")
    if dataset_task != model_task:
        raise ValueError(
            f"Dataset task {dataset_task!r} does not match model task {model_task!r}"
        )

    ratio = float(config.get("experiment", {}).get("synthetic_ratio", 0.0))
    if ratio < 0:
        raise ValueError("synthetic_ratio cannot be negative")
