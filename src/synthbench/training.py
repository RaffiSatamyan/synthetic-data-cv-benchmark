from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .utils import environment_info, stable_hash, write_json, write_yaml


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_validation_metric: float,
    resolved_config: dict[str, Any],
) -> None:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError('Install training dependencies: pip install -e ".[training]"') from error

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_metric": best_validation_metric,
        "resolved_config": resolved_config,
        "config_hash": stable_hash(resolved_config),
        "random_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('Install training dependencies: pip install -e ".[training]"') from error

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def prepare_run_directory(run_dir: str | Path, config: dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "environment.json", environment_info())
    write_json(run_dir / "status.json", {"status": "created"})
    return run_dir


def train_experiment(config: dict[str, Any], run_dir: str | Path) -> None:
    """Task-training extension point with a fixed run/checkpoint contract."""
    task = config["dataset"]["task"]
    raise NotImplementedError(
        f"Implement the {task!r} training loop here, or dispatch to a new module once "
        "this file becomes large. Use prepare_run_directory(), save_checkpoint(), "
        "best.pt selected on validation data, and last.pt for resuming."
    )
