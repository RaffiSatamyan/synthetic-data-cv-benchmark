from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from .config import load_yaml
from .datasets import build_mixed_manifest, load_manifest
from .evaluation import classification_metrics, save_metrics
from .models import build_model
from .utils import environment_info, set_seed, stable_hash, write_json, write_yaml

# Maps a real_fraction to the split-manifest key declared in the dataset config.
FRACTION_TO_SPLIT = {
    0.01: "1pct",
    0.03: "3pct",
    0.05: "5pct",
    0.1: "10pct",
    0.25: "25pct",
    0.5: "50pct",
    1.0: "full",
}

# Generators that mix in no external synthetic images.
BASELINE_GENERATORS = {"real_only", "classical_augmentation"}


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


def _split_key_for_fraction(real_fraction: float) -> str:
    key = FRACTION_TO_SPLIT.get(round(real_fraction, 2))
    if key is None:
        raise ValueError(
            f"real_fraction {real_fraction} has no split; expected one of "
            f"{sorted(FRACTION_TO_SPLIT)}"
        )
    return key


def find_synthetic_manifest(
    data_root: Path, generator: str, generation_id: str | None
) -> Path | None:
    """Locate a synthetic manifest for a generator, newest generation by default."""
    generator_dir = data_root / "synthetic" / generator
    if generation_id:
        manifest = generator_dir / generation_id / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"Synthetic manifest not found: {manifest}")
        return manifest
    manifests = sorted(generator_dir.glob("*/manifest.csv"), key=lambda p: p.stat().st_mtime)
    if not manifests:
        raise FileNotFoundError(
            f"No synthetic manifests under {generator_dir}. Generate data first, "
            "or pass a baseline generator."
        )
    return manifests[-1]


def _append_log(run_dir: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n"
    with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(line)


def _run_validation(model, loader, device) -> dict[str, float]:
    import torch

    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device, non_blocking=True))
            predicted_labels.extend(outputs.argmax(dim=1).cpu().tolist())
            true_labels.extend(int(value) for value in targets.tolist())
    return classification_metrics(true_labels, predicted_labels)


def train_experiment(config: dict[str, Any], run_dir: str | Path) -> Path:
    """Train one downstream classification run under the fixed run contract."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from .data_loading import (
        build_eval_transforms,
        build_image_dataset,
        build_label_map,
        build_train_transforms,
        make_batch_mixer,
        mixup_loss,
    )

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    generator_cfg = config["generator"]
    experiment = config.get("experiment", {})

    if dataset_cfg.get("task") != "classification":
        raise NotImplementedError(
            f"train_experiment currently supports classification, not "
            f"{dataset_cfg.get('task')!r}."
        )

    # Keep the model head consistent with the dataset regardless of the model yaml.
    model_cfg["num_classes"] = int(dataset_cfg["num_classes"])

    run_dir = prepare_run_directory(run_dir, config)
    try:
        seed = int(experiment.get("seed", training_cfg.get("seed", 0)))
        set_seed(seed)

        real_fraction = float(experiment.get("real_fraction", 1.0))
        synthetic_ratio = float(experiment.get("synthetic_ratio", 0.0))
        generator = experiment.get("generator", generator_cfg.get("name"))
        data_root = Path(dataset_cfg["data_root"])
        label_column = dataset_cfg.get("label_column", "label")

        splits = dataset_cfg["splits"]
        real = load_manifest(splits[_split_key_for_fraction(real_fraction)])
        validation = load_manifest(splits["validation"])

        synthetic = None
        if synthetic_ratio > 0 and generator not in BASELINE_GENERATORS:
            manifest_path = find_synthetic_manifest(
                data_root, generator, experiment.get("generation_id")
            )
            synthetic = load_manifest(manifest_path)

        train_frame = build_mixed_manifest(real, synthetic, synthetic_ratio, seed)
        label_map = build_label_map(validation, label_column)
        write_json(run_dir / "label_map.json", {str(k): v for k, v in label_map.items()})

        train_transform = build_train_transforms(dataset_cfg, generator_cfg)
        eval_transform = build_eval_transforms(dataset_cfg)
        mixer = make_batch_mixer(generator_cfg)

        train_dataset = build_image_dataset(
            train_frame, label_map, train_transform, data_root, label_column
        )
        val_dataset = build_image_dataset(
            validation, label_map, eval_transform, data_root, label_column
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_workers = int(training_cfg.get("num_workers", 0))
        batch_size = int(training_cfg.get("batch_size", 64))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=len(train_dataset) > batch_size,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        model = build_model(model_cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        )
        epochs = int(training_cfg.get("epochs", 100))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        criterion = nn.CrossEntropyLoss()

        use_amp = bool(training_cfg.get("mixed_precision", False)) and device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except (AttributeError, TypeError):  # older torch
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        limit_batches = training_cfg.get("limit_batches")

        write_json(run_dir / "status.json", {"status": "running", "epochs": epochs})
        _append_log(
            run_dir,
            f"start run={experiment.get('run_id')} device={device.type} "
            f"train={len(train_dataset)} val={len(val_dataset)} epochs={epochs}",
        )

        best_metric = float("-inf")
        best_metrics: dict[str, float] = {}
        best_epoch = -1
        global_step = 0
        started = time.time()

        for epoch in range(epochs):
            model.train()
            for batch_index, (inputs, targets) in enumerate(train_loader):
                if limit_batches is not None and batch_index >= int(limit_batches):
                    break
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(inputs)
                    if mixer is not None:
                        mixed, target_a, target_b, lam = mixer(inputs, targets)
                        outputs = model(mixed)
                        loss = mixup_loss(criterion, outputs, target_a, target_b, lam)
                    else:
                        loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
            scheduler.step()

            metrics = _run_validation(model, val_loader, device)
            save_checkpoint(
                run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_validation_metric=best_metric,
                resolved_config=config,
            )
            if metrics["accuracy"] > best_metric:
                best_metric = metrics["accuracy"]
                best_metrics = metrics
                best_epoch = epoch
                save_checkpoint(
                    run_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_validation_metric=best_metric,
                    resolved_config=config,
                )
            _append_log(
                run_dir,
                f"epoch={epoch} val_accuracy={metrics['accuracy']:.4f} "
                f"val_macro_f1={metrics['macro_f1']:.4f} best={best_metric:.4f}",
            )
            write_json(
                run_dir / "status.json",
                {"status": "running", "epoch": epoch, "epochs": epochs, "best": best_metric},
            )

        final_metrics = {
            "run_id": experiment.get("run_id", run_dir.name),
            "dataset": dataset_cfg.get("name"),
            "model": model_cfg.get("name"),
            "generator": generator,
            "real_fraction": real_fraction,
            "synthetic_ratio": synthetic_ratio,
            "seed": seed,
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "val_accuracy": best_metrics.get("accuracy"),
            "val_macro_f1": best_metrics.get("macro_f1"),
            "train_seconds": round(time.time() - started, 2),
        }
        save_metrics(run_dir, final_metrics)
        write_json(run_dir / "status.json", {"status": "completed", "best": best_metric})
        _append_log(run_dir, f"completed best_val_accuracy={best_metric:.4f}")
        return run_dir
    except Exception as error:
        write_json(
            run_dir / "status.json",
            {"status": "failed", "reason": f"{type(error).__name__}: {error}"},
        )
        _append_log(run_dir, f"failed {type(error).__name__}: {error}")
        raise


def resolve_experiment_config(
    row: dict[str, Any], configs_dir: str | Path, generation_id: str | None = None
) -> dict[str, Any]:
    """Resolve one grid row into a full run config using the repo's config tree."""
    from .config import resolve_run_config

    configs_dir = Path(configs_dir)
    experiment = {
        key: row[key]
        for key in ("dataset", "model", "generator", "real_fraction", "synthetic_ratio", "seed")
        if key in row
    }
    experiment["run_id"] = row.get("run_id")
    if generation_id:
        experiment["generation_id"] = generation_id
    return resolve_run_config(
        dataset_path=configs_dir / "datasets" / f"{row['dataset']}.yaml",
        generator_path=configs_dir / "generators" / f"{row['generator']}.yaml",
        model_path=configs_dir / "models" / f"{row['model']}.yaml",
        training_path=configs_dir / "training.yaml",
        overrides={"experiment": experiment},
    )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a previously saved resolved run config."""
    return load_yaml(path)
