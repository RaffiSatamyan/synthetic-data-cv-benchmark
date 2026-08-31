from __future__ import annotations

import csv
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

from .config import load_yaml
from .datasets import build_mixed_manifest, load_manifest
from .evaluation import (
    classification_metrics,
    collect_predictions,
    confusion_matrix,
    per_class_accuracy,
    save_metrics,
    save_predictions_csv,
    topk_accuracy,
)
from .models import build_model
from .utils import environment_info, set_seed, stable_hash, write_json, write_yaml

# Column order for the per-epoch training history log used to plot learning curves.
HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "val_accuracy",
    "val_macro_f1",
    "val_top5_accuracy",
    "learning_rate",
    "epoch_seconds",
)

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


def _evaluate_split(model, loader, device, sample_ids, index_to_label):
    """Validate and return (metrics, per-sample records, true indices, pred indices)."""
    records, true_labels, predicted_labels = collect_predictions(
        model, loader, device, sample_ids=sample_ids, index_to_label=index_to_label
    )
    metrics = classification_metrics(true_labels, predicted_labels)
    metrics["top5_accuracy"] = topk_accuracy(records, k=5)
    return metrics, records, true_labels, predicted_labels


def _append_history(run_dir: Path, row: dict[str, Any]) -> None:
    """Append one epoch to history.csv, writing the header on first use."""
    path = run_dir / "history.csv"
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in HISTORY_COLUMNS})


def build_optimizer(model, training_cfg: dict[str, Any]):
    """Build the optimizer from the model's paper recipe.

    Reads ``training_cfg['optimizer']`` (name + hyper-parameters). When no
    ``optimizer`` block is present it falls back to AdamW using the flat
    ``learning_rate`` / ``weight_decay`` keys, preserving the original behaviour.
    """
    import torch

    spec = training_cfg.get("optimizer")
    default_lr = float(training_cfg.get("learning_rate", 1e-3))
    default_wd = float(training_cfg.get("weight_decay", 1e-4))
    if not spec:
        return torch.optim.AdamW(model.parameters(), lr=default_lr, weight_decay=default_wd)

    name = str(spec.get("name", "adamw")).lower()
    lr = float(spec.get("lr", default_lr))
    weight_decay = float(spec.get("weight_decay", default_wd))
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(spec.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(spec.get("nesterov", False)),
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=float(spec.get("momentum", 0.0)),
            alpha=float(spec.get("alpha", 0.99)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name!r}")


def build_scheduler(optimizer, training_cfg: dict[str, Any], epochs: int):
    """Build the LR scheduler from the model's paper recipe (per-epoch stepping).

    Supports step / multistep / cosine / exponential / none. Defaults to cosine
    over ``epochs`` when no ``scheduler`` block is given (original behaviour).
    """
    import torch

    spec = training_cfg.get("scheduler")
    if spec is None:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    name = str(spec.get("name", "cosine")).lower()
    if name in ("none", "constant"):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(spec.get("step_size", 30)), gamma=float(spec.get("gamma", 0.1))
        )
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(m) for m in spec.get("milestones", [30, 60])],
            gamma=float(spec.get("gamma", 0.1)),
        )
    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(spec.get("gamma", 0.95)))
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(spec.get("t_max", max(epochs, 1)))
        )
    raise ValueError(f"Unknown scheduler: {name!r}")


def completed_run(run_dir: str | Path, config_hash: str) -> Path | None:
    """Return the run dir if it holds a completed run matching ``config_hash``.

    This is what lets us honour "if a model is already trained, load its exact
    checkpoint instead of retraining", keyed on the full resolved config.
    """
    run_dir = Path(run_dir)
    status_path = run_dir / "status.json"
    if not status_path.exists() or not (run_dir / "best.pt").exists():
        return None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if status.get("status") == "completed" and status.get("config_hash") == config_hash:
        return run_dir
    return None


def train_experiment(
    config: dict[str, Any], run_dir: str | Path, *, force: bool = False
) -> Path:
    """Train one downstream classification run under the fixed run contract.

    If a completed run with an identical resolved config already exists at
    ``run_dir``, training is skipped and the existing directory is returned
    (unless ``force`` is set). This prevents wasteful retraining and, crucially,
    guarantees each dataset size trains from its own fresh checkpoint rather than
    inheriting weights from a larger split.
    """
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

    # Hash the fully resolved config so we can detect an identical completed run.
    config_hash = stable_hash(config)
    run_dir = Path(run_dir)
    already_trained = completed_run(run_dir, config_hash)
    if already_trained is not None and not force:
        return already_trained
    if run_dir.exists():
        # An incomplete, failed, or force-overwritten run: start clean.
        shutil.rmtree(run_dir)

    run_dir = prepare_run_directory(run_dir, config)
    try:
        seed = int(experiment.get("seed", training_cfg.get("seed", 0)))
        set_seed(seed)

        real_fraction = float(experiment.get("real_fraction", 1.0))
        synthetic_ratio = float(experiment.get("synthetic_ratio", 0.0))
        generator = experiment.get("generator", generator_cfg.get("name"))
        data_root = Path(dataset_cfg["data_root"])
        # Images may live apart from data_root (e.g. a read-only Kaggle mount),
        # so committed splits store paths relative to this per-device image_root.
        image_root = Path(dataset_cfg.get("image_root", data_root))
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
        index_to_label = {index: label for label, index in label_map.items()}
        num_classes = int(dataset_cfg["num_classes"])
        val_sample_ids = (
            validation["sample_id"].tolist() if "sample_id" in validation.columns else None
        )
        write_json(run_dir / "label_map.json", {str(k): v for k, v in label_map.items()})

        train_transform = build_train_transforms(dataset_cfg, generator_cfg)
        eval_transform = build_eval_transforms(dataset_cfg)
        mixer = make_batch_mixer(generator_cfg)

        train_dataset = build_image_dataset(
            train_frame, label_map, train_transform, image_root, label_column
        )
        val_dataset = build_image_dataset(
            validation, label_map, eval_transform, image_root, label_column
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
        epochs = int(training_cfg.get("epochs", 100))
        optimizer = build_optimizer(model, training_cfg)
        scheduler = build_scheduler(optimizer, training_cfg, epochs)
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
            epoch_started = time.time()
            loss_total = 0.0
            loss_batches = 0
            current_lr = optimizer.param_groups[0]["lr"]
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
                loss_total += float(loss.detach())
                loss_batches += 1
                global_step += 1
            scheduler.step()
            train_loss = loss_total / loss_batches if loss_batches else float("nan")

            metrics, val_records, val_true, val_pred = _evaluate_split(
                model, val_loader, device, val_sample_ids, index_to_label
            )
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
                # Snapshot the winning epoch's per-sample predictions and confusion.
                save_predictions_csv(run_dir / "predictions_val.csv", val_records)
                write_json(
                    run_dir / "val_confusion_matrix.json",
                    {
                        "labels": [index_to_label.get(i, i) for i in range(num_classes)],
                        "matrix": confusion_matrix(val_true, val_pred, num_classes),
                        "per_class_accuracy": per_class_accuracy(
                            val_true, val_pred, num_classes
                        ),
                    },
                )
            _append_history(
                run_dir,
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_accuracy": round(metrics["accuracy"], 6),
                    "val_macro_f1": round(metrics["macro_f1"], 6),
                    "val_top5_accuracy": round(metrics["top5_accuracy"], 6),
                    "learning_rate": current_lr,
                    "epoch_seconds": round(time.time() - epoch_started, 2),
                },
            )
            _append_log(
                run_dir,
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"val_accuracy={metrics['accuracy']:.4f} "
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
            "val_top5_accuracy": best_metrics.get("top5_accuracy"),
            "config_hash": config_hash,
            "train_seconds": round(time.time() - started, 2),
        }
        save_metrics(run_dir, final_metrics)
        write_json(
            run_dir / "status.json",
            {"status": "completed", "best": best_metric, "config_hash": config_hash},
        )
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
    config = resolve_run_config(
        dataset_path=configs_dir / "datasets" / f"{row['dataset']}.yaml",
        generator_path=configs_dir / "generators" / f"{row['generator']}.yaml",
        model_path=configs_dir / "models" / f"{row['model']}.yaml",
        training_path=configs_dir / "training.yaml",
        overrides={"experiment": experiment},
    )
    # A model's paper recipe (optimizer/scheduler/epochs/batch) lives in its
    # model yaml under `train:` and overrides the shared training defaults, so
    # each architecture trains exactly the way its source paper prescribes.
    recipe = config.get("model", {}).get("train")
    if recipe:
        from .config import deep_merge

        config["training"] = deep_merge(config["training"], recipe)
    return config


def train(
    dataset: str,
    model: str,
    *,
    generator: str = "real_only",
    real_fraction: float = 1.0,
    synthetic_ratio: float = 0.0,
    seed: int = 0,
    configs_dir: str | Path = "configs",
    outputs_dir: str | Path = "outputs",
    generation_id: str | None = None,
    training_overrides: dict[str, Any] | None = None,
    force: bool = False,
    provision: bool = True,
    evaluate: bool = True,
    make_plots: bool = True,
    all_subsets: bool = False,
) -> Path | list[Path]:
    """One-call entry point: data + model + params -> a fully trained, evaluated run.

    This is the "hand it everything and it does the rest" function: it resolves
    the config from the repo's config tree, provisions the dataset if missing,
    trains from scratch (or loads the exact existing checkpoint when a matching
    completed run exists), evaluates on the held-out test split, and writes the
    per-run figures. ``run_id`` matches the experiment manifest so a single run
    and a full sweep stay addressable the same way.

    With ``all_subsets=True`` it instead sweeps every ``real_fraction`` in the
    dataset config, ascending (1% -> 3% -> ... -> 100%), training each subset
    from its own fresh checkpoint, and finally writes the cross-run
    accuracy-vs-data-size figure. Returns the list of run directories.
    """
    from .config import deep_merge
    from .experiments import make_run_id

    if all_subsets:
        return _train_all_subsets(
            dataset,
            model,
            generator=generator,
            synthetic_ratio=synthetic_ratio,
            seed=seed,
            configs_dir=configs_dir,
            outputs_dir=outputs_dir,
            generation_id=generation_id,
            training_overrides=training_overrides,
            force=force,
            provision=provision,
            evaluate=evaluate,
            make_plots=make_plots,
        )

    row: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "generator": generator,
        "real_fraction": float(real_fraction),
        "synthetic_ratio": float(synthetic_ratio),
        "seed": int(seed),
    }
    row["config_hash"] = stable_hash(row)
    row["run_id"] = make_run_id(row)

    config = resolve_experiment_config(row, configs_dir, generation_id)
    if training_overrides:
        config = deep_merge(config, {"training": training_overrides})

    if provision:
        from .provision import ensure_dataset

        ensure_dataset(config["dataset"])

    run_dir = train_experiment(config, Path(outputs_dir) / row["run_id"], force=force)

    if evaluate:
        from .evaluation import evaluate_run

        evaluate_run(run_dir, save_predictions=True)
    if make_plots:
        from .plots import plot_run

        plot_run(run_dir)
    return run_dir


def _train_all_subsets(
    dataset: str,
    model: str,
    *,
    generator: str,
    synthetic_ratio: float,
    seed: int,
    configs_dir: str | Path,
    outputs_dir: str | Path,
    generation_id: str | None,
    training_overrides: dict[str, Any] | None,
    force: bool,
    provision: bool,
    evaluate: bool,
    make_plots: bool,
) -> list[Path]:
    """Train every real-data fraction in the dataset config, then plot the sweep.

    Each fraction is a fully independent from-scratch run (this is what makes the
    "does more data help" comparison valid — no fraction inherits weights from a
    larger one). Provisioning runs only for the first fraction; the rest reuse the
    now-present data and split.
    """
    dataset_cfg = load_yaml(Path(configs_dir) / "datasets" / f"{dataset}.yaml")
    fractions = sorted(float(value) for value in dataset_cfg.get("real_fractions", [1.0]))

    run_dirs: list[Path] = []
    for index, fraction in enumerate(fractions):
        run_dir = train(
            dataset,
            model,
            generator=generator,
            real_fraction=fraction,
            synthetic_ratio=synthetic_ratio,
            seed=seed,
            configs_dir=configs_dir,
            outputs_dir=outputs_dir,
            generation_id=generation_id,
            training_overrides=training_overrides,
            force=force,
            provision=provision and index == 0,  # provision once, reuse thereafter
            evaluate=evaluate,
            make_plots=make_plots,
            all_subsets=False,
        )
        run_dirs.append(run_dir)

    # The payoff figure: accuracy against real-data fraction across the sweep.
    if make_plots:
        from .plots import aggregate_runs, plot_data_scaling

        frame = aggregate_runs(outputs_dir)
        metric = "test_accuracy" if evaluate else "val_accuracy"
        if not frame.empty and metric in frame.columns:
            plot_data_scaling(
                frame, Path(outputs_dir) / f"data_scaling_{metric}.png", metric=metric
            )
    return run_dirs


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a previously saved resolved run config."""
    return load_yaml(path)
