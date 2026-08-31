"""Tests for the experiment-runner additions: manifest provisioning, per-sample
prediction logging, resume-skip, plots, and the unified train() entry point.

The lightweight manifest/provision tests run everywhere; the ones that actually
train or plot skip when torch / torchvision / matplotlib are absent, matching
the existing smoke-test convention so dev-only CI stays green.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest
from PIL import Image

from synthbench.provision import build_image_manifest, ensure_dataset, select_classes

CLASSES = ["c0", "c1", "c2", "c3"]


def _make_imagefolder(root, counts):
    """Create <root>/<split>/<class>/*.png and return the root."""
    for split, count in counts.items():
        for label in CLASSES:
            directory = root / split / label
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                Image.new("RGB", (32, 32), color=(index * 7 % 255, 40, 90)).save(
                    directory / f"{label}_{index}.png"
                )
    return root


# --------------------------- provisioning ---------------------------


def test_build_image_manifest_columns_and_relative_paths(tmp_path):
    root = _make_imagefolder(tmp_path / "data", {"train": 3})
    manifest = build_image_manifest(
        root / "train", path_relative_to=root, split_prefix="train"
    )
    assert list(manifest.columns) == ["sample_id", "image_path", "label"]
    assert len(manifest) == len(CLASSES) * 3
    assert manifest["label"].nunique() == len(CLASSES)
    # Paths are stored relative to data_root, so manifests stay portable.
    assert manifest["image_path"].str.startswith("train/").all()
    # split_prefix keeps ids namespaced.
    assert manifest["sample_id"].str.startswith("train_").all()
    assert not manifest["sample_id"].duplicated().any()


def test_select_classes_max_and_explicit(tmp_path):
    root = _make_imagefolder(tmp_path / "data", {"train": 2})
    # "First N sorted" turns ImageNet-1k into ImageNet-100 deterministically.
    first_two = select_classes(root / "train", max_classes=2)
    assert [p.name for p in first_two] == ["c0", "c1"]
    # An explicit list restricts to exactly those classes.
    explicit = select_classes(root / "train", include_classes=["c1", "c3"])
    assert [p.name for p in explicit] == ["c1", "c3"]
    # A missing requested class is a hard error, not a silent drop.
    with pytest.raises(ValueError):
        select_classes(root / "train", include_classes=["c1", "nope"])


def test_build_image_manifest_subset_and_absolute(tmp_path):
    root = _make_imagefolder(tmp_path / "data", {"train": 2})
    manifest = build_image_manifest(
        root / "train", max_classes=2, absolute_paths=True
    )
    assert manifest["label"].nunique() == 2
    assert set(manifest["label"]) == {"c0", "c1"}
    # Absolute paths resolve for a reader that has no notion of data_root.
    assert manifest["image_path"].map(lambda p: os.path.isabs(p)).all()


def test_ensure_dataset_mounted_source_stores_portable_paths(tmp_path):
    # Simulate Kaggle: images on a read-only "mount" (image_root), splits in a
    # separate writable data_root. Stored paths must be RELATIVE to image_root so
    # the split is portable and can be committed once and reused anywhere.
    mount = _make_imagefolder(tmp_path / "mount" / "train_root", {"train": 6})
    image_root = mount / "train"
    data_root = tmp_path / "working" / "imagenet100"
    dataset_cfg = {
        "name": "in100",
        "task": "classification",
        "num_classes": 3,
        "label_column": "label",
        "data_root": str(data_root),
        "image_root": str(image_root),
        "splits": {
            "full": str(data_root / "splits" / "full" / "train.csv"),
            "validation": str(data_root / "splits" / "val.csv"),
            "test": str(data_root / "splits" / "test.csv"),
        },
        "real_fractions": [0.5, 1.0],
        "provision": {
            "manifest_sources": [{"path": str(image_root), "max_classes": 3}],
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "seed": 42,
        },
    }
    ensure_dataset(dataset_cfg)
    assert (data_root / "splits" / "full" / "train.csv").exists()
    manifest = pd.read_csv(data_root / "source_manifest.csv")
    assert manifest["label"].nunique() == 3  # only the first 3 of 4 classes
    # Portable: relative paths that resolve under image_root, not absolute.
    assert not manifest["image_path"].map(lambda p: os.path.isabs(p)).any()
    first = manifest.iloc[0]
    assert (image_root / first["image_path"]).exists()


def test_build_image_manifest_rejects_empty(tmp_path):
    empty = tmp_path / "empty"
    (empty / "c0").mkdir(parents=True)
    with pytest.raises(ValueError):
        build_image_manifest(empty)


def test_ensure_dataset_builds_splits(tmp_path):
    root = _make_imagefolder(tmp_path / "data" / "tiny", {"train": 6, "val": 3})
    dataset_cfg = {
        "name": "tiny",
        "task": "classification",
        "num_classes": len(CLASSES),
        "label_column": "label",
        "data_root": str(root),
        "splits": {
            "full": str(root / "splits" / "full" / "train.csv"),
            "validation": str(root / "splits" / "val.csv"),
            "test": str(root / "splits" / "test.csv"),
        },
        "real_fractions": [0.5, 1.0],
        "provision": {
            "manifest_sources": [
                {"path": "train", "prefix": "train"},
                {"path": "val", "prefix": "val"},
            ],
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "seed": 42,
        },
    }
    ensure_dataset(dataset_cfg)
    for relative in ("splits/full/train.csv", "splits/val.csv", "splits/test.csv"):
        assert (root / relative).exists(), f"missing {relative}"
    assert (root / "source_manifest.csv").exists()
    # Idempotent: a second call is a no-op because the full split now exists.
    ensure_dataset(dataset_cfg)


def test_model_recipe_merges_into_training():
    # resnet18.yaml's paper recipe (SGD/step/90ep/bs256) must override the shared
    # training defaults when a run config is resolved from the repo config tree.
    from pathlib import Path

    from synthbench.training import resolve_experiment_config

    configs = Path(__file__).resolve().parents[1] / "configs"
    row = {
        "dataset": "imagenet100",
        "model": "resnet18",
        "generator": "real_only",
        "real_fraction": 1.0,
        "synthetic_ratio": 0.0,
        "seed": 0,
        "run_id": "recipe-check",
    }
    config = resolve_experiment_config(row, configs)
    training = config["training"]
    assert training["optimizer"]["name"] == "sgd"
    assert training["optimizer"]["lr"] == 0.1
    assert training["scheduler"]["name"] == "step"
    assert training["epochs"] == 90
    assert training["batch_size"] == 256
    assert config["model"]["pretrained"] is False


def test_ensure_dataset_without_provision_raises(tmp_path):
    dataset_cfg = {
        "name": "tiny",
        "data_root": str(tmp_path / "nope"),
        "splits": {"full": str(tmp_path / "nope" / "splits" / "full" / "train.csv")},
    }
    with pytest.raises(FileNotFoundError):
        ensure_dataset(dataset_cfg)


# --------------------------- training-dependent ---------------------------
# torch/torchvision are imported per-test (not at module scope) so the pure
# manifest/provision tests above still run in the dev-only CI without torch.


def _require_training():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")


def _dataset_from_folder(root, num_classes):
    """Build split manifests directly and return a dataset config for training."""
    ensure_cfg = {
        "name": "tinyset",
        "task": "classification",
        "num_classes": num_classes,
        "label_column": "label",
        "data_root": str(root),
        "splits": {
            "full": str(root / "splits" / "full" / "train.csv"),
            "validation": str(root / "splits" / "val.csv"),
            "test": str(root / "splits" / "test.csv"),
        },
        "real_fractions": [1.0],
        "provision": {
            "manifest_sources": [
                {"path": "train", "prefix": "train"},
                {"path": "val", "prefix": "val"},
                {"path": "test", "prefix": "test"},
            ],
            "validation_fraction": 0.25,
            "test_fraction": 0.25,
            "seed": 0,
        },
        "image": {"size": 32},
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    ensure_dataset(ensure_cfg)
    return ensure_cfg


def _config(dataset_cfg):
    return {
        "dataset": dataset_cfg,
        "model": {"name": "resnet18", "task": "classification", "pretrained": False},
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_workers": 0,
            "mixed_precision": False,
            "seed": 0,
        },
        "generator": {"name": "real_only", "type": "baseline", "enabled": False},
        "experiment": {
            "dataset": "tinyset",
            "model": "resnet18",
            "generator": "real_only",
            "real_fraction": 1.0,
            "synthetic_ratio": 0.0,
            "seed": 0,
            "run_id": "runner",
        },
    }


def test_build_optimizer_and_scheduler_from_recipe():
    _require_training()
    import torch

    from synthbench.training import build_optimizer, build_scheduler

    model = torch.nn.Linear(4, 3)
    recipe = {
        "optimizer": {"name": "sgd", "lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        "scheduler": {"name": "step", "step_size": 30, "gamma": 0.1},
    }
    optimizer = build_optimizer(model, recipe)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[0]["momentum"] == 0.9
    scheduler = build_scheduler(optimizer, recipe, epochs=90)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    # No recipe block -> original AdamW + cosine defaults.
    fallback_opt = build_optimizer(model, {"learning_rate": 1e-3})
    assert isinstance(fallback_opt, torch.optim.AdamW)
    fallback_sched = build_scheduler(fallback_opt, {}, epochs=10)
    assert isinstance(fallback_sched, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_per_sample_logging_and_history(tmp_path):
    _require_training()
    from synthbench.training import train_experiment

    root = _make_imagefolder(tmp_path / "data" / "tinyset", {"train": 8, "val": 4, "test": 4})
    config = _config(_dataset_from_folder(root, len(CLASSES)))
    run_dir = train_experiment(config, tmp_path / "outputs" / "runner")

    history = pd.read_csv(run_dir / "history.csv")
    assert list(history.columns) == [
        "epoch",
        "train_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_top5_accuracy",
        "learning_rate",
        "epoch_seconds",
    ]
    assert len(history) == 2

    predictions = pd.read_csv(run_dir / "predictions_val.csv")
    for column in ("sample_id", "true_label", "predicted_label", "confidence", "topk_labels"):
        assert column in predictions.columns
    assert (predictions["confidence"] >= 0).all() and (predictions["confidence"] <= 1).all()

    confusion = json.loads((run_dir / "val_confusion_matrix.json").read_text())
    assert len(confusion["matrix"]) == len(CLASSES)


def test_resume_skips_completed_run(tmp_path):
    _require_training()
    from synthbench.training import train_experiment

    root = _make_imagefolder(tmp_path / "data" / "tinyset", {"train": 8, "val": 4, "test": 4})
    config = _config(_dataset_from_folder(root, len(CLASSES)))
    out = tmp_path / "outputs" / "runner"

    run_dir = train_experiment(config, out)
    first_mtime = (run_dir / "best.pt").stat().st_mtime_ns

    # A second call with the same config must skip and leave the checkpoint untouched.
    again = train_experiment(config, out)
    assert again == run_dir
    assert (run_dir / "best.pt").stat().st_mtime_ns == first_mtime

    # force=True retrains and rewrites the checkpoint.
    forced = train_experiment(config, out, force=True)
    assert forced == run_dir
    assert (run_dir / "best.pt").stat().st_mtime_ns != first_mtime


def test_all_subsets_sweep(tmp_path):
    _require_training()
    pytest.importorskip("matplotlib")
    import yaml

    from synthbench.provision import ensure_dataset
    from synthbench.training import train

    root = _make_imagefolder(tmp_path / "data" / "tinyset", {"train": 12, "val": 4, "test": 4})
    dataset_cfg = {
        "name": "tinyset",
        "task": "classification",
        "num_classes": len(CLASSES),
        "label_column": "label",
        "data_root": str(root),
        "splits": {
            "full": str(root / "splits" / "full" / "train.csv"),
            "50pct": str(root / "splits" / "50pct" / "train.csv"),
            "validation": str(root / "splits" / "val.csv"),
            "test": str(root / "splits" / "test.csv"),
        },
        "real_fractions": [0.5, 1.0],  # two subsets to sweep
        "provision": {
            "manifest_sources": [
                {"path": "train", "prefix": "train"},
                {"path": "val", "prefix": "val"},
                {"path": "test", "prefix": "test"},
            ],
            "validation_fraction": 0.25,
            "test_fraction": 0.25,
            "seed": 0,
        },
        "image": {"size": 32},
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    ensure_dataset(dataset_cfg)

    configs = tmp_path / "configs"
    (configs / "datasets").mkdir(parents=True)
    (configs / "models").mkdir()
    (configs / "generators").mkdir()
    (configs / "datasets" / "tinyset.yaml").write_text(yaml.safe_dump(dataset_cfg))
    (configs / "models" / "resnet18.yaml").write_text(
        yaml.safe_dump({"name": "resnet18", "task": "classification", "pretrained": False})
    )
    (configs / "generators" / "real_only.yaml").write_text(
        yaml.safe_dump({"name": "real_only", "type": "baseline", "enabled": False})
    )
    (configs / "training.yaml").write_text(
        yaml.safe_dump({"epochs": 1, "batch_size": 4, "num_workers": 0, "seed": 0})
    )

    outputs = tmp_path / "outputs"
    run_dirs = train(
        "tinyset", "resnet18", configs_dir=configs, outputs_dir=outputs,
        provision=False, all_subsets=True,
    )
    # One independent run per fraction, plus the cross-run data-scaling figure.
    assert isinstance(run_dirs, list) and len(run_dirs) == 2
    assert len({p.name for p in run_dirs}) == 2  # distinct run ids per fraction
    for path in run_dirs:
        assert (path / "metrics.json").exists()
    assert (outputs / "data_scaling_test_accuracy.png").exists()


def test_train_wrapper_end_to_end(tmp_path):
    _require_training()
    pytest.importorskip("matplotlib")
    import yaml

    from synthbench.training import train

    root = _make_imagefolder(tmp_path / "data" / "tinyset", {"train": 8, "val": 4, "test": 4})
    dataset_cfg = _dataset_from_folder(root, len(CLASSES))

    configs = tmp_path / "configs"
    (configs / "datasets").mkdir(parents=True)
    (configs / "models").mkdir()
    (configs / "generators").mkdir()
    (configs / "datasets" / "tinyset.yaml").write_text(yaml.safe_dump(dataset_cfg))
    (configs / "models" / "resnet18.yaml").write_text(
        yaml.safe_dump({"name": "resnet18", "task": "classification", "pretrained": False})
    )
    (configs / "generators" / "real_only.yaml").write_text(
        yaml.safe_dump({"name": "real_only", "type": "baseline", "enabled": False})
    )
    (configs / "training.yaml").write_text(
        yaml.safe_dump({"epochs": 1, "batch_size": 4, "num_workers": 0, "seed": 0})
    )

    run_dir = train(
        "tinyset",
        "resnet18",
        configs_dir=configs,
        outputs_dir=tmp_path / "outputs",
        provision=False,
    )
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "test_accuracy" in metrics
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "figures" / "training_curves.png").exists()
