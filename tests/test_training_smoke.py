"""End-to-end smoke test for the classification training + evaluation pipeline.

Skips automatically when the training extras (torch/torchvision) are absent,
so the default CI (dev-only) stays green while local GPU/CPU runs verify it.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image

from synthbench.evaluation import evaluate_run
from synthbench.training import train_experiment

CLASSES = ["c0", "c1", "c2"]


def _make_dataset(root):
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    index = 0
    # 6 train, 2 val, 2 test per class -> every split has all classes.
    for split, count in (("train", 6), ("val", 2), ("test", 2)):
        split_rows = []
        for label in CLASSES:
            for _ in range(count):
                name = f"{split}_{index}.png"
                Image.new("RGB", (40, 40), color=(index % 255, 40, 80)).save(images / name)
                split_rows.append(
                    {"sample_id": f"s{index}", "image_path": f"images/{name}", "label": label}
                )
                index += 1
        rows.append((split, pd.DataFrame(split_rows)))
    return dict(rows)


def _write_splits(root, frames):
    splits = root / "splits"
    (splits / "full").mkdir(parents=True, exist_ok=True)
    frames["train"].to_csv(splits / "full" / "train.csv", index=False)
    frames["val"].to_csv(splits / "val.csv", index=False)
    frames["test"].to_csv(splits / "test.csv", index=False)
    return {
        "full": str(splits / "full" / "train.csv"),
        "validation": str(splits / "val.csv"),
        "test": str(splits / "test.csv"),
    }


def _config(root, split_paths, generator):
    return {
        "dataset": {
            "name": "tinyset",
            "task": "classification",
            "num_classes": len(CLASSES),
            "label_column": "label",
            "data_root": str(root),
            "splits": split_paths,
            "image": {"size": 32},
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "model": {"name": "resnet18", "task": "classification", "pretrained": False},
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_workers": 0,
            "mixed_precision": False,
            "seed": 0,
        },
        "generator": generator,
        "experiment": {
            "dataset": "tinyset",
            "model": "resnet18",
            "generator": generator["name"],
            "real_fraction": 1.0,
            "synthetic_ratio": 0.0,
            "seed": 0,
            "run_id": "smoke",
        },
    }


def test_train_and_evaluate_end_to_end(tmp_path):
    root = tmp_path / "data" / "tinyset"
    split_paths = _write_splits(root, _make_dataset(root))
    generator = {"name": "real_only", "type": "baseline", "enabled": False}
    config = _config(root, split_paths, generator)

    run_dir = train_experiment(config, tmp_path / "outputs" / "smoke")

    for name in ("config.yaml", "best.pt", "last.pt", "metrics.json", "label_map.json", "train.log"):
        assert (run_dir / name).exists(), f"missing {name}"

    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["num_train_samples"] == len(CLASSES) * 6
    assert metrics["val_accuracy"] is not None

    result = evaluate_run(run_dir, save_predictions=True)
    assert "test_accuracy" in result
    assert (run_dir / "predictions.csv").exists()


def test_classical_augmentation_with_mixup_runs(tmp_path):
    root = tmp_path / "data" / "tinyset"
    split_paths = _write_splits(root, _make_dataset(root))
    generator = {
        "name": "classical_augmentation",
        "type": "classical_augmentation",
        "enabled": True,
        "selected_method": "mixup",
        "methods": {"mixup": {"alpha": 0.2, "probability": 1.0}},
    }
    config = _config(root, split_paths, generator)
    config["experiment"]["generator"] = "classical_augmentation"

    run_dir = train_experiment(config, tmp_path / "outputs" / "aug")
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
