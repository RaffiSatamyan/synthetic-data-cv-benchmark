from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .utils import write_json


def classification_metrics(
    true_labels: Iterable[int], predicted_labels: Iterable[int]
) -> dict[str, float]:
    true = list(true_labels)
    predicted = list(predicted_labels)
    if len(true) != len(predicted) or not true:
        raise ValueError("true and predicted labels must have the same non-zero length")

    accuracy = sum(a == b for a, b in zip(true, predicted, strict=True)) / len(true)
    labels = sorted(set(true) | set(predicted))
    f1_values = []

    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(true, predicted, strict=True))
        fp = sum(a != label and b == label for a, b in zip(true, predicted, strict=True))
        fn = sum(a == label and b != label for a, b in zip(true, predicted, strict=True))
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else (2 * tp) / denominator)

    return {"accuracy": accuracy, "macro_f1": sum(f1_values) / len(f1_values)}


def save_metrics(run_dir: str | Path, metrics: dict) -> Path:
    destination = Path(run_dir) / "metrics.json"
    write_json(destination, metrics)
    return destination


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_run(run_dir: str | Path, save_predictions: bool = False) -> dict[str, Any]:
    """Evaluate a completed run on the untouched test split using best.pt."""
    import torch
    from torch.utils.data import DataLoader

    from .config import load_yaml
    from .data_loading import build_eval_transforms, build_image_dataset
    from .datasets import load_manifest
    from .models import build_model
    from .training import load_checkpoint

    run_dir = Path(run_dir)
    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing validation-selected checkpoint: {checkpoint}")

    config = load_yaml(run_dir / "config.yaml")
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    label_column = dataset_cfg.get("label_column", "label")
    data_root = Path(dataset_cfg["data_root"])

    label_map = {key: int(value) for key, value in _load_json(run_dir / "label_map.json").items()}
    test = load_manifest(dataset_cfg["splits"]["test"])
    # label_map keys were serialized as strings; match on the string form.
    normalized = test.copy()
    normalized[label_column] = normalized[label_column].astype(str)

    model_cfg["num_classes"] = int(dataset_cfg["num_classes"])
    model = build_model(model_cfg)
    load_checkpoint(checkpoint, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = build_image_dataset(
        normalized, label_map, build_eval_transforms(dataset_cfg), data_root, label_column
    )
    loader = DataLoader(dataset, batch_size=int(config["training"].get("batch_size", 64)))

    true_labels: list[int] = []
    predicted_labels: list[int] = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device))
            predicted_labels.extend(outputs.argmax(dim=1).cpu().tolist())
            true_labels.extend(int(value) for value in targets.tolist())

    metrics = classification_metrics(true_labels, predicted_labels)

    existing = _load_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    existing["test_accuracy"] = metrics["accuracy"]
    existing["test_macro_f1"] = metrics["macro_f1"]
    existing["num_test_samples"] = len(true_labels)
    save_metrics(run_dir, existing)

    if save_predictions:
        import csv

        with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "true_label", "predicted_label"])
            for index, (true, predicted) in enumerate(zip(true_labels, predicted_labels)):
                writer.writerow([index, true, predicted])

    return existing
