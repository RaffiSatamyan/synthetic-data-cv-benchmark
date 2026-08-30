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


def confusion_matrix(
    true_labels: Iterable[int], predicted_labels: Iterable[int], num_classes: int
) -> list[list[int]]:
    """Row = true class, column = predicted class."""
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true, predicted in zip(true_labels, predicted_labels, strict=True):
        matrix[int(true)][int(predicted)] += 1
    return matrix


def per_class_accuracy(
    true_labels: Iterable[int], predicted_labels: Iterable[int], num_classes: int
) -> list[float | None]:
    """Recall per class; ``None`` when a class never appears in the truth."""
    correct = [0] * num_classes
    total = [0] * num_classes
    for true, predicted in zip(true_labels, predicted_labels, strict=True):
        total[int(true)] += 1
        if int(true) == int(predicted):
            correct[int(true)] += 1
    return [correct[i] / total[i] if total[i] else None for i in range(num_classes)]


def collect_predictions(
    model,
    loader,
    device,
    *,
    sample_ids: list | None = None,
    index_to_label: dict[int, Any] | None = None,
    topk: int = 5,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Run the model over a loader and return one record per sample.

    Assumes ``loader`` is unshuffled so position ``i`` maps to ``sample_ids[i]``.
    Each record carries the predicted label, its confidence, and the top-k labels
    with their probabilities, satisfying the "log every prediction" requirement.
    """
    import torch

    index_to_label = index_to_label or {}
    model.eval()
    records: list[dict[str, Any]] = []
    true_indices: list[int] = []
    predicted_indices: list[int] = []
    position = 0
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(device, non_blocking=True))
            probabilities = torch.softmax(logits.float(), dim=1)
            confidences, predictions = probabilities.max(dim=1)
            k = min(topk, probabilities.size(1))
            top_probs, top_indices = probabilities.topk(k, dim=1)
            for row in range(inputs.size(0)):
                true_index = int(targets[row])
                predicted_index = int(predictions[row])
                true_indices.append(true_index)
                predicted_indices.append(predicted_index)
                sample_id = (
                    sample_ids[position]
                    if sample_ids is not None and position < len(sample_ids)
                    else position
                )
                top_label_indices = [int(value) for value in top_indices[row].tolist()]
                records.append(
                    {
                        "sample_id": sample_id,
                        "true_index": true_index,
                        "true_label": index_to_label.get(true_index, true_index),
                        "predicted_index": predicted_index,
                        "predicted_label": index_to_label.get(predicted_index, predicted_index),
                        "correct": int(true_index == predicted_index),
                        "confidence": round(float(confidences[row]), 6),
                        "topk_labels": ";".join(
                            str(index_to_label.get(value, value)) for value in top_label_indices
                        ),
                        "topk_probs": ";".join(
                            f"{float(value):.6f}" for value in top_probs[row].tolist()
                        ),
                    }
                )
                position += 1
    return records, true_indices, predicted_indices


def topk_accuracy(records: Iterable[dict[str, Any]], k: int = 5) -> float:
    """Fraction of records whose true label is within the stored top-k labels."""
    records = list(records)
    if not records:
        return 0.0
    hits = 0
    for record in records:
        labels = str(record.get("topk_labels", "")).split(";")[:k]
        if str(record.get("true_label")) in labels:
            hits += 1
    return hits / len(records)


def save_predictions_csv(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write per-sample prediction records to a CSV with a fixed column order."""
    import csv

    records = list(records)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "true_index",
        "true_label",
        "predicted_index",
        "predicted_label",
        "correct",
        "confidence",
        "topk_labels",
        "topk_probs",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})
    return destination


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
    image_root = Path(dataset_cfg.get("image_root", data_root))

    label_map = {key: int(value) for key, value in _load_json(run_dir / "label_map.json").items()}
    index_to_label = {index: label for label, index in label_map.items()}
    num_classes = int(dataset_cfg["num_classes"])
    test = load_manifest(dataset_cfg["splits"]["test"])
    # label_map keys were serialized as strings; match on the string form.
    normalized = test.copy()
    normalized[label_column] = normalized[label_column].astype(str)
    sample_ids = (
        normalized["sample_id"].tolist() if "sample_id" in normalized.columns else None
    )

    model_cfg["num_classes"] = num_classes
    model = build_model(model_cfg)
    load_checkpoint(checkpoint, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = build_image_dataset(
        normalized, label_map, build_eval_transforms(dataset_cfg), image_root, label_column
    )
    loader = DataLoader(dataset, batch_size=int(config["training"].get("batch_size", 64)))

    records, true_labels, predicted_labels = collect_predictions(
        model, loader, device, sample_ids=sample_ids, index_to_label=index_to_label
    )
    metrics = classification_metrics(true_labels, predicted_labels)

    existing = _load_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    existing["test_accuracy"] = metrics["accuracy"]
    existing["test_macro_f1"] = metrics["macro_f1"]
    existing["test_top5_accuracy"] = topk_accuracy(records, k=5)
    existing["num_test_samples"] = len(true_labels)
    save_metrics(run_dir, existing)

    # Confusion matrix and per-class accuracy back the diagnostic graphs.
    write_json(
        run_dir / "test_confusion_matrix.json",
        {
            "labels": [index_to_label.get(i, i) for i in range(num_classes)],
            "matrix": confusion_matrix(true_labels, predicted_labels, num_classes),
            "per_class_accuracy": per_class_accuracy(
                true_labels, predicted_labels, num_classes
            ),
        },
    )

    if save_predictions:
        save_predictions_csv(run_dir / "predictions.csv", records)

    return existing
