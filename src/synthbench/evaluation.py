from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

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


def evaluate_run(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing validation-selected checkpoint: {checkpoint}")
    raise NotImplementedError(
        "Load config.yaml and best.pt, build the real test loader, calculate task "
        "metrics, save metrics.json, and optionally predictions.csv. Never tune "
        "anything using the test set."
    )
