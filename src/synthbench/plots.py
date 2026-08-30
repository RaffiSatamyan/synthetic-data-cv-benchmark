"""Figures for a run and for the whole experiment sweep.

Per-run plots (learning curves, confusion matrix, per-class accuracy) are built
from the artifacts ``train_experiment`` writes: ``history.csv`` and the
``*_confusion_matrix.json`` files. The cross-run plot answers the project's
core question -- for which dataset size does augmentation / synthetic data help
-- by charting accuracy against the real-data fraction.

matplotlib is imported lazily with the non-interactive Agg backend so these work
on a headless remote box. Install with: pip install -e ".[viz]".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('Install plotting deps: pip install -e ".[viz]"') from error
    return plt


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def plot_training_curves(run_dir: str | Path, output: str | Path | None = None) -> Path:
    """Loss and validation-accuracy curves over epochs from history.csv."""
    run_dir = Path(run_dir)
    history = pd.read_csv(run_dir / "history.csv")
    plt = _pyplot()

    figure, left = plt.subplots(figsize=(7, 4.5))
    right = left.twinx()
    left.plot(history["epoch"], history["train_loss"], color="tab:red", label="train_loss")
    right.plot(
        history["epoch"], history["val_accuracy"], color="tab:blue", label="val_accuracy"
    )
    left.set_xlabel("epoch")
    left.set_ylabel("train loss", color="tab:red")
    right.set_ylabel("val accuracy", color="tab:blue")
    left.set_ylim(bottom=0)
    right.set_ylim(0, 1)
    figure.suptitle(f"Training curves — {run_dir.name}")
    figure.tight_layout()

    output = Path(output) if output else run_dir / "figures" / "training_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def plot_confusion_matrix(
    run_dir: str | Path, split: str = "val", output: str | Path | None = None
) -> Path:
    """Heatmap of the stored confusion matrix for ``val`` or ``test``."""
    run_dir = Path(run_dir)
    payload = _load_json(run_dir / f"{split}_confusion_matrix.json")
    matrix = payload["matrix"]
    labels = [str(label) for label in payload["labels"]]
    plt = _pyplot()

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title(f"{split} confusion — {run_dir.name}")
    if len(labels) <= 25:
        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=90, fontsize=7)
        axis.set_yticklabels(labels, fontsize=7)
    figure.tight_layout()

    output = Path(output) if output else run_dir / "figures" / f"confusion_{split}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def plot_per_class_accuracy(
    run_dir: str | Path, split: str = "val", output: str | Path | None = None
) -> Path:
    """Bar chart of per-class accuracy for ``val`` or ``test``."""
    run_dir = Path(run_dir)
    payload = _load_json(run_dir / f"{split}_confusion_matrix.json")
    labels = [str(label) for label in payload["labels"]]
    accuracies = [value if value is not None else 0.0 for value in payload["per_class_accuracy"]]
    plt = _pyplot()

    figure, axis = plt.subplots(figsize=(max(6.0, len(labels) * 0.22), 4.0))
    axis.bar(range(len(labels)), accuracies, color="tab:green")
    axis.set_ylim(0, 1)
    axis.set_ylabel("accuracy")
    axis.set_title(f"{split} per-class accuracy — {run_dir.name}")
    if len(labels) <= 40:
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=90, fontsize=7)
    figure.tight_layout()

    output = Path(output) if output else run_dir / "figures" / f"per_class_accuracy_{split}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def plot_run(run_dir: str | Path) -> list[Path]:
    """Build every available per-run figure into ``run_dir/figures/``."""
    run_dir = Path(run_dir)
    outputs: list[Path] = []
    if (run_dir / "history.csv").exists():
        outputs.append(plot_training_curves(run_dir))
    for split in ("val", "test"):
        if (run_dir / f"{split}_confusion_matrix.json").exists():
            outputs.append(plot_confusion_matrix(run_dir, split))
            outputs.append(plot_per_class_accuracy(run_dir, split))
    return outputs


def aggregate_runs(outputs_dir: str | Path) -> pd.DataFrame:
    """Collect every completed run's metrics.json under ``outputs_dir`` into a frame."""
    outputs_dir = Path(outputs_dir)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(outputs_dir.glob("*/metrics.json")):
        try:
            rows.append(_load_json(metrics_path))
        except (OSError, json.JSONDecodeError):
            continue
    return pd.DataFrame(rows)


def plot_data_scaling(
    frame: pd.DataFrame,
    output: str | Path,
    *,
    metric: str = "test_accuracy",
) -> Path:
    """Accuracy vs real-data fraction, one line per generator × synthetic ratio.

    This is the headline figure: it shows at which dataset sizes a given
    augmentation or synthetic-data setting beats the real-only baseline.
    """
    if metric not in frame.columns:
        raise ValueError(f"metric {metric!r} not in aggregated columns: {list(frame.columns)}")
    plt = _pyplot()
    working = frame.dropna(subset=[metric, "real_fraction"]).copy()

    figure, axis = plt.subplots(figsize=(7.5, 5))
    group_keys = [key for key in ("generator", "synthetic_ratio") if key in working.columns]
    for name, group in working.groupby(group_keys) if group_keys else [("all", working)]:
        series = group.groupby("real_fraction")[metric].mean().sort_index()
        label = name if isinstance(name, str) else " / ".join(str(part) for part in name)
        axis.plot(series.index, series.values, marker="o", label=str(label))

    axis.set_xscale("log")
    axis.set_xlabel("real-data fraction")
    axis.set_ylabel(metric)
    axis.set_ylim(0, 1)
    axis.set_title("Data-scaling: accuracy vs real fraction")
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.legend(fontsize=8, title="generator / synth-ratio")
    figure.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output
