"""Image loading, transforms, and batch mixing for classification training.

This module is the bridge between the manifest/DataFrame layer in
``datasets.py`` and PyTorch. Torch and torchvision are imported lazily inside
each function so the package stays importable without the training extras.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_label_map(frame: pd.DataFrame, label_column: str) -> dict[Any, int]:
    """Map every label to a stable integer index, sorted for determinism.

    Build this from a split that contains all classes (e.g. validation) and
    reuse the same map for train/val/test so indices never drift between runs.
    """
    if label_column not in frame.columns:
        raise ValueError(f"Manifest is missing label column: {label_column!r}")
    labels = sorted(frame[label_column].dropna().unique(), key=str)
    return {label: index for index, label in enumerate(labels)}


def _image_params(dataset_config: dict[str, Any]) -> tuple[int, tuple, tuple]:
    size = int(dataset_config.get("image", {}).get("size", 224))
    normalization = dataset_config.get("normalization", {})
    mean = tuple(normalization.get("mean", _DEFAULT_MEAN))
    std = tuple(normalization.get("std", _DEFAULT_STD))
    return size, mean, std


def _augmentation_enabled(generator_config: dict[str, Any]) -> bool:
    if not generator_config.get("enabled", False):
        return False
    identity = generator_config.get("type") or generator_config.get("name")
    return identity == "classical_augmentation"


def _resolve_image_path(image_path: str, data_root: Path) -> Path:
    """Accept manifest paths that are absolute, or relative to the dataset root."""
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    joined = data_root / image_path
    if joined.exists():
        return joined
    return candidate


def build_eval_transforms(dataset_config: dict[str, Any]):
    """Deterministic resize + center crop + normalize for val/test."""
    from torchvision import transforms

    size, mean, std = _image_params(dataset_config)
    resize = round(size * 1.14)
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_train_transforms(dataset_config: dict[str, Any], generator_config: dict[str, Any]):
    """Strong augmentation only when the generator is classical_augmentation.

    real_only and synthetic generators fall back to the deterministic eval
    transform, so augmentation stays a clean, separable experimental axis.
    """
    from torchvision import transforms

    if not _augmentation_enabled(generator_config):
        return build_eval_transforms(dataset_config)

    size, mean, std = _image_params(dataset_config)
    methods = generator_config.get("methods", {})
    selected = generator_config.get("selected_method")

    crop = methods.get("random_resized_crop", {})
    steps: list[Any] = [
        transforms.RandomResizedCrop(
            size,
            scale=tuple(crop.get("scale", (0.08, 1.0))),
            ratio=tuple(crop.get("ratio", (0.75, 1.3333333))),
        ),
        transforms.RandomHorizontalFlip(
            methods.get("horizontal_flip", {}).get("probability", 0.5)
        ),
    ]

    if selected == "color_jitter":
        jitter = methods.get("color_jitter", {})
        steps.append(
            transforms.ColorJitter(
                brightness=jitter.get("brightness", 0.4),
                contrast=jitter.get("contrast", 0.4),
                saturation=jitter.get("saturation", 0.4),
                hue=jitter.get("hue", 0.1),
            )
        )
    elif selected == "randaugment":
        rand = methods.get("randaugment", {})
        steps.append(
            transforms.RandAugment(
                num_ops=int(rand.get("num_operations", 2)),
                magnitude=int(rand.get("magnitude", 9)),
            )
        )

    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


def build_image_dataset(
    frame: pd.DataFrame,
    label_map: dict[Any, int],
    transform,
    data_root: str | Path,
    label_column: str,
):
    """Build a torch Dataset of (image_tensor, label_index) from a manifest."""
    from PIL import Image
    from torch.utils.data import Dataset

    root = Path(data_root)
    samples: list[tuple[Path, int]] = []
    for record in frame.to_dict("records"):
        label = record[label_column]
        if label not in label_map:
            raise KeyError(f"Label {label!r} is not present in the label map")
        samples.append((_resolve_image_path(str(record["image_path"]), root), label_map[label]))

    class ImageManifestDataset(Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int):
            path, label = samples[index]
            with Image.open(path) as image:
                return transform(image.convert("RGB")), label

    return ImageManifestDataset()


def make_batch_mixer(
    generator_config: dict[str, Any],
) -> Callable | None:
    """Return a MixUp/CutMix batch mixer when that method is selected, else None."""
    if not _augmentation_enabled(generator_config):
        return None
    selected = generator_config.get("selected_method")
    if selected not in {"mixup", "cutmix"}:
        return None

    import numpy as np
    import torch

    config = generator_config.get("methods", {}).get(selected, {})
    alpha = float(config.get("alpha", 1.0))
    probability = float(config.get("probability", 1.0))
    mode = selected

    def mixer(inputs, targets):
        if alpha <= 0 or np.random.rand() > probability:
            return inputs, targets, targets, 1.0

        lam = float(np.random.beta(alpha, alpha))
        index = torch.randperm(inputs.size(0), device=inputs.device)
        target_a, target_b = targets, targets[index]

        if mode == "mixup":
            mixed = lam * inputs + (1.0 - lam) * inputs[index]
            return mixed, target_a, target_b, lam

        height, width = inputs.size(2), inputs.size(3)
        ratio = (1.0 - lam) ** 0.5
        cut_h, cut_w = int(height * ratio), int(width * ratio)
        center_y, center_x = np.random.randint(height), np.random.randint(width)
        y1, y2 = max(center_y - cut_h // 2, 0), min(center_y + cut_h // 2, height)
        x1, x2 = max(center_x - cut_w // 2, 0), min(center_x + cut_w // 2, width)
        mixed = inputs.clone()
        mixed[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
        return mixed, target_a, target_b, lam

    return mixer


def mixup_loss(criterion, outputs, target_a, target_b, lam: float):
    """Soft-label loss for a mixed batch."""
    return lam * criterion(outputs, target_a) + (1.0 - lam) * criterion(outputs, target_b)
