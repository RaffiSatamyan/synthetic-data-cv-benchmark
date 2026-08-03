from __future__ import annotations

from itertools import product


def build_grid(preset: str) -> list[dict]:
    """Build a deduplicated pilot or full experiment grid."""
    if preset not in {"pilot", "full"}:
        raise ValueError("preset must be 'pilot' or 'full'")

    datasets = [
        ("classification", "cifar10"),
        ("classification", "eurosat"),
        ("detection", "voc"),
        ("detection", "coco_subset"),
        ("segmentation", "cityscapes"),
        ("segmentation", "voc_segmentation"),
    ]
    fractions = [0.01, 0.05, 0.10, 1.0]
    seeds = [0, 1, 2]
    generators = ["dcgan", "stylegan2", "stable_diffusion"]
    ratios = [0.5, 1.0, 2.0]

    if preset == "pilot":
        datasets = [("classification", "cifar10"), ("detection", "voc"), ("segmentation", "cityscapes")]
        fractions = [0.10, 1.0]
        seeds = [0]
        generators = ["stable_diffusion"]
        ratios = [1.0]

    rows: list[dict] = []
    for task, dataset in datasets:
        for fraction, seed in product(fractions, seeds):
            baselines = ["real_only", "classical_augmentation"]
            if preset == "full":
                baselines.append("strong_augmentation")
            for regime in baselines:
                rows.append({"task": task, "dataset": dataset, "real_fraction": fraction,
                             "seed": seed, "regime": regime, "generator": "none",
                             "synthetic_ratio": 0.0})
            for generator, ratio in product(generators, ratios):
                rows.append({"task": task, "dataset": dataset, "real_fraction": fraction,
                             "seed": seed, "regime": "synthetic", "generator": generator,
                             "synthetic_ratio": ratio})
    return rows
