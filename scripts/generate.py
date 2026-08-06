from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthbench.config import load_yaml
from synthbench.generation import (
    generate_dataset,
    GenerationRequest,
    PlaceholderGenerator,
)


ALLOWED_SYNTHETIC_RATIOS = (0.5, 1.0, 2.0, 5.0, 10.0)


def build_generation_requests(
    real_manifest: pd.DataFrame,
    synthetic_ratio: float,
    label_column: str,
    prompt_template: str,
    base_seed: int,
) -> list[GenerationRequest]:
    """
    Create class-aware synthetic generation requests.

    A ratio of 2.0 creates approximately two synthetic samples
    for every real sample, while preserving class proportions.
    """

    if real_manifest.empty:
        raise ValueError("Real-data manifest is empty.")

    if label_column not in real_manifest.columns:
        raise ValueError(
            f"Manifest is missing label column: {label_column}"
        )

    if synthetic_ratio not in ALLOWED_SYNTHETIC_RATIOS:
        raise ValueError(
            "Synthetic ratio must be one of "
            f"{ALLOWED_SYNTHETIC_RATIOS}"
        )

    requests: list[GenerationRequest] = []
    sample_index = 0

    class_counts = real_manifest[label_column].value_counts(
        sort=False
    )

    for label, real_count in class_counts.items():
        synthetic_count = round(
            int(real_count) * synthetic_ratio
        )

        # Every class should have at least one generated image.
        synthetic_count = max(1, synthetic_count)

        prompt = prompt_template.format(label=label)

        for _ in range(synthetic_count):
            requests.append(
                GenerationRequest(
                    sample_id=f"syn_{sample_index:08d}",
                    seed=base_seed + sample_index,
                    label=label,
                    prompt=prompt,
                )
            )

            sample_index += 1

    return requests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one synthetic dataset"
    )

    parser.add_argument(
        "--dataset-config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--generator-config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--real-manifest",
        type=Path,
        required=True,
        help=(
            "Real subset manifest used to determine labels "
            "and synthetic sample count."
        ),
    )

    parser.add_argument(
        "--generation-id",
        required=True,
    )

    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        choices=ALLOWED_SYNTHETIC_RATIOS,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--placeholder",
        action="store_true",
        help=(
            "Use the deterministic test generator. "
            "Do not use it for final experiments."
        ),
    )

    args = parser.parse_args()

    dataset = load_yaml(args.dataset_config)
    generator = load_yaml(args.generator_config)

    dataset_root = dataset.get(
        "root",
        dataset.get("data_root"),
    )

    if dataset_root is None:
        raise ValueError(
            "Dataset config must define root or data_root."
        )

    label_column = dataset.get(
        "label_column",
        "label",
    )

    prompt_template = generator.get(
        "prompt_template",
        "a high-quality photo of a {label}",
    )

    real_manifest = pd.read_csv(
        args.real_manifest
    )

    requests = build_generation_requests(
        real_manifest=real_manifest,
        synthetic_ratio=args.synthetic_ratio,
        label_column=label_column,
        prompt_template=prompt_template,
        base_seed=args.seed,
    )

    output_dir = (
        Path(dataset_root)
        / "synthetic"
        / generator["name"]
        / args.generation_id
    )

    if not args.placeholder:
        raise NotImplementedError(
            "The selected real generator adapter has not "
            "been connected yet. Use --placeholder only "
            "for a storage and manifest smoke test."
        )

    generation_config = {
        "dataset": dataset,
        "generator": generator,
        "real_manifest": str(args.real_manifest),
        "synthetic_ratio": args.synthetic_ratio,
        "number_of_requests": len(requests),
        "base_seed": args.seed,
    }

    manifest = generate_dataset(
        adapter=PlaceholderGenerator(),
        requests=requests,
        output_dir=output_dir,
        generation_config=generation_config,
    )

    print(
        f"Created {len(requests)} generation requests "
        f"for synthetic ratio x{args.synthetic_ratio}"
    )
    print(f"Synthetic manifest: {manifest}")


if __name__ == "__main__":
    main()
