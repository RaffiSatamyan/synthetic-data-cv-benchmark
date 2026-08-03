from __future__ import annotations

import argparse
from pathlib import Path

from synthbench.config import load_yaml
from synthbench.generation import GenerationRequest, PlaceholderGenerator, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one synthetic dataset")
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--generator-config", type=Path, required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Use the deterministic test generator. Real adapters must be implemented.",
    )
    args = parser.parse_args()

    dataset = load_yaml(args.dataset_config)
    generator = load_yaml(args.generator_config)
    output_dir = Path(dataset["root"]) / "synthetic" / generator["name"] / args.generation_id

    if not args.placeholder:
        raise NotImplementedError(
            "Add the selected generator adapter in src/synthbench/generation.py. "
            "Use --placeholder only to test storage and manifests."
        )

    requests = [
        GenerationRequest(
            sample_id=f"syn_{index:06d}",
            seed=index,
            label=index % int(dataset.get("num_classes", 1)),
            prompt=f"placeholder sample {index}",
        )
        for index in range(args.num_samples)
    ]
    manifest = generate_dataset(
        PlaceholderGenerator(),
        requests,
        output_dir,
        {"dataset": dataset, "generator": generator},
    )
    print(f"Synthetic manifest: {manifest}")


if __name__ == "__main__":
    main()
