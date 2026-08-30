"""Build a source manifest (sample_id, image_path, label) from an image tree.

Scans one or more ImageFolder-style directories (``<root>/<class>/<image>``)
and writes a single CSV that ``scripts/prepare_data.py`` can split. Use this to
turn a freshly downloaded ImageNet-100 (train/ + val/ class folders) into the
manifest the rest of the pipeline expects.

Example:
    python scripts/build_manifest.py \
        --data-root data/imagenet100 \
        --source data/imagenet100/train:train \
        --source data/imagenet100/val:val \
        --output data/imagenet100/source_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthbench.provision import build_image_manifest


def _parse_source(value: str) -> tuple[str, str | None]:
    """Parse ``path[:prefix]`` into (path, prefix)."""
    if ":" in value and not Path(value).exists():
        path, prefix = value.rsplit(":", 1)
        return path, prefix
    return value, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a source manifest from image folders")
    parser.add_argument("--data-root", type=Path, required=True, help="Root images are relative to")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Image directory, optionally 'path:prefix' to namespace sample ids. Repeatable.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    for source in args.source:
        path, prefix = _parse_source(source)
        directory = args.data_root / path if not Path(path).is_absolute() else Path(path)
        frames.append(
            build_image_manifest(directory, path_relative_to=args.data_root, split_prefix=prefix)
        )

    manifest = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    classes = manifest["label"].nunique()
    print(f"Wrote {len(manifest)} rows ({classes} classes) to {args.output}")


if __name__ == "__main__":
    main()
