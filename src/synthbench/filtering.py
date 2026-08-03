from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from .utils import sha256_file


def filter_synthetic_manifest(
    manifest: pd.DataFrame,
    generation_dir: str | Path,
    min_annotation_confidence: float | None = None,
) -> pd.DataFrame:
    """Mark invalid samples without deleting them, preserving provenance."""
    result = manifest.copy()
    root = Path(generation_dir)
    result["accepted"] = result.get("accepted", True).astype(bool)
    result["rejection_reason"] = result.get("rejection_reason", "").fillna("").astype(str)

    seen_hashes: set[str] = set()
    for index, row in result.iterrows():
        if not result.at[index, "accepted"]:
            continue

        relative_path = row.get("image_path", "")
        image_path = root / relative_path if relative_path else None
        reason = ""

        if image_path is None or not image_path.exists():
            reason = "missing_image"
        else:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError):
                reason = "corrupt_image"

        if not reason and image_path is not None:
            digest = row.get("image_sha256") or sha256_file(image_path)
            result.at[index, "image_sha256"] = digest
            if digest in seen_hashes:
                reason = "exact_duplicate"
            seen_hashes.add(digest)

        if (
            not reason
            and min_annotation_confidence is not None
            and "annotation_confidence" in result
            and float(row.get("annotation_confidence", 0.0)) < min_annotation_confidence
        ):
            reason = "low_annotation_confidence"

        if reason:
            result.at[index, "accepted"] = False
            result.at[index, "rejection_reason"] = reason

    return result
