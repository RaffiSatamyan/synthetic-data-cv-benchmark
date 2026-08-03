from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .utils import sha256_file, write_yaml


@dataclass(frozen=True)
class GenerationRequest:
    sample_id: str
    seed: int
    label: str | int | None = None
    prompt: str | None = None
    conditioning_path: str | None = None


@dataclass
class GeneratedSample:
    sample_id: str
    image_path: str
    generation_seed: int
    generator: str
    label: str | int | None = None
    prompt: str | None = None
    annotation_path: str | None = None
    accepted: bool = True
    rejection_reason: str = ""
    image_sha256: str = ""


class GeneratorAdapter(ABC):
    """All generator implementations must follow this interface."""

    name: str

    def load(self) -> None:
        """Load weights. Override when explicit loading is required."""

    @abstractmethod
    def generate_one(self, request: GenerationRequest) -> Image.Image:
        """Generate one RGB image for a request."""

    def close(self) -> None:
        """Release model/GPU resources. Override when needed."""


class PlaceholderGenerator(GeneratorAdapter):
    """Deterministic generator for pipeline tests, not paper experiments."""

    name = "placeholder"

    def generate_one(self, request: GenerationRequest) -> Image.Image:
        value = request.seed % 256
        return Image.new("RGB", (64, 64), (value, value, value))


def generate_dataset(
    adapter: GeneratorAdapter,
    requests: Iterable[GenerationRequest],
    output_dir: str | Path,
    generation_config: dict[str, Any],
    resume: bool = True,
) -> Path:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    config_path = output_dir / "config.yaml"

    completed = _completed_ids(manifest_path) if resume else set()
    fieldnames = list(GeneratedSample.__dataclass_fields__)

    adapter.load()
    try:
        write_header = not manifest_path.exists() or not resume
        mode = "a" if resume else "w"
        with manifest_path.open(mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for request in requests:
                if request.sample_id in completed:
                    continue

                image_path = images_dir / f"{request.sample_id}.png"
                try:
                    image = adapter.generate_one(request).convert("RGB")
                    image.save(image_path)
                    sample = GeneratedSample(
                        sample_id=request.sample_id,
                        image_path=str(image_path.relative_to(output_dir)),
                        generation_seed=request.seed,
                        generator=adapter.name,
                        label=request.label,
                        prompt=request.prompt,
                        image_sha256=sha256_file(image_path),
                    )
                except Exception as error:
                    sample = GeneratedSample(
                        sample_id=request.sample_id,
                        image_path="",
                        generation_seed=request.seed,
                        generator=adapter.name,
                        label=request.label,
                        prompt=request.prompt,
                        accepted=False,
                        rejection_reason=f"generation_error:{type(error).__name__}",
                    )
                writer.writerow(asdict(sample))
                handle.flush()
    finally:
        adapter.close()

    write_yaml(config_path, generation_config)
    return manifest_path


def _completed_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"] for row in csv.DictReader(handle)}
