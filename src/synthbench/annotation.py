from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class Annotator(ABC):
    """Converts generated images into task labels, boxes, or masks."""

    @abstractmethod
    def annotate(self, manifest: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
        raise NotImplementedError


class ClassificationAnnotator(Annotator):
    """Classification generation already knows the requested class label."""

    def annotate(self, manifest: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
        if "label" not in manifest:
            raise ValueError("Classification manifest must contain a label column")
        result = manifest.copy()
        result["annotation_method"] = "requested_class"
        result["annotation_confidence"] = 1.0
        return result


class TeacherModelAnnotator(Annotator):
    """Extension point for detection and segmentation teacher models."""

    def __init__(self, checkpoint: str | Path, confidence_threshold: float = 0.5):
        self.checkpoint = Path(checkpoint)
        self.confidence_threshold = confidence_threshold

    def load_teacher(self) -> Any:
        raise NotImplementedError("Implement teacher loading for the selected task")

    def annotate(self, manifest: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
        raise NotImplementedError(
            "Implement boxes or masks, save them under output_dir/annotations, and "
            "update annotation_path and annotation_confidence columns."
        )
