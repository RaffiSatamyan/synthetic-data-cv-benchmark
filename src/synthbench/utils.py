from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import tempfile
from pathlib import Path
from typing import Any

import yaml


def set_seed(seed: int) -> None:
    """Seed Python and, when installed, NumPy and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any, length: int = 8) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(destination)


def write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_yaml(path: str | Path, value: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(value, sort_keys=False))


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        info["torch"] = None
    return info
