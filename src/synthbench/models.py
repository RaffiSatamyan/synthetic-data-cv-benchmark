from __future__ import annotations

from typing import Any


def build_model(config: dict[str, Any]):
    """Build a torchvision downstream model from a compact model config."""
    try:
        import torch.nn as nn
        from torchvision.models import ResNet18_Weights, resnet18
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn,
        )
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights,
            deeplabv3_resnet50,
        )
    except ImportError as error:
        raise RuntimeError('Install training dependencies: pip install -e ".[training]"') from error

    name = config["name"]
    num_classes = int(config["num_classes"])
    pretrained = bool(config.get("pretrained", True))

    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "fasterrcnn_resnet50_fpn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        return fasterrcnn_resnet50_fpn(weights=weights, num_classes=num_classes)

    if name == "deeplabv3_resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        return deeplabv3_resnet50(weights=weights, num_classes=num_classes)

    raise ValueError(f"Unknown model: {name}")
