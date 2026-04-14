"""Backbone factory with torch.hub and timm fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(slots=True)
class BackboneConfig:
    name: str
    source: str = "torchhub"
    repo: str = "facebookresearch/dinov3"
    pretrained: bool = True


class BackboneAdapter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = _infer_feature_dim(backbone)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(images)
            if isinstance(features, dict):
                for key in ("x_norm_clstoken", "cls_token", "pooler_output", "x_prenorm"):
                    if key in features:
                        tensor = features[key]
                        return tensor if tensor.ndim == 2 else tensor[:, 0]
                first_value = next(iter(features.values()))
                return first_value if first_value.ndim == 2 else first_value[:, 0]
            return features if features.ndim == 2 else features[:, 0]

        output = self.backbone(images)
        return output if output.ndim == 2 else output[:, 0]


def _infer_feature_dim(backbone: nn.Module) -> int:
    for attr in ("num_features", "embed_dim", "hidden_size"):
        value = getattr(backbone, attr, None)
        if isinstance(value, int):
            return value
    head = getattr(backbone, "head", None)
    if isinstance(head, nn.Linear):
        return head.in_features
    raise ValueError("Could not infer backbone feature dimension")


def create_backbone(config: BackboneConfig) -> BackboneAdapter:
    source = config.source.lower()
    if source == "torchhub":
        backbone = torch.hub.load(config.repo, config.name, pretrained=config.pretrained)
    elif source == "timm":
        import timm

        backbone = timm.create_model(config.name, pretrained=config.pretrained, num_classes=0)
    else:
        raise ValueError(f"Unsupported backbone source: {config.source}")

    return BackboneAdapter(backbone)
