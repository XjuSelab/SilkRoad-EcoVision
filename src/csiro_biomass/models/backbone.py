"""Backbone factory with torch.hub and timm fallbacks."""

from __future__ import annotations

import os
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
    hf_endpoint: str | None = None
    img_size: int | None = None


class BackboneAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, data_config: dict[str, Any] | None = None):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = _infer_feature_dim(backbone)
        self.data_config = data_config or {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "interpolation": "bicubic",
        }

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
    if config.hf_endpoint:
        # timm/huggingface_hub respects HF_ENDPOINT for model downloads.
        os.environ["HF_ENDPOINT"] = config.hf_endpoint

    source = config.source.lower()
    if source == "torchhub":
        backbone = torch.hub.load(config.repo, config.name, pretrained=config.pretrained)
        data_config = {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "interpolation": "bicubic",
        }
    elif source == "timm":
        import timm
        from timm.data import resolve_model_data_config

        create_kwargs: dict[str, Any] = {"pretrained": config.pretrained, "num_classes": 0}
        if config.img_size is not None:
            create_kwargs["img_size"] = int(config.img_size)
        backbone = timm.create_model(config.name, **create_kwargs)
        raw_data_config = resolve_model_data_config(backbone)
        data_config = {
            "mean": tuple(raw_data_config.get("mean", (0.485, 0.456, 0.406))),
            "std": tuple(raw_data_config.get("std", (0.229, 0.224, 0.225))),
            "interpolation": raw_data_config.get("interpolation", "bicubic"),
        }
    else:
        raise ValueError(f"Unsupported backbone source: {config.source}")

    return BackboneAdapter(backbone, data_config=data_config)
