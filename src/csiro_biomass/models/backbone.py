"""Backbone factory with torch.hub, timm, and transformers backbones."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torch import nn


@dataclass(slots=True)
class BackboneConfig:
    name: str
    source: str = "torchhub"
    repo: str = "facebookresearch/dinov3"
    pretrained: bool = True
    weights: str | None = None
    check_hash: bool = False
    path: str | None = None
    local_files_only: bool = True
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
            return _extract_feature_tensor(features)

        output = self.backbone(images)
        return _extract_feature_tensor(output)


def _extract_feature_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("x_norm_clstoken", "cls_token", "pooler_output", "x_prenorm", "last_hidden_state"):
            if key in output:
                tensor = output[key]
                return tensor if tensor.ndim == 2 else tensor[:, 0]
        first_value = next(iter(output.values()))
        return first_value if first_value.ndim == 2 else first_value[:, 0]

    for attr in ("pooler_output", "last_hidden_state"):
        tensor = getattr(output, attr, None)
        if tensor is not None:
            return tensor if tensor.ndim == 2 else tensor[:, 0]

    if isinstance(output, (tuple, list)) and output:
        first_value = output[0]
        return first_value if first_value.ndim == 2 else first_value[:, 0]

    if isinstance(output, torch.Tensor):
        return output if output.ndim == 2 else output[:, 0]

    raise TypeError(f"Unsupported backbone output type: {type(output)!r}")


def _infer_feature_dim(backbone: nn.Module) -> int:
    for attr in ("num_features", "embed_dim", "hidden_size"):
        value = getattr(backbone, attr, None)
        if isinstance(value, int):
            return value
    model_config = getattr(backbone, "config", None)
    if model_config is not None:
        for attr in ("hidden_size", "projection_dim"):
            value = getattr(model_config, attr, None)
            if isinstance(value, int):
                return value
    head = getattr(backbone, "head", None)
    if isinstance(head, nn.Linear):
        return head.in_features
    raise ValueError("Could not infer backbone feature dimension")


def _resolve_transformers_interpolation(resample: Any) -> str:
    if isinstance(resample, str):
        return resample.lower()

    value = getattr(resample, "value", resample)
    mapping = {
        Image.Resampling.NEAREST.value: "nearest",
        Image.Resampling.BILINEAR.value: "bilinear",
        Image.Resampling.BICUBIC.value: "bicubic",
        Image.Resampling.LANCZOS.value: "lanczos",
        Image.Resampling.BOX.value: "area",
        Image.Resampling.HAMMING.value: "bilinear",
    }
    return mapping.get(value, "bicubic")


def _resolve_transformers_data_config(processor: Any) -> dict[str, Any]:
    mean = tuple(getattr(processor, "image_mean", (0.485, 0.456, 0.406)))
    std = tuple(getattr(processor, "image_std", (0.229, 0.224, 0.225)))
    interpolation = _resolve_transformers_interpolation(getattr(processor, "resample", None))
    return {"mean": mean, "std": std, "interpolation": interpolation}


def create_backbone(config: BackboneConfig) -> BackboneAdapter:
    if config.hf_endpoint:
        # timm/huggingface_hub respects HF_ENDPOINT for model downloads.
        os.environ["HF_ENDPOINT"] = config.hf_endpoint

    source = config.source.lower()
    if source == "torchhub":
        hub_kwargs: dict[str, Any] = {"pretrained": config.pretrained}
        if config.weights:
            hub_kwargs["weights"] = config.weights
            hub_kwargs["check_hash"] = config.check_hash
        backbone = torch.hub.load(config.repo, config.name, **hub_kwargs)
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
    elif source == "transformers":
        from transformers import AutoConfig, AutoImageProcessor, AutoModel

        model_path = config.path or config.name
        hf_kwargs: dict[str, Any] = {"local_files_only": config.local_files_only}
        if config.pretrained:
            backbone = AutoModel.from_pretrained(model_path, **hf_kwargs)
        else:
            backbone_config = AutoConfig.from_pretrained(model_path, **hf_kwargs)
            backbone = AutoModel.from_config(backbone_config)
        processor = AutoImageProcessor.from_pretrained(model_path, **hf_kwargs)
        data_config = _resolve_transformers_data_config(processor)
    else:
        raise ValueError(f"Unsupported backbone source: {config.source}")

    return BackboneAdapter(backbone, data_config=data_config)
