"""Dual-stream biomass model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from csiro_biomass.data.constants import TARGET_COLUMNS
from csiro_biomass.models.backbone import BackboneConfig, create_backbone


def build_three_layer_head(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    hidden_dim = input_dim
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, output_dim),
    )


@dataclass(slots=True)
class ModelConfig:
    backbone_name: str
    backbone_source: str
    backbone_repo: str
    pretrained: bool
    fusion_dim: int
    trunk_dim: int
    num_attention_heads: int
    dropout: float
    hf_endpoint: str | None = None


class DualStreamBiomassModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.backbone = create_backbone(
            BackboneConfig(
                name=config.backbone_name,
                source=config.backbone_source,
                repo=config.backbone_repo,
                pretrained=config.pretrained,
                hf_endpoint=config.hf_endpoint,
            )
        )
        self.stream_projection = nn.Linear(self.backbone.feature_dim, config.fusion_dim)
        self.stream_norm = nn.LayerNorm(config.fusion_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.fusion_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.trunk = nn.Sequential(
            nn.Linear(config.fusion_dim * 2, config.trunk_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.trunk_dim, config.trunk_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.regression_heads = nn.ModuleDict(
            {target: build_three_layer_head(config.trunk_dim, 1, config.dropout) for target in TARGET_COLUMNS}
        )
        self.classification_heads = nn.ModuleDict(
            {target: build_three_layer_head(config.trunk_dim, 7, config.dropout) for target in TARGET_COLUMNS}
        )

    def freeze_backbone(self, frozen: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor) -> dict[str, torch.Tensor]:
        left_features = self.backbone(left_image)
        right_features = self.backbone(right_image)

        left_token = self.stream_projection(left_features)
        right_token = self.stream_projection(right_features)
        tokens = torch.stack([left_token, right_token], dim=1)
        attended, _ = self.cross_attention(tokens, tokens, tokens, need_weights=False)
        fused = self.stream_norm(tokens + attended)
        combined = fused.reshape(fused.shape[0], -1)
        shared = self.trunk(combined)

        regression = {
            target: self.regression_heads[target](shared).squeeze(-1) for target in TARGET_COLUMNS
        }
        classification = {target: self.classification_heads[target](shared) for target in TARGET_COLUMNS}
        return {"regression": regression, "classification": classification}
