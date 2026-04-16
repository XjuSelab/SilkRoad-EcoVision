"""Dual-stream biomass model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from csiro_biomass.data.constants import BASE_TARGET_COLUMNS, TARGET_COLUMNS
from csiro_biomass.models.backbone import BackboneConfig, create_backbone

FIVE_HEAD = "five_head"
THREE_HEAD_CONSTRAINED = "three_head_constrained"


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
    backbone_weights: str | None
    backbone_check_hash: bool
    backbone_path: str | None
    backbone_local_files_only: bool
    image_size: int | None
    fusion_dim: int
    trunk_dim: int
    num_attention_heads: int
    dropout: float
    target_head_mode: str = FIVE_HEAD
    hf_endpoint: str | None = None


class DualStreamBiomassModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.target_head_mode = config.target_head_mode
        self.backbone = create_backbone(
            BackboneConfig(
                name=config.backbone_name,
                source=config.backbone_source,
                repo=config.backbone_repo,
                pretrained=config.pretrained,
                weights=config.backbone_weights,
                check_hash=config.backbone_check_hash,
                path=config.backbone_path,
                local_files_only=config.backbone_local_files_only,
                hf_endpoint=config.hf_endpoint,
                img_size=config.image_size,
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
        regression_targets = self._head_targets()
        self.regression_heads = nn.ModuleDict(
            {target: build_three_layer_head(config.trunk_dim, 1, config.dropout) for target in regression_targets}
        )
        self.classification_heads = nn.ModuleDict(
            {target: build_three_layer_head(config.trunk_dim, 7, config.dropout) for target in regression_targets}
        )

    def _head_targets(self) -> list[str]:
        if self.target_head_mode == THREE_HEAD_CONSTRAINED:
            return list(BASE_TARGET_COLUMNS)
        if self.target_head_mode == FIVE_HEAD:
            return list(TARGET_COLUMNS)
        raise ValueError(f"Unsupported target_head_mode: {self.target_head_mode}")

    def _build_regression_outputs(self, shared: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.target_head_mode == FIVE_HEAD:
            return {
                target: self.regression_heads[target](shared).squeeze(-1) for target in TARGET_COLUMNS
            }

        green = self.regression_heads["Dry_Green_g"](shared).squeeze(-1)
        dead = self.regression_heads["Dry_Dead_g"](shared).squeeze(-1)
        clover = self.regression_heads["Dry_Clover_g"](shared).squeeze(-1)
        gdm = green + clover
        total = green + dead + clover
        return {
            "Dry_Green_g": green,
            "Dry_Dead_g": dead,
            "Dry_Clover_g": clover,
            "GDM_g": gdm,
            "Dry_Total_g": total,
        }

    def _build_classification_outputs(self, shared: torch.Tensor) -> dict[str, torch.Tensor]:
        head_targets = self._head_targets()
        return {target: self.classification_heads[target](shared) for target in head_targets}

    def freeze_backbone(
        self,
        frozen: bool,
        *,
        unfreeze_last_n_blocks: int = 0,
        unfreeze_norm: bool = True,
        unfreeze_pos_embed: bool = True,
    ) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen
        if frozen or unfreeze_last_n_blocks <= 0:
            return

        inner_backbone = self.backbone.backbone
        blocks = getattr(inner_backbone, "blocks", None)
        if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
            for block in blocks[-unfreeze_last_n_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

        if unfreeze_norm:
            for name in ("norm", "fc_norm", "head_norm"):
                module = getattr(inner_backbone, name, None)
                if isinstance(module, nn.Module):
                    for parameter in module.parameters():
                        parameter.requires_grad = True

        if unfreeze_pos_embed:
            for attribute in ("pos_embed", "cls_token", "dist_token"):
                tensor = getattr(inner_backbone, attribute, None)
                if isinstance(tensor, nn.Parameter):
                    tensor.requires_grad = True

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

        regression = self._build_regression_outputs(shared)
        classification = self._build_classification_outputs(shared)
        return {"regression": regression, "classification": classification}
