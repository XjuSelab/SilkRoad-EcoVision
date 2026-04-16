import torch

from csiro_biomass.data.constants import BASE_TARGET_COLUMNS, TARGET_COLUMNS
from csiro_biomass.models.backbone import BackboneConfig, create_backbone
from csiro_biomass.models.dual_stream import (
    THREE_HEAD_CONSTRAINED,
    DualStreamBiomassModel,
    ModelConfig,
)


def test_timm_backbone_respects_custom_image_size() -> None:
    adapter = create_backbone(
        BackboneConfig(
            name="vit_tiny_patch16_224",
            source="timm",
            pretrained=False,
            img_size=448,
        )
    )
    assert tuple(adapter.backbone.patch_embed.img_size) == (448, 448)


def test_dual_stream_model_passes_image_size_to_timm_backbone() -> None:
    model = DualStreamBiomassModel(
        ModelConfig(
            backbone_name="vit_tiny_patch16_224",
            backbone_source="timm",
            backbone_repo="facebookresearch/dinov3",
            pretrained=False,
            backbone_weights=None,
            backbone_check_hash=False,
            backbone_path=None,
            backbone_local_files_only=True,
            image_size=448,
            fusion_dim=192,
            trunk_dim=192,
            num_attention_heads=8,
            dropout=0.1,
            target_head_mode="five_head",
            hf_endpoint=None,
        )
    )
    assert tuple(model.backbone.backbone.patch_embed.img_size) == (448, 448)

    dummy = torch.randn(1, 3, 448, 448)
    outputs = model(dummy, dummy)
    assert outputs["regression"]["Dry_Green_g"].shape == (1,)


def test_dual_stream_model_three_head_mode_derives_total_targets() -> None:
    model = DualStreamBiomassModel(
        ModelConfig(
            backbone_name="vit_tiny_patch16_224",
            backbone_source="timm",
            backbone_repo="facebookresearch/dinov3",
            pretrained=False,
            backbone_weights=None,
            backbone_check_hash=False,
            backbone_path=None,
            backbone_local_files_only=True,
            image_size=448,
            fusion_dim=192,
            trunk_dim=192,
            num_attention_heads=8,
            dropout=0.1,
            target_head_mode=THREE_HEAD_CONSTRAINED,
            hf_endpoint=None,
            use_metadata=True,
            metadata_feature_dim=6,
            metadata_hidden_dim=32,
        )
    )

    dummy = torch.randn(2, 3, 448, 448)
    metadata = torch.randn(2, 6)
    outputs = model(dummy, dummy, metadata_features=metadata)
    assert set(outputs["regression"]) == set(TARGET_COLUMNS)
    assert set(outputs["classification"]) == set(BASE_TARGET_COLUMNS)
    assert torch.allclose(
        outputs["regression"]["GDM_g"],
        outputs["regression"]["Dry_Green_g"] + outputs["regression"]["Dry_Clover_g"],
    )
    assert torch.allclose(
        outputs["regression"]["Dry_Total_g"],
        outputs["regression"]["Dry_Green_g"]
        + outputs["regression"]["Dry_Dead_g"]
        + outputs["regression"]["Dry_Clover_g"],
    )


def test_dual_stream_model_metadata_defaults_to_zero_tensor() -> None:
    model = DualStreamBiomassModel(
        ModelConfig(
            backbone_name="vit_tiny_patch16_224",
            backbone_source="timm",
            backbone_repo="facebookresearch/dinov3",
            pretrained=False,
            backbone_weights=None,
            backbone_check_hash=False,
            backbone_path=None,
            backbone_local_files_only=True,
            image_size=224,
            fusion_dim=192,
            trunk_dim=192,
            num_attention_heads=8,
            dropout=0.1,
            target_head_mode=THREE_HEAD_CONSTRAINED,
            hf_endpoint=None,
            use_metadata=True,
            metadata_feature_dim=4,
            metadata_hidden_dim=16,
        )
    )

    dummy = torch.randn(1, 3, 224, 224)
    outputs = model(dummy, dummy)
    assert outputs["regression"]["Dry_Green_g"].shape == (1,)
