import torch

from csiro_biomass.models.backbone import BackboneConfig, create_backbone
from csiro_biomass.models.dual_stream import DualStreamBiomassModel, ModelConfig


def test_timm_backbone_respects_custom_image_size() -> None:
    adapter = create_backbone(
        BackboneConfig(
            name="vit_so400m_patch14_siglip_384",
            source="timm",
            pretrained=False,
            img_size=448,
        )
    )
    assert tuple(adapter.backbone.patch_embed.img_size) == (448, 448)


def test_dual_stream_model_passes_image_size_to_timm_backbone() -> None:
    model = DualStreamBiomassModel(
        ModelConfig(
            backbone_name="vit_so400m_patch14_siglip_384",
            backbone_source="timm",
            backbone_repo="facebookresearch/dinov3",
            pretrained=False,
            image_size=448,
            fusion_dim=1152,
            trunk_dim=1152,
            num_attention_heads=8,
            dropout=0.1,
            hf_endpoint=None,
        )
    )
    assert tuple(model.backbone.backbone.patch_embed.img_size) == (448, 448)

    dummy = torch.randn(1, 3, 448, 448)
    outputs = model(dummy, dummy)
    assert outputs["regression"]["Dry_Green_g"].shape == (1,)
