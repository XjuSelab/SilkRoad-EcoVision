import sys
import types

import pytest
import torch
from torch import nn

from csiro_biomass.models.backbone import BackboneConfig, create_backbone


class _FakeProcessor:
    image_mean = [0.1, 0.2, 0.3]
    image_std = [0.4, 0.5, 0.6]
    resample = 3


class _FakeConfig:
    hidden_size = 32


class _FakeOutput:
    def __init__(self, batch_size: int, hidden_size: int, use_pooler: bool):
        self.pooler_output = torch.ones(batch_size, hidden_size) if use_pooler else None
        self.last_hidden_state = torch.ones(batch_size, 5, hidden_size)


class _FakeModel(nn.Module):
    def __init__(self, hidden_size: int, *, use_pooler: bool):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self._use_pooler = use_pooler

    def forward(self, images: torch.Tensor) -> _FakeOutput:
        return _FakeOutput(images.shape[0], self.config.hidden_size, self._use_pooler)


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("transformers")

    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeConfig()

    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeModel(32, use_pooler=True)

        @staticmethod
        def from_config(config):
            return _FakeModel(config.hidden_size, use_pooler=False)

    class AutoImageProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeProcessor()

    module.AutoConfig = AutoConfig
    module.AutoModel = AutoModel
    module.AutoImageProcessor = AutoImageProcessor
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_transformers_backbone_uses_processor_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch)

    adapter = create_backbone(
        BackboneConfig(
            name="facebook/dinov3-vitl16-pretrain-lvd1689m",
            source="transformers",
            path="artifacts/pretrained/modelscope/facebook/dinov3-vitl16-pretrain-lvd1689m",
            pretrained=True,
            local_files_only=True,
        )
    )

    assert adapter.feature_dim == 32
    assert adapter.data_config["mean"] == (0.1, 0.2, 0.3)
    assert adapter.data_config["std"] == (0.4, 0.5, 0.6)
    assert adapter.data_config["interpolation"] == "bicubic"

    outputs = adapter(torch.randn(2, 3, 16, 16))
    assert outputs.shape == (2, 32)


def test_transformers_backbone_falls_back_to_cls_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch)

    adapter = create_backbone(
        BackboneConfig(
            name="facebook/dinov3-vitl16-pretrain-lvd1689m",
            source="transformers",
            path="artifacts/pretrained/modelscope/facebook/dinov3-vitl16-pretrain-lvd1689m",
            pretrained=False,
            local_files_only=True,
        )
    )

    outputs = adapter(torch.randn(3, 3, 16, 16))
    assert outputs.shape == (3, 32)
