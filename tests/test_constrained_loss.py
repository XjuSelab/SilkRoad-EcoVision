import torch

from csiro_biomass.data.constants import BASE_TARGET_COLUMNS, TARGET_COLUMNS
from csiro_biomass.models.dual_stream import THREE_HEAD_CONSTRAINED
from csiro_biomass.training.losses import WeightedBiomassLoss


def test_weighted_biomass_loss_three_head_mode_uses_five_regression_targets() -> None:
    batch_size = 2
    predictions = {
        "regression": {
            target: torch.full((batch_size,), 1.0, requires_grad=True) for target in TARGET_COLUMNS
        },
        "classification": {
            target: torch.randn(batch_size, 7, requires_grad=True) for target in BASE_TARGET_COLUMNS
        },
    }
    targets = torch.ones(batch_size, len(TARGET_COLUMNS))
    cls_labels = torch.zeros(batch_size, len(TARGET_COLUMNS), dtype=torch.long)

    criterion = WeightedBiomassLoss(cls_weight=0.3, target_head_mode=THREE_HEAD_CONSTRAINED)
    loss_output = criterion(predictions, targets, cls_labels)

    assert loss_output.total.ndim == 0
    assert loss_output.regression.ndim == 0
    assert loss_output.classification.ndim == 0
    loss_output.total.backward()

    for target in TARGET_COLUMNS:
        assert predictions["regression"][target].grad is not None
    for target in BASE_TARGET_COLUMNS:
        assert predictions["classification"][target].grad is not None
