"""Losses and metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from csiro_biomass.data.constants import BASE_TARGET_COLUMNS, TARGET_COLUMNS, TARGET_TO_WEIGHT
from csiro_biomass.models.dual_stream import FIVE_HEAD, THREE_HEAD_CONSTRAINED


@dataclass(slots=True)
class LossOutput:
    total: torch.Tensor
    regression: torch.Tensor
    classification: torch.Tensor


class WeightedBiomassLoss(nn.Module):
    def __init__(self, cls_weight: float = 0.3, target_head_mode: str = FIVE_HEAD):
        super().__init__()
        self.regression_criterion = nn.SmoothL1Loss()
        self.classification_criterion = nn.CrossEntropyLoss()
        self.cls_weight = cls_weight
        self.weights = TARGET_TO_WEIGHT
        self.target_head_mode = target_head_mode

    def _classification_targets(self) -> list[str]:
        if self.target_head_mode == THREE_HEAD_CONSTRAINED:
            return list(BASE_TARGET_COLUMNS)
        if self.target_head_mode == FIVE_HEAD:
            return list(TARGET_COLUMNS)
        raise ValueError(f"Unsupported target_head_mode: {self.target_head_mode}")

    def forward(
        self,
        predictions: dict[str, dict[str, torch.Tensor]],
        targets: torch.Tensor,
        cls_labels: torch.Tensor,
    ) -> LossOutput:
        regression_loss = torch.zeros((), device=targets.device)
        classification_loss = torch.zeros((), device=targets.device)

        for idx, target_name in enumerate(TARGET_COLUMNS):
            weight = self.weights[target_name]
            regression_loss = regression_loss + weight * self.regression_criterion(
                predictions["regression"][target_name], targets[:, idx]
            )

        for idx, target_name in enumerate(self._classification_targets()):
            weight = self.weights[target_name]
            classification_loss = classification_loss + weight * self.classification_criterion(
                predictions["classification"][target_name], cls_labels[:, idx]
            )

        total_loss = regression_loss + self.cls_weight * classification_loss
        return LossOutput(total=total_loss, regression=regression_loss, classification=classification_loss)


class EpsilonInsensitiveLoss(nn.Module):
    def __init__(self, eps_point: float = 20, max_eps: float = 5, scale_ratio: float = 0.1):
        super().__init__()
        self.eps_point = eps_point
        self.max_eps = max_eps
        self.scale_ratio = scale_ratio

    def make_epsilon(self, targets: torch.Tensor) -> torch.Tensor:
        epsilon = torch.where(targets <= self.eps_point, torch.ones_like(targets), targets * self.scale_ratio)
        return torch.clamp(epsilon, max=self.max_eps)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        epsilon = self.make_epsilon(targets.detach())
        return torch.relu(torch.abs(predictions - targets) - epsilon).mean()


def weighted_r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    eps = 1e-8
    scores = []
    for idx, target_name in enumerate(TARGET_COLUMNS):
        target = y_true[:, idx]
        prediction = y_pred[:, idx]
        ss_res = torch.sum((target - prediction) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + eps))
        scores.append(float(r2) * TARGET_TO_WEIGHT[target_name])
    return float(sum(scores))
