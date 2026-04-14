"""Training loop utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from csiro_biomass.data.constants import TARGET_COLUMNS
from csiro_biomass.training.losses import WeightedBiomassLoss, weighted_r2_score


@dataclass(slots=True)
class EpochResult:
    loss: float
    regression_loss: float
    classification_loss: float
    weighted_r2: float
    predictions: pd.DataFrame


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    name = config.get("name", "adamw").lower()
    lr = float(config.get("lr", 2e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer: {name}")
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict, steps_per_epoch: int):
    name = config.get("name", "cosine").lower()
    epochs = int(config["epochs"])
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(config.get("warmup_steps", 0))

    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name}")

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def _collect_batch_predictions(outputs: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    return torch.stack([outputs["regression"][target] for target in TARGET_COLUMNS], dim=1)


def _run_epoch(
    *,
    model: nn.Module,
    dataloader,
    criterion: WeightedBiomassLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    grad_accum_steps: int = 1,
) -> EpochResult:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_regression = 0.0
    total_classification = 0.0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_image_ids: list[str] = []

    progress = tqdm(dataloader, disable=False)
    optimizer.zero_grad(set_to_none=True) if is_training else None
    for batch_index, batch in enumerate(progress):
        left_image = batch["left_image"].to(device, non_blocking=True)
        right_image = batch["right_image"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        cls_labels = batch["cls_labels"].to(device, non_blocking=True)
        image_ids = batch["image_id"]

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled and device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            outputs = model(left_image, right_image)
            loss_output = criterion(outputs, targets, cls_labels)

        if is_training:
            scaled_loss = loss_output.total / grad_accum_steps
            scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % grad_accum_steps == 0 or (batch_index + 1) == len(dataloader)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        predictions = _collect_batch_predictions(outputs).detach().float().cpu()
        all_predictions.append(predictions)
        all_targets.append(targets.detach().cpu())
        all_image_ids.extend(image_ids)

        batch_size = targets.shape[0]
        total_loss += float(loss_output.total.detach().cpu()) * batch_size
        total_regression += float(loss_output.regression.detach().cpu()) * batch_size
        total_classification += float(loss_output.classification.detach().cpu()) * batch_size

    prediction_tensor = torch.cat(all_predictions, dim=0)
    target_tensor = torch.cat(all_targets, dim=0)
    num_samples = len(all_image_ids)
    frame = pd.DataFrame(prediction_tensor.numpy(), columns=TARGET_COLUMNS)
    frame.insert(0, "image_id", all_image_ids)

    return EpochResult(
        loss=total_loss / max(1, num_samples),
        regression_loss=total_regression / max(1, num_samples),
        classification_loss=total_classification / max(1, num_samples),
        weighted_r2=weighted_r2_score(target_tensor, prediction_tensor),
        predictions=frame,
    )


def train_one_epoch(**kwargs) -> EpochResult:
    return _run_epoch(**kwargs)


def evaluate_one_epoch(**kwargs) -> EpochResult:
    with torch.no_grad():
        return _run_epoch(optimizer=None, scheduler=None, **kwargs)


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
