"""Supervised training entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from csiro_biomass.data.dataset import CSIROBiomassDataset, DatasetConfig
from csiro_biomass.models.dual_stream import DualStreamBiomassModel, ModelConfig
from csiro_biomass.training.engine import (
    build_optimizer,
    build_scheduler,
    evaluate_one_epoch,
    save_checkpoint,
    seed_everything,
    select_device,
    train_one_epoch,
)
from csiro_biomass.training.losses import WeightedBiomassLoss
from csiro_biomass.utils.config import ensure_dir, load_yaml_config
from csiro_biomass.utils.distributed import destroy_distributed, init_distributed


def _build_dataloader(frame: pd.DataFrame, config: dict, train: bool, distributed: bool):
    dataset = CSIROBiomassDataset(
        frame,
        DatasetConfig(
            image_root=config["data"]["image_root"],
            image_size=int(config["data"]["image_size"]),
            train=train,
            black_padding_probability=float(config["data"].get("black_padding_probability", 0.2)),
        ),
    )
    sampler = None
    if distributed and train:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, shuffle=True)
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=int(config["train"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=train,
    )


def run_training(config: dict) -> None:
    distributed = init_distributed()
    device = select_device(distributed.local_rank)
    seed_everything(int(config["train"].get("seed", 42)) + distributed.rank)

    train_wide = pd.read_parquet(config["data"]["train_manifest"])
    fold_df = pd.read_parquet(config["data"]["fold_manifest"])
    frame = train_wide.merge(fold_df, on="image_id", how="left")

    valid_fold = int(config["train"]["valid_fold"])
    train_frame = frame[frame["fold"] != valid_fold].reset_index(drop=True)
    valid_frame = frame[frame["fold"] == valid_fold].reset_index(drop=True)

    train_loader = _build_dataloader(train_frame, config, train=True, distributed=distributed.distributed)
    valid_loader = _build_dataloader(valid_frame, config, train=False, distributed=False)

    model = DualStreamBiomassModel(
        ModelConfig(
            backbone_name=config["model"]["backbone_name"],
            backbone_source=config["model"].get("backbone_source", "torchhub"),
            backbone_repo=config["model"].get("backbone_repo", "facebookresearch/dinov3"),
            pretrained=bool(config["model"].get("pretrained", True)),
            fusion_dim=int(config["model"].get("fusion_dim", 1024)),
            trunk_dim=int(config["model"].get("trunk_dim", 1024)),
            num_attention_heads=int(config["model"].get("num_attention_heads", 8)),
            dropout=float(config["model"].get("dropout", 0.1)),
        )
    ).to(device)

    if distributed.distributed:
        torch.cuda.set_device(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[distributed.local_rank], output_device=distributed.local_rank
        )

    criterion = WeightedBiomassLoss(cls_weight=float(config["train"].get("cls_weight", 0.3)))
    scaler = GradScaler(enabled=device.type == "cuda")
    output_dir = ensure_dir(config["train"]["output_dir"])
    history: list[dict] = []
    best_score = float("-inf")

    model_module = model.module if hasattr(model, "module") else model
    global_step = 0
    for stage_index, stage in enumerate(config["train"]["stages"], start=1):
        freeze_backbone = bool(stage.get("freeze_backbone", False))
        model_module.freeze_backbone(freeze_backbone)

        optimizer = build_optimizer(model, stage["optimizer"])
        scheduler = build_scheduler(optimizer, stage["scheduler"], steps_per_epoch=len(train_loader))

        for epoch in range(int(stage["scheduler"]["epochs"])):
            if distributed.distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_result = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                amp_enabled=bool(config["train"].get("amp", True)),
            )
            valid_result = evaluate_one_epoch(
                model=model,
                dataloader=valid_loader,
                criterion=criterion,
                device=device,
                scaler=scaler,
                amp_enabled=bool(config["train"].get("amp", True)),
            )

            global_step += len(train_loader)
            epoch_record = {
                "stage": stage_index,
                "epoch": epoch + 1,
                "train_loss": train_result.loss,
                "train_r2": train_result.weighted_r2,
                "valid_loss": valid_result.loss,
                "valid_r2": valid_result.weighted_r2,
                "global_step": global_step,
            }
            history.append(epoch_record)

            if distributed.is_main_process and valid_result.weighted_r2 > best_score:
                best_score = valid_result.weighted_r2
                save_checkpoint(
                    output_dir / "best.pt",
                    {
                        "config": config,
                        "state_dict": model_module.state_dict(),
                        "metrics": epoch_record,
                    },
                )
                valid_result.predictions.to_parquet(output_dir / "valid_predictions.parquet", index=False)

    if distributed.is_main_process:
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    destroy_distributed(distributed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the supervised CSIRO biomass baseline.")
    parser.add_argument("--config", required=True, help="Path to supervised YAML config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(load_yaml_config(args.config))


if __name__ == "__main__":
    main()
