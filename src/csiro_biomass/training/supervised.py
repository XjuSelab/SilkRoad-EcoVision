"""Supervised training entrypoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from csiro_biomass.data.dataset import CSIROBiomassDataset, DatasetConfig, validate_frame_image_paths
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
from csiro_biomass.training.metrics import (
    compute_weighted_r2_from_frame,
    save_metrics_csv,
    summarize_validation,
)
from csiro_biomass.training.oof import aggregate_experiment_root
from csiro_biomass.training.losses import WeightedBiomassLoss
from csiro_biomass.utils.config import ensure_dir, load_yaml_config
from csiro_biomass.utils.distributed import destroy_distributed, init_distributed


def _resolve_data_config(
    config: dict[str, Any],
    model: DualStreamBiomassModel,
) -> dict[str, Any]:
    data_config = config["data"]
    mean = tuple(data_config.get("mean", model.backbone.data_config["mean"]))
    std = tuple(data_config.get("std", model.backbone.data_config["std"]))
    interpolation = data_config.get("interpolation", model.backbone.data_config["interpolation"])
    return {"mean": mean, "std": std, "interpolation": interpolation}


def _build_dataloader(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    train: bool,
    distributed: bool,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    interpolation: str,
):
    dataset = CSIROBiomassDataset(
        frame,
        DatasetConfig(
            image_root=config["data"]["image_root"],
            image_size=int(config["data"]["image_size"]),
            train=train,
            black_padding_probability=float(config["data"].get("black_padding_probability", 0.2)),
            mean=mean,
            std=std,
            interpolation=interpolation,
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


def _build_model(config: dict[str, Any], device: torch.device) -> DualStreamBiomassModel:
    return DualStreamBiomassModel(
        ModelConfig(
            backbone_name=config["model"]["backbone_name"],
            backbone_source=config["model"].get("backbone_source", "torchhub"),
            backbone_repo=config["model"].get("backbone_repo", "facebookresearch/dinov3"),
            pretrained=bool(config["model"].get("pretrained", True)),
            backbone_weights=config["model"].get("backbone_weights"),
            backbone_check_hash=bool(config["model"].get("backbone_check_hash", False)),
            backbone_path=config["model"].get("backbone_path"),
            backbone_local_files_only=bool(config["model"].get("backbone_local_files_only", True)),
            image_size=int(config["data"].get("image_size")) if config["data"].get("image_size") else None,
            fusion_dim=int(config["model"].get("fusion_dim", 1024)),
            trunk_dim=int(config["model"].get("trunk_dim", 1024)),
            num_attention_heads=int(config["model"].get("num_attention_heads", 8)),
            dropout=float(config["model"].get("dropout", 0.1)),
            target_head_mode=config["model"].get("target_head_mode", "five_head"),
            hf_endpoint=config["model"].get("hf_endpoint"),
        )
    ).to(device)


def _single_run_output_dir(config: dict[str, Any]) -> Path:
    return ensure_dir(config["train"]["output_dir"])


def _write_summary(
    *,
    output_dir: Path,
    config: dict[str, Any],
    validation_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    history: list[dict[str, Any]],
) -> None:
    best_valid_r2 = max(record["valid_r2"] for record in history)
    summary = {
        "experiment_name": config["train"].get("experiment_name", output_dir.name),
        "backbone_name": config["model"]["backbone_name"],
        "backbone_source": config["model"].get("backbone_source", "torchhub"),
        "image_size": int(config["data"]["image_size"]),
        "fold": int(config["train"]["valid_fold"]),
        "seed": int(config["train"].get("seed", 42)),
        "best_valid_r2": float(best_valid_r2),
        "oof_weighted_r2": float(compute_weighted_r2_from_frame(validation_frame)),
        "pred_std_min": float(metrics_frame["pred_std"].min()),
        "pred_std_mean": float(metrics_frame["pred_std"].mean()),
        "targets": metrics_frame.to_dict("records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_training_job(config: dict[str, Any]) -> Path:
    distributed = init_distributed()
    device = select_device(distributed.local_rank)
    seed_everything(int(config["train"].get("seed", 42)) + distributed.rank)
    output_dir = _single_run_output_dir(config)

    train_wide = pd.read_parquet(config["data"]["train_manifest"])
    fold_df = pd.read_parquet(config["data"]["fold_manifest"])
    frame = train_wide.merge(fold_df, on="image_id", how="left")

    valid_fold = int(config["train"]["valid_fold"])
    train_frame = frame[frame["fold"] != valid_fold].reset_index(drop=True)
    valid_frame = frame[frame["fold"] == valid_fold].reset_index(drop=True)
    validate_frame_image_paths(train_frame, config["data"]["image_root"], frame_name="train_frame")
    validate_frame_image_paths(valid_frame, config["data"]["image_root"], frame_name="valid_frame")

    model = _build_model(config, device)
    data_defaults = _resolve_data_config(config, model)

    train_loader = _build_dataloader(
        train_frame,
        config,
        train=True,
        distributed=distributed.distributed,
        **data_defaults,
    )
    valid_loader = _build_dataloader(
        valid_frame,
        config,
        train=False,
        distributed=False,
        **data_defaults,
    )

    if distributed.distributed:
        torch.cuda.set_device(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
        )

    criterion = WeightedBiomassLoss(
        cls_weight=float(config["train"].get("cls_weight", 0.3)),
        target_head_mode=config["model"].get("target_head_mode", "five_head"),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_score = float("-inf")

    model_module = model.module if hasattr(model, "module") else model
    global_step = 0
    grad_accum_steps = max(1, int(config["train"].get("grad_accum_steps", 1)))
    best_validation_frame: pd.DataFrame | None = None
    best_metrics_frame: pd.DataFrame | None = None
    for stage_index, stage in enumerate(config["train"]["stages"], start=1):
        freeze_backbone = bool(stage.get("freeze_backbone", False))
        model_module.freeze_backbone(
            freeze_backbone,
            unfreeze_last_n_blocks=int(stage.get("unfreeze_last_n_blocks", 0)),
            unfreeze_norm=bool(stage.get("unfreeze_norm", True)),
            unfreeze_pos_embed=bool(stage.get("unfreeze_pos_embed", True)),
        )

        optimizer = build_optimizer(model, stage["optimizer"])
        scheduler = build_scheduler(
            optimizer,
            stage["scheduler"],
            steps_per_epoch=max(1, len(train_loader) // grad_accum_steps),
        )

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
                grad_accum_steps=grad_accum_steps,
            )
            valid_result = evaluate_one_epoch(
                model=model,
                dataloader=valid_loader,
                criterion=criterion,
                device=device,
                scaler=scaler,
                amp_enabled=bool(config["train"].get("amp", True)),
            )
            validation_frame, metrics_frame = summarize_validation(
                valid_result.predictions,
                valid_frame,
                fold=valid_fold,
                seed=int(config["train"].get("seed", 42)),
            )

            global_step += len(train_loader)
            epoch_record = {
                "stage": stage_index,
                "epoch": epoch + 1,
                "train_loss": train_result.loss,
                "train_r2": train_result.weighted_r2,
                "valid_loss": valid_result.loss,
                "valid_r2": valid_result.weighted_r2,
                "oof_weighted_r2": compute_weighted_r2_from_frame(validation_frame),
                "global_step": global_step,
            }
            history.append(epoch_record)

            if distributed.is_main_process and valid_result.weighted_r2 > best_score:
                best_score = valid_result.weighted_r2
                best_validation_frame = validation_frame
                best_metrics_frame = metrics_frame
                save_checkpoint(
                    output_dir / "best.pt",
                    {
                        "config": config,
                        "state_dict": model_module.state_dict(),
                        "metrics": epoch_record,
                    },
                )
                validation_frame.to_parquet(output_dir / "valid_predictions.parquet", index=False)
                save_metrics_csv(output_dir / "valid_metrics.csv", metrics_frame)

    if distributed.is_main_process:
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if best_validation_frame is not None and best_metrics_frame is not None:
            _write_summary(
                output_dir=output_dir,
                config=config,
                validation_frame=best_validation_frame,
                metrics_frame=best_metrics_frame,
                history=history,
            )

    destroy_distributed(distributed)
    return output_dir


def _expand_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    folds = config["train"].get("folds")
    seeds = config["train"].get("seeds")
    if folds is None and seeds is None:
        return [config]

    if int(torch.cuda.device_count()) > 1 and any(isinstance(v, list) for v in (folds, seeds)):
        # This sweep mode is intended for independent single-process jobs.
        pass

    folds = folds or [int(config["train"].get("valid_fold", 0))]
    seeds = seeds or [int(config["train"].get("seed", 42))]
    base_output_dir = Path(config["train"]["output_dir"])
    experiment_name = config["train"].get("experiment_name", base_output_dir.name)

    runs = []
    for fold in folds:
        for seed in seeds:
            child = copy.deepcopy(config)
            child["train"]["valid_fold"] = int(fold)
            child["train"]["seed"] = int(seed)
            child["train"]["experiment_name"] = experiment_name
            child["train"]["output_dir"] = str(base_output_dir / f"fold{fold}_seed{seed}")
            runs.append(child)
    return runs


def run_training(config: dict[str, Any]) -> None:
    runs = _expand_runs(config)
    run_dirs = [run_training_job(run_config) for run_config in runs]

    if len(run_dirs) > 1:
        aggregate_experiment_root(
            experiment_root=Path(config["train"]["output_dir"]),
            train_manifest=Path(config["data"]["train_manifest"]),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the supervised CSIRO biomass baseline.")
    parser.add_argument("--config", required=True, help="Path to supervised YAML config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(load_yaml_config(args.config))


if __name__ == "__main__":
    main()
