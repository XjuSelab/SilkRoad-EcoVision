"""Ensemble inference utilities and CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from csiro_biomass.data.constants import TARGET_COLUMNS
from csiro_biomass.data.dataset import CSIROBiomassDataset, DatasetConfig
from csiro_biomass.models.dual_stream import DualStreamBiomassModel, ModelConfig
from csiro_biomass.utils.config import ensure_dir, load_yaml_config
from csiro_biomass.utils.postprocess import apply_rule_based_postprocess, clamp_prediction_dict


def _build_model_from_config(config: dict[str, Any], device: torch.device) -> DualStreamBiomassModel:
    model = DualStreamBiomassModel(
        ModelConfig(
            backbone_name=config["model"]["backbone_name"],
            backbone_source=config["model"].get("backbone_source", "torchhub"),
            backbone_repo=config["model"].get("backbone_repo", "facebookresearch/dinov3"),
            pretrained=False,
            image_size=int(config["data"].get("image_size")) if config["data"].get("image_size") else None,
            fusion_dim=int(config["model"].get("fusion_dim", 1024)),
            trunk_dim=int(config["model"].get("trunk_dim", 1024)),
            num_attention_heads=int(config["model"].get("num_attention_heads", 8)),
            dropout=float(config["model"].get("dropout", 0.1)),
            hf_endpoint=config["model"].get("hf_endpoint"),
        )
    )
    model.to(device)
    model.eval()
    return model


def _average_state_dicts(checkpoint_payloads: list[dict]) -> dict[str, torch.Tensor]:
    state_dict = {}
    reference = checkpoint_payloads[0]["state_dict"]
    for key in reference:
        tensors = [payload["state_dict"][key].float() for payload in checkpoint_payloads]
        state_dict[key] = sum(tensors) / len(tensors)
    return state_dict


def load_model_from_checkpoint_member(member: dict[str, Any], device: torch.device) -> DualStreamBiomassModel:
    checkpoint_paths = member.get("checkpoints") or [member["checkpoint"]]
    payloads = [torch.load(path, map_location=device, weights_only=False) for path in checkpoint_paths]
    config = payloads[0]["config"]
    model = _build_model_from_config(config, device)
    state_dict = (
        _average_state_dicts(payloads)
        if len(payloads) > 1
        else payloads[0]["state_dict"]
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def _apply_tta(images: torch.Tensor, policy: str) -> torch.Tensor:
    if policy == "identity":
        return images
    if policy == "hflip":
        return torch.flip(images, dims=[3])
    if policy == "vflip":
        return torch.flip(images, dims=[2])
    if policy == "rot90":
        return torch.rot90(images, k=1, dims=[2, 3])
    raise ValueError(f"Unsupported TTA policy: {policy}")


def predict_with_model(
    model: DualStreamBiomassModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    tta_policies: list[str] | None = None,
) -> pd.DataFrame:
    policies = tta_policies or ["identity"]
    rows: list[dict] = []
    with torch.no_grad():
        for batch in dataloader:
            tta_predictions = []
            for policy in policies:
                outputs = model(
                    _apply_tta(batch["left_image"].to(device, non_blocking=True), policy),
                    _apply_tta(batch["right_image"].to(device, non_blocking=True), policy),
                )
                tta_predictions.append(
                    torch.stack([outputs["regression"][target] for target in TARGET_COLUMNS], dim=1).float()
                )
            stacked = torch.stack(tta_predictions, dim=0).mean(dim=0).float().cpu().numpy()
            for image_id, prediction in zip(batch["image_id"], stacked, strict=True):
                row = {"image_id": image_id}
                row.update({target: float(value) for target, value in zip(TARGET_COLUMNS, prediction, strict=True)})
                rows.append(row)
    return pd.DataFrame(rows)


def build_prediction_dataloader(frame: pd.DataFrame, config: dict) -> DataLoader:
    data_settings = config["data"]
    mean = tuple(data_settings.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(data_settings.get("std", (0.229, 0.224, 0.225)))
    interpolation = data_settings.get("interpolation", "bicubic")
    dataset = CSIROBiomassDataset(
        frame,
        DatasetConfig(
            image_root=data_settings["image_root"],
            image_size=int(data_settings["image_size"]),
            train=False,
            mean=mean,
            std=std,
            interpolation=interpolation,
        ),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["infer"].get("batch_size", 4)),
        shuffle=False,
        num_workers=int(config["infer"].get("num_workers", 4)),
        pin_memory=True,
    )


def reshape_submission(test_long: pd.DataFrame, prediction_wide: pd.DataFrame) -> pd.DataFrame:
    merged = test_long.copy()
    merged["image_id"] = merged["sample_id"].str.split("__").str[0]
    melted = prediction_wide.melt(
        id_vars=["image_id"],
        value_vars=TARGET_COLUMNS,
        var_name="target_name",
        value_name="target",
    )
    merged = merged.merge(melted, on=["image_id", "target_name"], how="left")
    return merged[["sample_id", "target"]]


def run_inference(config: dict) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(config["infer"]["output_dir"])

    test_wide = pd.read_parquet(config["data"]["test_manifest"])
    test_long = pd.read_parquet(config["data"]["test_long_manifest"])
    dataloader = build_prediction_dataloader(test_wide, config)

    weighted_frames: list[pd.DataFrame] = []
    total_weight = 0.0
    tta_policies = list(config["infer"].get("tta_policies", ["identity"]))
    for member in config["infer"]["members"]:
        model = load_model_from_checkpoint_member(member, device=device)
        frame = predict_with_model(model, dataloader, device=device, tta_policies=tta_policies)
        weight = float(member["weight"])
        frame[TARGET_COLUMNS] = frame[TARGET_COLUMNS] * weight
        weighted_frames.append(frame)
        total_weight += weight

    ensemble = weighted_frames[0].copy()
    for frame in weighted_frames[1:]:
        ensemble[TARGET_COLUMNS] = ensemble[TARGET_COLUMNS].add(frame[TARGET_COLUMNS], fill_value=0.0)
    ensemble[TARGET_COLUMNS] = ensemble[TARGET_COLUMNS] / total_weight

    if config["infer"].get("apply_postprocess", True):
        processed_rows = []
        for row in ensemble.to_dict("records"):
            processed = clamp_prediction_dict(apply_rule_based_postprocess(row))
            processed_rows.append({"image_id": row["image_id"], **processed})
        ensemble = pd.DataFrame(processed_rows)

    ensemble.to_parquet(output_dir / "predictions_wide.parquet", index=False)
    submission = reshape_submission(test_long, ensemble)
    submission.to_csv(output_dir / "submission.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CSIRO biomass ensemble inference.")
    parser.add_argument("--config", required=True, help="Path to inference YAML config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_inference(load_yaml_config(args.config))


if __name__ == "__main__":
    main()
