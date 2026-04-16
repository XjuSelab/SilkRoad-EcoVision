"""Pseudo-label training loop."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from csiro_biomass.data.constants import TARGET_COLUMNS
from csiro_biomass.inference.predict import build_prediction_dataloader, load_model_from_checkpoint, predict_with_model
from csiro_biomass.training.supervised import run_training
from csiro_biomass.utils.config import ensure_dir, load_yaml_config


def generate_pseudo_labels(member_checkpoints: list[str], weights: list[float], config: dict) -> pd.DataFrame:
    import torch

    test_wide = pd.read_parquet(config["data"]["test_manifest"])
    dataloader = build_prediction_dataloader(test_wide, config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weighted_frames = []
    total_weight = 0.0
    for checkpoint_path, weight in zip(member_checkpoints, weights, strict=True):
        model = load_model_from_checkpoint(checkpoint_path, device)
        frame = predict_with_model(model, dataloader, device)
        frame[TARGET_COLUMNS] = frame[TARGET_COLUMNS] * weight
        weighted_frames.append(frame)
        total_weight += weight

    pseudo = weighted_frames[0].copy()
    for frame in weighted_frames[1:]:
        pseudo[TARGET_COLUMNS] = pseudo[TARGET_COLUMNS].add(frame[TARGET_COLUMNS], fill_value=0.0)
    pseudo[TARGET_COLUMNS] = pseudo[TARGET_COLUMNS] / total_weight
    pseudo = test_wide.drop(columns=TARGET_COLUMNS, errors="ignore").merge(
        pseudo,
        on="image_id",
        how="inner",
    )
    pseudo["cv_group"] = "pseudo"
    pseudo["is_pseudo"] = True
    return pseudo


def merge_train_and_pseudo(train_manifest: str | Path, pseudo_frame: pd.DataFrame, output_path: str | Path) -> str:
    train_wide = pd.read_parquet(train_manifest)
    combined = pd.concat([train_wide, pseudo_frame], axis=0, ignore_index=True, sort=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return str(output_path)


def run_pseudo_training(config: dict) -> None:
    pseudo_dir = ensure_dir(config["pseudo"]["output_dir"])
    current_teacher_checkpoints = list(config["pseudo"]["initial_teacher_checkpoints"])

    for round_index, round_config in enumerate(config["pseudo"]["rounds"], start=1):
        weights = round_config.get("teacher_weights") or [1.0] * len(current_teacher_checkpoints)
        pseudo_frame = generate_pseudo_labels(current_teacher_checkpoints, weights, config)
        pseudo_path = pseudo_dir / f"pseudo_round{round_index}.parquet"
        pseudo_frame.to_parquet(pseudo_path, index=False)

        combined_manifest = merge_train_and_pseudo(
            config["data"]["train_manifest"],
            pseudo_frame,
            pseudo_dir / f"train_plus_pseudo_round{round_index}.parquet",
        )

        produced_checkpoints = []
        for model_index in range(int(round_config["num_models"])):
            child_config = copy.deepcopy(config["pseudo"]["student_template"])
            child_config["data"]["train_manifest"] = combined_manifest
            child_config["model"]["backbone_name"] = round_config["student_backbone_name"]
            child_config["model"]["backbone_source"] = round_config.get("student_backbone_source", "timm")
            child_config["train"]["output_dir"] = str(
                pseudo_dir / f"round{round_index}" / f"student_{model_index + 1}"
            )
            child_config["train"]["seed"] = int(child_config["train"].get("seed", 42)) + model_index
            run_training(child_config)
            produced_checkpoints.append(str(Path(child_config["train"]["output_dir"]) / "best.pt"))

        current_teacher_checkpoints = produced_checkpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pseudo-label online training.")
    parser.add_argument("--config", required=True, help="Path to pseudo-label YAML config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pseudo_training(load_yaml_config(args.config))


if __name__ == "__main__":
    main()
