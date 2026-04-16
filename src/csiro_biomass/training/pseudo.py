"""Pseudo-label training loop."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import pandas as pd

from csiro_biomass.data.constants import TARGET_COLUMNS
from csiro_biomass.data.dataset import validate_frame_image_paths
from csiro_biomass.inference.predict import build_prediction_dataloader, load_model_from_checkpoint, predict_with_model
from csiro_biomass.training.supervised import run_training
from csiro_biomass.utils.config import ensure_dir, load_yaml_config


def _log_pseudo_environment(config: dict) -> None:
    import torch

    print(
        "[pseudo-env] "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()} "
        f"output_dir={config['pseudo']['output_dir']} "
        f"teachers={len(config['pseudo']['initial_teacher_checkpoints'])} "
        f"rounds={len(config['pseudo']['rounds'])}"
    )
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(current_device)
        print(
            "[pseudo-env] "
            f"current_device={current_device} "
            f"device_name={torch.cuda.get_device_name(current_device)} "
            f"total_memory_gb={properties.total_memory / (1024 ** 3):.2f}"
        )


def _validate_pseudo_frame(pseudo_frame: pd.DataFrame, image_root: str | Path, *, frame_name: str) -> None:
    if pseudo_frame["image_id"].duplicated().any():
        duplicated_ids = pseudo_frame.loc[pseudo_frame["image_id"].duplicated(), "image_id"].head(5).tolist()
        raise ValueError(f"{frame_name} contains duplicate image_id values: {duplicated_ids}")

    missing_targets = [target for target in TARGET_COLUMNS if target not in pseudo_frame.columns]
    if missing_targets:
        raise ValueError(f"{frame_name} is missing target columns: {missing_targets}")

    validate_frame_image_paths(pseudo_frame, image_root, frame_name=frame_name)


def generate_pseudo_labels(member_checkpoints: list[str], weights: list[float], config: dict) -> pd.DataFrame:
    import torch

    test_wide = pd.read_parquet(config["data"]["test_manifest"])
    validate_frame_image_paths(test_wide, config["data"]["image_root"], frame_name="test_wide_for_pseudo")
    dataloader = build_prediction_dataloader(test_wide, config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        "[pseudo-teacher] "
        f"device={device} "
        f"test_samples={len(test_wide)} "
        f"batch_size={dataloader.batch_size} "
        f"num_workers={dataloader.num_workers}"
    )

    weighted_frames = []
    total_weight = 0.0
    for checkpoint_path, weight in zip(member_checkpoints, weights, strict=True):
        model = load_model_from_checkpoint(checkpoint_path, device)
        model_device = next(model.parameters()).device
        print(
            "[pseudo-teacher] "
            f"checkpoint={checkpoint_path} "
            f"weight={weight} "
            f"model_device={model_device}"
        )
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
    _validate_pseudo_frame(pseudo, config["data"]["image_root"], frame_name="pseudo_frame")
    return pseudo


def merge_train_and_pseudo(
    train_manifest: str | Path,
    pseudo_frame: pd.DataFrame,
    output_path: str | Path,
    *,
    image_root: str | Path,
) -> str:
    train_wide = pd.read_parquet(train_manifest)
    validate_frame_image_paths(train_wide, image_root, frame_name="train_wide")
    _validate_pseudo_frame(pseudo_frame, image_root, frame_name="pseudo_frame")
    combined = pd.concat([train_wide, pseudo_frame], axis=0, ignore_index=True, sort=False)
    validate_frame_image_paths(combined, image_root, frame_name="combined_train_plus_pseudo")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return str(output_path)


def run_pseudo_training(config: dict) -> None:
    _log_pseudo_environment(config)
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
            image_root=config["data"]["image_root"],
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
