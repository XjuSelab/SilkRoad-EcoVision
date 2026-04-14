"""Torch datasets and transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from csiro_biomass.data.constants import BORDERS_DICT, TARGET_COLUMNS


INTERPOLATION_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def split_left_right(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    midpoint = image.shape[1] // 2
    return image[:, :midpoint].copy(), image[:, midpoint:].copy()


def apply_random_black_padding(image: np.ndarray, probability: float = 0.2) -> np.ndarray:
    if np.random.random() >= probability:
        return image

    background = np.zeros_like(image)
    resize_ratio = float(np.random.uniform(0.85, 1.0))
    resized = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_CUBIC)
    h, w = resized.shape[:2]
    bg_h, bg_w = background.shape[:2]
    top = int(np.random.randint(0, bg_h - h + 1))
    left = int(np.random.randint(0, bg_w - w + 1))
    background[top : top + h, left : left + w] = resized
    return background


def _resolve_interpolation(name: str) -> int:
    return INTERPOLATION_MAP.get(name.lower(), cv2.INTER_CUBIC)


def build_train_transforms(
    image_size: int,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    interpolation: str,
) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.75),
            A.Normalize(mean=mean, std=std),
            A.Resize(image_size, image_size, interpolation=_resolve_interpolation(interpolation)),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(
    image_size: int,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    interpolation: str,
) -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            A.Resize(image_size, image_size, interpolation=_resolve_interpolation(interpolation)),
            ToTensorV2(),
        ]
    )


def make_interval_labels(targets: np.ndarray) -> np.ndarray:
    labels = []
    for target_idx, target_name in enumerate(TARGET_COLUMNS):
        borders = np.asarray(BORDERS_DICT[target_name], dtype=np.float32)
        labels.append(np.digitize(targets[:, target_idx], borders, right=False))
    return np.stack(labels, axis=1).astype(np.int64)


@dataclass(slots=True)
class DatasetConfig:
    image_root: str
    image_size: int
    train: bool
    black_padding_probability: float = 0.2
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: str = "bicubic"


class CSIROBiomassDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, config: DatasetConfig):
        self.frame = frame.reset_index(drop=True).copy()
        self.config = config
        self.image_root = Path(config.image_root)
        self.transforms = (
            build_train_transforms(
                config.image_size,
                mean=config.mean,
                std=config.std,
                interpolation=config.interpolation,
            )
            if config.train
            else build_eval_transforms(
                config.image_size,
                mean=config.mean,
                std=config.std,
                interpolation=config.interpolation,
            )
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.image_root / str(row["image_path"])
        image = np.array(Image.open(image_path).convert("RGB"))
        left_image, right_image = split_left_right(image)

        if self.config.train:
            left_image = apply_random_black_padding(
                left_image, probability=self.config.black_padding_probability
            )
            right_image = apply_random_black_padding(
                right_image, probability=self.config.black_padding_probability
            )

        left_tensor = self.transforms(image=left_image)["image"]
        right_tensor = self.transforms(image=right_image)["image"]

        sample = {
            "image_id": str(row["image_id"]),
            "left_image": left_tensor,
            "right_image": right_tensor,
        }

        has_targets = all(pd.notna(row[target]) for target in TARGET_COLUMNS)
        if has_targets:
            targets = row[TARGET_COLUMNS].to_numpy(dtype=np.float32)
            sample["targets"] = torch.tensor(targets, dtype=torch.float32)
            sample["cls_labels"] = torch.tensor(make_interval_labels(targets[None, :])[0], dtype=torch.long)
        else:
            sample["targets"] = torch.zeros(len(TARGET_COLUMNS), dtype=torch.float32)
            sample["cls_labels"] = torch.zeros(len(TARGET_COLUMNS), dtype=torch.long)

        return sample
