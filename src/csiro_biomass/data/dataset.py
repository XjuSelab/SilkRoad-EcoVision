"""Torch datasets and transforms."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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

from csiro_biomass.data.constants import (
    BORDERS_DICT,
    DEFAULT_METADATA_CATEGORICAL_COLUMNS,
    DEFAULT_METADATA_NUMERIC_COLUMNS,
    TARGET_COLUMNS,
)


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


def _is_missing_image_path(value: Any) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none"}


def _format_bad_path_rows(
    frame: pd.DataFrame,
    *,
    indices: pd.Index,
    image_root: Path,
    max_examples: int = 5,
) -> str:
    examples = []
    for _, row in frame.loc[indices, ["image_id", "image_path"]].head(max_examples).iterrows():
        examples.append(
            f"image_id={row['image_id']}, image_path={row['image_path']}, "
            f"resolved={image_root / str(row['image_path'])}"
        )
    return "; ".join(examples)


def validate_frame_image_paths(frame: pd.DataFrame, image_root: str | Path, *, frame_name: str) -> None:
    image_root = Path(image_root)
    if "image_path" not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column 'image_path'")

    missing_mask = frame["image_path"].map(_is_missing_image_path)
    if bool(missing_mask.any()):
        bad_indices = frame.index[missing_mask]
        raise ValueError(
            f"{frame_name} contains {int(missing_mask.sum())} rows with missing image_path. "
            f"Examples: {_format_bad_path_rows(frame, indices=bad_indices, image_root=image_root)}"
        )

    resolved_paths = frame["image_path"].map(lambda value: image_root / str(value))
    exists_mask = resolved_paths.map(Path.exists)
    if not bool(exists_mask.all()):
        bad_indices = frame.index[~exists_mask]
        raise FileNotFoundError(
            f"{frame_name} contains {int((~exists_mask).sum())} rows whose image files do not exist. "
            f"Examples: {_format_bad_path_rows(frame, indices=bad_indices, image_root=image_root)}"
        )


def _safe_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return 0.0
    return float(numeric)


def _encode_sampling_date_cyclical(value: Any) -> list[float]:
    if pd.isna(value):
        return [0.0, 0.0]

    text = str(value).strip()
    if not text:
        return [0.0, 0.0]

    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            angle = 2.0 * np.pi * (parsed.timetuple().tm_yday / 366.0)
            return [float(np.sin(angle)), float(np.cos(angle))]
        except ValueError:
            continue
    return [0.0, 0.0]


def build_metadata_spec(
    frame: pd.DataFrame,
    *,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    include_sampling_date_cyclical: bool = True,
) -> dict[str, Any]:
    numeric_columns = list(numeric_columns or DEFAULT_METADATA_NUMERIC_COLUMNS)
    categorical_columns = list(categorical_columns or DEFAULT_METADATA_CATEGORICAL_COLUMNS)

    categorical_vocabs: dict[str, list[str]] = {}
    feature_dim = len(numeric_columns)
    if include_sampling_date_cyclical:
        feature_dim += 2

    for column in categorical_columns:
        values = (
            frame[column]
            .dropna()
            .map(lambda value: str(value).strip())
            .loc[lambda series: series != ""]
            .unique()
            .tolist()
        )
        values = sorted(values)
        categorical_vocabs[column] = values
        feature_dim += len(values) + 1

    return {
        "enabled": True,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categorical_vocabs": categorical_vocabs,
        "include_sampling_date_cyclical": bool(include_sampling_date_cyclical),
        "feature_dim": feature_dim,
    }


def encode_metadata_features(row: pd.Series, metadata_spec: dict[str, Any] | None) -> np.ndarray:
    if not metadata_spec or not metadata_spec.get("enabled", False):
        return np.zeros(0, dtype=np.float32)

    features: list[float] = []
    for column in metadata_spec.get("numeric_columns", []):
        features.append(_safe_float(row.get(column)))

    if metadata_spec.get("include_sampling_date_cyclical", False):
        features.extend(_encode_sampling_date_cyclical(row.get("Sampling_Date")))

    categorical_vocabs = metadata_spec.get("categorical_vocabs", {})
    for column in metadata_spec.get("categorical_columns", []):
        vocab = categorical_vocabs.get(column, [])
        one_hot = np.zeros(len(vocab) + 1, dtype=np.float32)
        raw_value = row.get(column)
        encoded_value = "" if pd.isna(raw_value) else str(raw_value).strip()
        if encoded_value in vocab:
            one_hot[vocab.index(encoded_value)] = 1.0
        else:
            one_hot[-1] = 1.0
        features.extend(one_hot.tolist())

    return np.asarray(features, dtype=np.float32)


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
    metadata_spec: dict[str, Any] | None = None


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
        raw_image_path = row["image_path"]
        if _is_missing_image_path(raw_image_path):
            raise FileNotFoundError(
                f"Missing image_path for image_id={row['image_id']} at dataset index={index}. "
                f"Raw image_path value={raw_image_path!r}"
            )

        image_path = self.image_root / str(raw_image_path)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found for image_id={row['image_id']} at dataset index={index}. "
                f"image_path={raw_image_path!r}, resolved_path={image_path}"
            )
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
        if self.config.metadata_spec and self.config.metadata_spec.get("enabled", False):
            metadata_features = encode_metadata_features(row, self.config.metadata_spec)
            sample["metadata_features"] = torch.tensor(metadata_features, dtype=torch.float32)

        has_targets = all(pd.notna(row[target]) for target in TARGET_COLUMNS)
        if has_targets:
            targets = row[TARGET_COLUMNS].to_numpy(dtype=np.float32)
            sample["targets"] = torch.tensor(targets, dtype=torch.float32)
            sample["cls_labels"] = torch.tensor(make_interval_labels(targets[None, :])[0], dtype=torch.long)
        else:
            sample["targets"] = torch.zeros(len(TARGET_COLUMNS), dtype=torch.float32)
            sample["cls_labels"] = torch.zeros(len(TARGET_COLUMNS), dtype=torch.long)

        return sample
