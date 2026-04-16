from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from csiro_biomass.data.constants import TARGET_COLUMNS
from csiro_biomass.data.dataset import validate_frame_image_paths
from csiro_biomass.training.pseudo import merge_train_and_pseudo


def _make_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(path)


def _base_row(image_id: str, image_path: str) -> dict:
    row = {
        "image_id": image_id,
        "image_path": image_path,
        "Sampling_Date": None,
        "State": None,
        "Species": None,
        "Pre_GSHH_NDVI": None,
        "Height_Ave_cm": None,
        "cv_group": "g",
        "is_pseudo": False,
    }
    for target in TARGET_COLUMNS:
        row[target] = 1.0
    return row


def test_validate_frame_image_paths_rejects_nan_image_path(tmp_path: Path) -> None:
    frame = pd.DataFrame([_base_row("ID1", pd.NA)])

    with pytest.raises(ValueError, match="missing image_path"):
        validate_frame_image_paths(frame, tmp_path, frame_name="train_frame")


def test_validate_frame_image_paths_rejects_missing_file(tmp_path: Path) -> None:
    frame = pd.DataFrame([_base_row("ID1", "train/ID1.jpg")])

    with pytest.raises(FileNotFoundError, match="do not exist"):
        validate_frame_image_paths(frame, tmp_path, frame_name="train_frame")


def test_merge_train_and_pseudo_fails_fast_on_bad_pseudo_image_path(tmp_path: Path) -> None:
    train_image = tmp_path / "train" / "ID1.jpg"
    _make_rgb(train_image)

    train_manifest = tmp_path / "train_wide.parquet"
    pd.DataFrame([_base_row("ID1", "train/ID1.jpg")]).to_parquet(train_manifest, index=False)

    pseudo_row = _base_row("ID2", pd.NA)
    pseudo_row["cv_group"] = "pseudo"
    pseudo_row["is_pseudo"] = True
    pseudo_frame = pd.DataFrame([pseudo_row])

    with pytest.raises(ValueError, match="pseudo_frame contains"):
        merge_train_and_pseudo(
            train_manifest,
            pseudo_frame,
            tmp_path / "combined.parquet",
            image_root=tmp_path,
        )
