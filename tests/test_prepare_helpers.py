from pathlib import Path

import pandas as pd

from csiro_biomass.data.prepare import build_fold_manifest, pivot_test_frame, pivot_training_frame, split_sample_id


def test_split_sample_id() -> None:
    assert split_sample_id("ID1011485656__Dry_Clover_g") == "ID1011485656"


def test_pivot_training_frame() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "ID1__Dry_Green_g",
                "image_path": "train/ID1.jpg",
                "Sampling_Date": "2015/1/1",
                "State": "NSW",
                "Species": "Ryegrass",
                "Pre_GSHH_NDVI": "0.5",
                "Height_Ave_cm": "2.5",
                "target_name": "Dry_Green_g",
                "target": "11.0",
            },
            {
                "sample_id": "ID1__Dry_Dead_g",
                "image_path": "train/ID1.jpg",
                "Sampling_Date": "2015/1/1",
                "State": "NSW",
                "Species": "Ryegrass",
                "Pre_GSHH_NDVI": "0.5",
                "Height_Ave_cm": "2.5",
                "target_name": "Dry_Dead_g",
                "target": "4.0",
            },
        ]
    )
    wide = pivot_training_frame(frame)
    assert wide.loc[0, "image_id"] == "ID1"
    assert wide.loc[0, "Dry_Green_g"] == 11.0
    assert wide.loc[0, "Dry_Dead_g"] == 4.0


def test_pivot_test_frame_and_folds() -> None:
    test_frame = pd.DataFrame(
        [
            {"sample_id": "ID2__Dry_Green_g", "image_path": "test/ID2.jpg", "target_name": "Dry_Green_g"},
            {"sample_id": "ID2__Dry_Dead_g", "image_path": "test/ID2.jpg", "target_name": "Dry_Dead_g"},
        ]
    )
    wide = pivot_test_frame(test_frame)
    assert wide.loc[0, "image_id"] == "ID2"

    train_wide = pd.DataFrame(
        {
            "image_id": ["A", "B", "C"],
            "cv_group": ["x", "y", "z"],
        }
    )
    folds = build_fold_manifest(train_wide, n_splits=3)
    assert set(folds["fold"]) == {0, 1, 2}
