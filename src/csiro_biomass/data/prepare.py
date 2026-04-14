"""Data preparation pipeline for the CSIRO competition bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from csiro_biomass.data.constants import METADATA_COLUMNS, TARGET_COLUMNS
from csiro_biomass.utils.config import ensure_dir


def split_sample_id(sample_id: str) -> str:
    return sample_id.split("__", maxsplit=1)[0]


def load_competition_tables(zip_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        train_df = pd.read_csv(archive.open("train.csv"))
        test_df = pd.read_csv(archive.open("test.csv"))
        submission_df = pd.read_csv(archive.open("sample_submission.csv"))
    return train_df, test_df, submission_df


def pivot_training_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    frame = train_df.copy()
    frame["image_id"] = frame["sample_id"].map(split_sample_id)
    frame["Pre_GSHH_NDVI"] = pd.to_numeric(frame["Pre_GSHH_NDVI"], errors="coerce")
    frame["Height_Ave_cm"] = pd.to_numeric(frame["Height_Ave_cm"], errors="coerce")
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")

    targets = (
        frame.pivot_table(index="image_id", columns="target_name", values="target", aggfunc="first")
        .reset_index()
        .rename_axis(columns=None)
    )
    metadata = frame.groupby("image_id", as_index=False).first()[METADATA_COLUMNS]
    wide = metadata.merge(targets, on="image_id", how="inner")
    wide["cv_group"] = wide["State"].astype(str) + "__" + wide["Sampling_Date"].astype(str)
    wide["is_pseudo"] = False
    return wide


def pivot_test_frame(test_df: pd.DataFrame) -> pd.DataFrame:
    frame = test_df.copy()
    frame["image_id"] = frame["sample_id"].map(split_sample_id)
    metadata = frame.groupby("image_id", as_index=False).first()[["image_id", "image_path"]]
    for column in ["Sampling_Date", "State", "Species"]:
        metadata[column] = None
    metadata["Pre_GSHH_NDVI"] = None
    metadata["Height_Ave_cm"] = None
    for target in TARGET_COLUMNS:
        metadata[target] = pd.NA
    metadata["cv_group"] = "test"
    metadata["is_pseudo"] = False
    return metadata[METADATA_COLUMNS + TARGET_COLUMNS + ["cv_group", "is_pseudo"]]


def build_fold_manifest(train_wide: pd.DataFrame, n_splits: int = 3) -> pd.DataFrame:
    frame = train_wide.copy()
    frame["fold"] = -1
    splitter = GroupKFold(n_splits=n_splits)
    groups = frame["cv_group"].to_numpy()
    for fold, (_, valid_idx) in enumerate(splitter.split(frame, groups=groups)):
        frame.loc[valid_idx, "fold"] = fold
    return frame[["image_id", "cv_group", "fold"]]


def extract_archive(zip_path: str | Path, output_dir: str | Path) -> None:
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def maybe_copy_bundle(zip_path: Path, raw_dir: Path) -> Path:
    raw_dir = ensure_dir(raw_dir)
    bundle_target = raw_dir / zip_path.name
    if zip_path.resolve() != bundle_target.resolve():
        shutil.copy2(zip_path, bundle_target)
    return bundle_target


def write_summary(
    train_wide: pd.DataFrame,
    test_wide: pd.DataFrame,
    fold_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    summary = {
        "train_images": int(len(train_wide)),
        "test_images": int(len(test_wide)),
        "fold_counts": {str(k): int(v) for k, v in fold_df["fold"].value_counts().sort_index().items()},
        "state_date_groups": int(train_wide["cv_group"].nunique()),
        "targets": TARGET_COLUMNS,
    }
    Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_prepare_data(args: argparse.Namespace) -> None:
    zip_path = Path(args.zip_path)
    raw_root = ensure_dir(args.raw_dir)
    processed_root = ensure_dir(args.processed_dir)
    raw_extract_dir = raw_root / "csiro-biomass"
    metadata_dir = ensure_dir(processed_root / "metadata")
    folds_dir = ensure_dir(processed_root / "folds")

    maybe_copy_bundle(zip_path, raw_root)
    if args.extract_images:
        extract_archive(zip_path, raw_extract_dir)

    train_df, test_df, submission_df = load_competition_tables(zip_path)
    train_wide = pivot_training_frame(train_df)
    test_wide = pivot_test_frame(test_df)
    fold_df = build_fold_manifest(train_wide, n_splits=args.n_splits)

    train_wide.to_parquet(metadata_dir / "train_wide.parquet", index=False)
    test_wide.to_parquet(metadata_dir / "test_wide.parquet", index=False)
    train_df.to_parquet(metadata_dir / "train_long.parquet", index=False)
    test_df.to_parquet(metadata_dir / "test_long.parquet", index=False)
    submission_df.to_csv(metadata_dir / "submission_template.csv", index=False)
    fold_df.to_parquet(folds_dir / "folds_v1.parquet", index=False)
    write_summary(train_wide, test_wide, fold_df, metadata_dir / "summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CSIRO biomass manifests from the Kaggle bundle.")
    parser.add_argument("--zip-path", required=True, help="Path to csiro-biomass.zip")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory for the raw bundle and extraction")
    parser.add_argument(
        "--processed-dir",
        default="data/processed/csiro-biomass",
        help="Directory for processed metadata/manifests",
    )
    parser.add_argument("--n-splits", type=int, default=3, help="Number of CV folds")
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Extract the archive into data/raw/csiro-biomass/ before building manifests",
    )
    return parser


def main() -> None:
    parser = build_parser()
    run_prepare_data(parser.parse_args())


if __name__ == "__main__":
    main()
