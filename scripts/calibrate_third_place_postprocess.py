from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from csiro_biomass.data.constants import TARGET_COLUMNS
from csiro_biomass.utils.postprocess import fit_third_place_params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit third-place style postprocess params from OOF predictions.")
    parser.add_argument("--train-manifest", required=True, help="Path to train_wide.parquet")
    parser.add_argument("--prediction-path", required=True, help="Path to oof_predictions.parquet")
    parser.add_argument("--output-yaml", required=True, help="Path to write fitted postprocess params")
    return parser


def load_truth_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)[["image_id", *TARGET_COLUMNS]].copy()


def load_prediction_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    pred_columns = [f"{target}_pred" for target in TARGET_COLUMNS]
    if {"image_id", *pred_columns}.issubset(frame.columns):
        return frame[["image_id", *pred_columns]].copy()
    if {"image_id", *TARGET_COLUMNS}.issubset(frame.columns):
        return frame[["image_id", *TARGET_COLUMNS]].rename(columns={target: f"{target}_pred" for target in TARGET_COLUMNS})
    raise ValueError(f"Unsupported prediction schema in {path}")


def main() -> None:
    args = build_parser().parse_args()
    truth = load_truth_frame(args.train_manifest)
    prediction = load_prediction_frame(args.prediction_path)
    validation = truth.merge(prediction, on="image_id", how="inner").rename(
        columns={target: f"{target}_true" for target in TARGET_COLUMNS}
    )
    params = fit_third_place_params(validation)
    output_path = Path(args.output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(params, sort_keys=False))


if __name__ == "__main__":
    main()
