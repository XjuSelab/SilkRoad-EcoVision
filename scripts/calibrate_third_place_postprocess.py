from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from csiro_biomass.data.constants import TARGET_COLUMNS


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


def _safe_median_ratio(true_values: pd.Series, pred_values: pd.Series) -> float:
    mask = pred_values > 0
    if not mask.any():
        return 1.0
    ratio = (true_values[mask] / pred_values[mask]).replace([float("inf"), float("-inf")], pd.NA).dropna()
    if ratio.empty:
        return 1.0
    return float(ratio.median())


def clip_multiplier(value: float) -> float:
    return float(min(1.3, max(0.7, value)))


def fit_params(validation_frame: pd.DataFrame) -> dict[str, object]:
    clover_pred = validation_frame["Dry_Clover_g_pred"]
    clover_true = validation_frame["Dry_Clover_g_true"]
    clover_mask = clover_pred > 0.5
    if clover_mask.any():
        clover_multiplier = clip_multiplier(_safe_median_ratio(clover_true[clover_mask], clover_pred[clover_mask]))
    else:
        clover_multiplier = 1.0

    dead_pred = validation_frame["Dry_Dead_g_pred"]
    dead_true = validation_frame["Dry_Dead_g_true"]
    dead_thresholds = [10.0, 20.0]
    dead_masks = [
        dead_pred < dead_thresholds[0],
        (dead_pred >= dead_thresholds[0]) & (dead_pred < dead_thresholds[1]),
        dead_pred >= dead_thresholds[1],
    ]
    dead_multipliers = []
    for mask in dead_masks:
        if mask.any():
            dead_multipliers.append(clip_multiplier(_safe_median_ratio(dead_true[mask], dead_pred[mask])))
        else:
            dead_multipliers.append(1.0)

    return {
        "clover_multiplier": round(clover_multiplier, 6),
        "dead_thresholds": dead_thresholds,
        "dead_multipliers": [round(value, 6) for value in dead_multipliers],
    }


def main() -> None:
    args = build_parser().parse_args()
    truth = load_truth_frame(args.train_manifest)
    prediction = load_prediction_frame(args.prediction_path)
    validation = truth.merge(prediction, on="image_id", how="inner").rename(
        columns={target: f"{target}_true" for target in TARGET_COLUMNS}
    )
    params = fit_params(validation)
    output_path = Path(args.output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(params, sort_keys=False))


if __name__ == "__main__":
    main()
