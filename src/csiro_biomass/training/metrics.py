"""Validation and OOF metric helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from csiro_biomass.data.constants import TARGET_COLUMNS, TARGET_TO_WEIGHT


def build_validation_frame(
    prediction_frame: pd.DataFrame,
    truth_frame: pd.DataFrame,
    *,
    fold: int,
    seed: int,
) -> pd.DataFrame:
    merged = truth_frame[["image_id", *TARGET_COLUMNS]].merge(
        prediction_frame,
        on="image_id",
        how="inner",
        suffixes=("_true", "_pred"),
    )
    merged["fold"] = fold
    merged["seed"] = seed
    return merged


def compute_per_target_metrics(validation_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        true_col = f"{target}_true"
        pred_col = f"{target}_pred"
        diff = validation_frame[pred_col] - validation_frame[true_col]
        corr = validation_frame[true_col].corr(validation_frame[pred_col])
        rows.append(
            {
                "target": target,
                "corr": None if pd.isna(corr) else float(corr),
                "mae": float(diff.abs().mean()),
                "rmse": float(np.sqrt(np.square(diff).mean())),
                "pred_mean": float(validation_frame[pred_col].mean()),
                "pred_std": float(validation_frame[pred_col].std(ddof=1)),
                "true_mean": float(validation_frame[true_col].mean()),
                "true_std": float(validation_frame[true_col].std(ddof=1)),
                "weighted_component": float(
                    target_weighted_r2(validation_frame[true_col], validation_frame[pred_col])
                ),
            }
        )
    return pd.DataFrame(rows)


def target_weighted_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    eps = 1e-8
    ss_res = float(np.square(y_true.to_numpy() - y_pred.to_numpy()).sum())
    centered = y_true.to_numpy() - float(y_true.mean())
    ss_tot = float(np.square(centered).sum())
    return 1.0 - ss_res / (ss_tot + eps)


def compute_weighted_r2_from_frame(validation_frame: pd.DataFrame) -> float:
    score = 0.0
    for target in TARGET_COLUMNS:
        score += TARGET_TO_WEIGHT[target] * target_weighted_r2(
            validation_frame[f"{target}_true"],
            validation_frame[f"{target}_pred"],
        )
    return float(score)


def summarize_validation(
    prediction_frame: pd.DataFrame,
    truth_frame: pd.DataFrame,
    *,
    fold: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_frame = build_validation_frame(prediction_frame, truth_frame, fold=fold, seed=seed)
    metrics = compute_per_target_metrics(validation_frame)
    return validation_frame, metrics


def save_metrics_csv(path: str | Path, metrics: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False)
