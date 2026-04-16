"""Prediction post-processing."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from csiro_biomass.data.constants import TARGET_COLUMNS

DEFAULT_THIRD_PLACE_PARAMS: dict[str, Any] = {
    "clover_multiplier": 1.0,
    "dead_thresholds": [10.0, 20.0],
    "dead_multipliers": [1.0, 1.0, 1.0],
}


def _winner_legacy_postprocess(row: Mapping[str, float]) -> dict[str, float]:
    green = float(row["Dry_Green_g"])
    dead = float(row["Dry_Dead_g"])
    clover = float(row["Dry_Clover_g"]) * 0.8
    gdm = float(row["GDM_g"])
    total = float(row["Dry_Total_g"])

    if dead > 20:
        dead *= 1.1
    elif dead < 10:
        dead *= 0.9

    gdm = 0.5 * gdm + 0.5 * (green + clover)
    pred_total1 = green + clover + dead
    total = 0.5 * total + 0.5 * pred_total1

    return {
        "Dry_Green_g": green,
        "Dry_Dead_g": dead,
        "Dry_Clover_g": clover,
        "GDM_g": gdm,
        "Dry_Total_g": total,
    }


def _third_place_oof_scaled_postprocess(
    row: Mapping[str, float],
    params: Mapping[str, Any] | None,
) -> dict[str, float]:
    merged_params = {**DEFAULT_THIRD_PLACE_PARAMS, **(params or {})}

    green = float(row["Dry_Green_g"])
    dead = float(row["Dry_Dead_g"])
    clover = float(row["Dry_Clover_g"]) * float(merged_params["clover_multiplier"])

    thresholds = [float(value) for value in merged_params["dead_thresholds"]]
    multipliers = [float(value) for value in merged_params["dead_multipliers"]]
    if len(thresholds) != 2 or len(multipliers) != 3:
        raise ValueError("third_place_oof_scaled expects 2 dead thresholds and 3 dead multipliers")

    if dead < thresholds[0]:
        dead *= multipliers[0]
    elif dead < thresholds[1]:
        dead *= multipliers[1]
    else:
        dead *= multipliers[2]

    gdm = green + clover
    total = green + dead + clover
    return {
        "Dry_Green_g": green,
        "Dry_Dead_g": dead,
        "Dry_Clover_g": clover,
        "GDM_g": gdm,
        "Dry_Total_g": total,
    }


def resolve_postprocess_strategy(infer_config: Mapping[str, Any]) -> str:
    if "postprocess_strategy" in infer_config:
        return str(infer_config["postprocess_strategy"])
    if infer_config.get("apply_postprocess", True):
        return "winner_legacy"
    return "none"


def apply_postprocess(
    row: Mapping[str, float],
    *,
    strategy: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    if strategy == "none":
        return {key: float(row[key]) for key in TARGET_COLUMNS}
    if strategy == "winner_legacy":
        return _winner_legacy_postprocess(row)
    if strategy == "third_place_oof_scaled":
        return _third_place_oof_scaled_postprocess(row, params)
    raise ValueError(f"Unsupported postprocess strategy: {strategy}")


def apply_rule_based_postprocess(row: Mapping[str, float]) -> dict[str, float]:
    """Backward-compatible alias for the legacy winner-style rules."""
    return apply_postprocess(row, strategy="winner_legacy")


def clamp_prediction_dict(row: Mapping[str, float]) -> dict[str, float]:
    return {key: max(0.0, float(row[key])) for key in TARGET_COLUMNS}


def _safe_median_ratio(true_values: pd.Series, pred_values: pd.Series) -> float:
    mask = pred_values > 0
    if not mask.any():
        return 1.0
    ratio = (true_values[mask] / pred_values[mask]).replace([float("inf"), float("-inf")], pd.NA).dropna()
    if ratio.empty:
        return 1.0
    return float(ratio.median())


def _clip_multiplier(value: float) -> float:
    return float(min(1.3, max(0.7, value)))


def fit_third_place_params(validation_frame: pd.DataFrame) -> dict[str, Any]:
    clover_pred = validation_frame["Dry_Clover_g_pred"]
    clover_true = validation_frame["Dry_Clover_g_true"]
    clover_mask = clover_pred > 0.5
    if clover_mask.any():
        clover_multiplier = _clip_multiplier(_safe_median_ratio(clover_true[clover_mask], clover_pred[clover_mask]))
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
            dead_multipliers.append(_clip_multiplier(_safe_median_ratio(dead_true[mask], dead_pred[mask])))
        else:
            dead_multipliers.append(1.0)

    return {
        "clover_multiplier": round(clover_multiplier, 6),
        "dead_thresholds": dead_thresholds,
        "dead_multipliers": [round(value, 6) for value in dead_multipliers],
    }
