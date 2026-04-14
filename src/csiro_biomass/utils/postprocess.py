"""Prediction post-processing."""

from __future__ import annotations

from typing import Mapping

from csiro_biomass.data.constants import TARGET_COLUMNS


def apply_rule_based_postprocess(row: Mapping[str, float]) -> dict[str, float]:
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


def clamp_prediction_dict(row: Mapping[str, float]) -> dict[str, float]:
    return {key: max(0.0, float(row[key])) for key in TARGET_COLUMNS}
