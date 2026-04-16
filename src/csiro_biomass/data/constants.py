"""Dataset-wide constants."""

from __future__ import annotations

BASE_TARGET_COLUMNS = [
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
]

DERIVED_TARGET_COLUMNS = [
    "GDM_g",
    "Dry_Total_g",
]

TARGET_COLUMNS = [*BASE_TARGET_COLUMNS, *DERIVED_TARGET_COLUMNS]

TARGET_TO_WEIGHT = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}

BORDERS_DICT = {
    "Dry_Clover_g": [1.6e-05, 3.9, 10.5353, 20.6523, 37.5911, 71.7865],
    "Dry_Dead_g": [1.6e-05, 6.1407, 13.1192, 23.277, 38.8581, 83.8407],
    "Dry_Green_g": [1.6e-05, 13.4232, 27.0782, 45.5236, 79.834, 157.9836],
    "Dry_Total_g": [1.6e-05, 23.4907, 41.1, 61.1, 96.8288, 185.7],
    "GDM_g": [1.6e-05, 16.5143, 30.507, 49.5585, 81.0, 157.9836],
}

METADATA_COLUMNS = [
    "image_id",
    "image_path",
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
]

DEFAULT_METADATA_NUMERIC_COLUMNS = [
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
]

DEFAULT_METADATA_CATEGORICAL_COLUMNS = [
    "State",
    "Species",
]
