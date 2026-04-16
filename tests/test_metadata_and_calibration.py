from __future__ import annotations

import pandas as pd

from csiro_biomass.data.dataset import build_metadata_spec, encode_metadata_features
from csiro_biomass.utils.postprocess import fit_third_place_params


def test_build_metadata_spec_and_encode_features() -> None:
    frame = pd.DataFrame(
        [
            {
                "Sampling_Date": "2015/09/04",
                "State": "Tas",
                "Species": "Ryegrass_Clover",
                "Pre_GSHH_NDVI": "0.62",
                "Height_Ave_cm": "4.6667",
            }
        ]
    )
    spec = build_metadata_spec(frame)

    encoded = encode_metadata_features(frame.iloc[0], spec)
    assert encoded.shape == (spec["feature_dim"],)
    assert spec["feature_dim"] > 4
    assert encoded[0] == 0.62
    assert encoded[1] == 4.6667


def test_fit_third_place_params_returns_clipped_segment_scaling() -> None:
    validation = pd.DataFrame(
        {
            "Dry_Clover_g_true": [2.0, 10.0, 30.0],
            "Dry_Clover_g_pred": [1.0, 8.0, 20.0],
            "Dry_Dead_g_true": [4.0, 24.0, 54.0],
            "Dry_Dead_g_pred": [5.0, 20.0, 45.0],
        }
    )

    params = fit_third_place_params(validation)
    assert params["dead_thresholds"] == [10.0, 20.0]
    assert len(params["dead_multipliers"]) == 3
    assert 0.7 <= params["clover_multiplier"] <= 1.3
