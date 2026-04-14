from csiro_biomass.utils.postprocess import apply_rule_based_postprocess


def test_apply_rule_based_postprocess() -> None:
    row = {
        "Dry_Green_g": 20.0,
        "Dry_Dead_g": 25.0,
        "Dry_Clover_g": 10.0,
        "GDM_g": 30.0,
        "Dry_Total_g": 45.0,
    }
    processed = apply_rule_based_postprocess(row)
    assert processed["Dry_Clover_g"] == 8.0
    assert processed["Dry_Dead_g"] == 27.500000000000004
    assert processed["Dry_Total_g"] > 0.0
