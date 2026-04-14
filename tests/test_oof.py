from pathlib import Path

import pandas as pd

from csiro_biomass.training.oof import aggregate_experiment_root, select_teachers


def _write_run(root: Path, fold: int, seed: int, rows: list[dict], score: float) -> None:
    run_dir = root / f"fold{fold}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(run_dir / "valid_predictions.parquet", index=False)
    (run_dir / "summary.json").write_text(
        f'{{"fold": {fold}, "seed": {seed}, "best_valid_r2": {score}}}',
        encoding="utf-8",
    )


def test_aggregate_experiment_root(tmp_path: Path) -> None:
    truth = pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "Dry_Green_g": [1.0, 2.0],
            "Dry_Dead_g": [1.0, 2.0],
            "Dry_Clover_g": [1.0, 2.0],
            "GDM_g": [1.0, 2.0],
            "Dry_Total_g": [1.0, 2.0],
        }
    )
    truth_path = tmp_path / "train.parquet"
    truth.to_parquet(truth_path, index=False)

    experiment_root = tmp_path / "exp"
    _write_run(
        experiment_root,
        fold=0,
        seed=42,
        rows=[
            {
                "image_id": "a",
                "Dry_Green_g_true": 1.0,
                "Dry_Dead_g_true": 1.0,
                "Dry_Clover_g_true": 1.0,
                "GDM_g_true": 1.0,
                "Dry_Total_g_true": 1.0,
                "Dry_Green_g_pred": 1.2,
                "Dry_Dead_g_pred": 1.2,
                "Dry_Clover_g_pred": 1.2,
                "GDM_g_pred": 1.2,
                "Dry_Total_g_pred": 1.2,
                "fold": 0,
                "seed": 42,
            }
        ],
        score=0.1,
    )
    _write_run(
        experiment_root,
        fold=1,
        seed=42,
        rows=[
            {
                "image_id": "b",
                "Dry_Green_g_true": 2.0,
                "Dry_Dead_g_true": 2.0,
                "Dry_Clover_g_true": 2.0,
                "GDM_g_true": 2.0,
                "Dry_Total_g_true": 2.0,
                "Dry_Green_g_pred": 1.8,
                "Dry_Dead_g_pred": 1.8,
                "Dry_Clover_g_pred": 1.8,
                "GDM_g_pred": 1.8,
                "Dry_Total_g_pred": 1.8,
                "fold": 1,
                "seed": 42,
            }
        ],
        score=0.2,
    )

    summary = aggregate_experiment_root(experiment_root, truth_path)
    assert summary["num_runs"] == 2
    assert (experiment_root / "oof_predictions.parquet").exists()
    assert (experiment_root / "oof_metrics.csv").exists()


def test_select_teachers(tmp_path: Path) -> None:
    roots = []
    for idx, scale in enumerate([1.0, 1.2], start=1):
        root = tmp_path / f"exp{idx}"
        root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "image_id": ["a", "b"],
                "Dry_Green_g_true": [1.0, 2.0],
                "Dry_Dead_g_true": [1.0, 2.0],
                "Dry_Clover_g_true": [1.0, 2.0],
                "GDM_g_true": [1.0, 2.0],
                "Dry_Total_g_true": [1.0, 2.0],
                "Dry_Green_g_pred": [1.0, 2.0 * scale],
                "Dry_Dead_g_pred": [1.0, 2.0 * scale],
                "Dry_Clover_g_pred": [1.0, 2.0 * scale],
                "GDM_g_pred": [1.0, 2.0 * scale],
                "Dry_Total_g_pred": [1.0, 2.0 * scale],
            }
        ).to_parquet(root / "oof_predictions.parquet", index=False)
        pd.DataFrame({"target": ["Dry_Green_g"], "corr": [0.5 + idx], "mae": [0.1]}).to_csv(
            root / "oof_metrics.csv",
            index=False,
        )
        (root / "oof_summary.json").write_text(
            f'{{"oof_weighted_r2": {0.1 * idx}}}',
            encoding="utf-8",
        )
        roots.append(root)

    selected, correlations = select_teachers(roots, top_k=2, correlation_threshold=1.1)
    assert len(selected) == 2
    assert correlations is not None
