from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from csiro_biomass.data.constants import TARGET_COLUMNS
from csiro_biomass.training.metrics import compute_per_target_metrics, compute_weighted_r2_from_frame
from csiro_biomass.utils.postprocess import apply_rule_based_postprocess, clamp_prediction_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze OOF ensemble combinations with optional postprocess.")
    parser.add_argument("--train-manifest", required=True, help="Path to train_wide.parquet")
    parser.add_argument(
        "--experiment-root",
        action="append",
        required=True,
        help="Experiment root containing oof_predictions.parquet; repeat this flag for multiple candidates.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to store CSV outputs")
    parser.add_argument("--min-combination-size", type=int, default=1)
    parser.add_argument("--max-combination-size", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=15, help="Number of top combinations to print")
    return parser


def _prediction_columns() -> list[str]:
    return [f"{target}_pred" for target in TARGET_COLUMNS]


def load_truth_frame(train_manifest: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(train_manifest)[["image_id", *TARGET_COLUMNS]].copy()
    if frame["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image_id detected in train manifest: {train_manifest}")
    return frame


def load_prediction_frame(experiment_root: str | Path) -> pd.DataFrame:
    experiment_root = Path(experiment_root)
    path = experiment_root / "oof_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF predictions: {path}")

    frame = pd.read_parquet(path)
    pred_columns = _prediction_columns()
    if {"image_id", *pred_columns}.issubset(frame.columns):
        frame = frame[["image_id", *pred_columns]].copy()
    elif {"image_id", *TARGET_COLUMNS}.issubset(frame.columns):
        frame = frame[["image_id", *TARGET_COLUMNS]].rename(
            columns={target: f"{target}_pred" for target in TARGET_COLUMNS}
        )
    else:
        raise ValueError(
            f"Unsupported OOF schema in {path}. "
            f"Expected either target columns {TARGET_COLUMNS} or prediction columns {pred_columns}."
        )
    if frame["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image_id detected in {path}")
    return frame


def average_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No prediction frames provided")

    pred_columns = _prediction_columns()
    averaged = frames[0].set_index("image_id")[pred_columns].sort_index()
    for frame in frames[1:]:
        current = frame.set_index("image_id")[pred_columns].sort_index()
        if not averaged.index.equals(current.index):
            raise ValueError("Experiment roots do not share the same image_id set in oof_predictions.parquet")
        averaged = averaged.add(current)
    averaged = averaged / len(frames)
    return averaged.reset_index()


def build_validation_frame(truth_frame: pd.DataFrame, prediction_frame: pd.DataFrame) -> pd.DataFrame:
    validation = truth_frame.merge(prediction_frame, on="image_id", how="inner")
    if len(validation) != len(truth_frame):
        raise ValueError(
            f"Prediction frame covers {len(validation)} images, expected {len(truth_frame)}. "
            "Make sure every candidate has complete OOF predictions."
        )
    return validation.rename(columns={target: f"{target}_true" for target in TARGET_COLUMNS})


def apply_postprocess_to_predictions(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    wide = prediction_frame.rename(columns={f"{target}_pred": target for target in TARGET_COLUMNS})
    processed_rows = []
    for row in wide.to_dict("records"):
        processed = clamp_prediction_dict(apply_rule_based_postprocess(row))
        processed_rows.append({"image_id": row["image_id"], **processed})
    processed = pd.DataFrame(processed_rows)
    return processed.rename(columns={target: f"{target}_pred" for target in TARGET_COLUMNS})


def evaluate_predictions(
    truth_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    validation = build_validation_frame(truth_frame, prediction_frame)
    metrics = compute_per_target_metrics(validation)[["target", "corr", "mae", "rmse", "weighted_component"]]
    score = compute_weighted_r2_from_frame(validation)
    return float(score), metrics


def build_score_row(
    combination_name: str,
    combination_size: int,
    raw_score: float,
    raw_metrics: pd.DataFrame,
    post_score: float,
    post_metrics: pd.DataFrame,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "combination_size": combination_size,
        "combination": combination_name,
        "raw_oof_weighted_r2": raw_score,
        "post_oof_weighted_r2": post_score,
        "post_delta": post_score - raw_score,
    }
    raw_lookup = raw_metrics.set_index("target")["weighted_component"]
    post_lookup = post_metrics.set_index("target")["weighted_component"]
    for target in TARGET_COLUMNS:
        row[f"raw_{target}_weighted_component"] = float(raw_lookup.loc[target])
        row[f"post_{target}_weighted_component"] = float(post_lookup.loc[target])
    return row


def build_best_by_size(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for combination_size, group in scores.groupby("combination_size", sort=True):
        raw_best = group.sort_values(
            by=["raw_oof_weighted_r2", "post_oof_weighted_r2"],
            ascending=False,
        ).iloc[0]
        post_best = group.sort_values(
            by=["post_oof_weighted_r2", "post_delta"],
            ascending=False,
        ).iloc[0]
        rows.append(
            {
                "combination_size": int(combination_size),
                "mode": "raw",
                "combination": raw_best["combination"],
                "oof_weighted_r2": float(raw_best["raw_oof_weighted_r2"]),
                "post_delta": float(raw_best["post_delta"]),
            }
        )
        rows.append(
            {
                "combination_size": int(combination_size),
                "mode": "postprocess",
                "combination": post_best["combination"],
                "oof_weighted_r2": float(post_best["post_oof_weighted_r2"]),
                "post_delta": float(post_best["post_delta"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()

    if args.min_combination_size < 1:
        raise ValueError("--min-combination-size must be >= 1")
    if args.max_combination_size < args.min_combination_size:
        raise ValueError("--max-combination-size must be >= --min-combination-size")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_frame = load_truth_frame(args.train_manifest)

    experiment_roots = [Path(root) for root in args.experiment_root]
    if len(experiment_roots) < args.min_combination_size:
        raise ValueError(
            f"Need at least {args.min_combination_size} experiment roots, got {len(experiment_roots)}."
        )

    frames = {root.name: load_prediction_frame(root) for root in experiment_roots}
    max_combination_size = min(args.max_combination_size, len(frames))

    score_rows: list[dict[str, float | int | str]] = []
    metrics_rows: list[pd.DataFrame] = []

    for combination_size in range(args.min_combination_size, max_combination_size + 1):
        for combo_names in combinations(frames.keys(), combination_size):
            combination_name = " + ".join(combo_names)
            averaged_predictions = average_prediction_frames([frames[name] for name in combo_names])

            raw_score, raw_metrics = evaluate_predictions(truth_frame, averaged_predictions)
            raw_metrics = raw_metrics.copy()
            raw_metrics.insert(0, "mode", "raw")
            raw_metrics.insert(0, "combination_size", combination_size)
            raw_metrics.insert(0, "combination", combination_name)

            post_predictions = apply_postprocess_to_predictions(averaged_predictions)
            post_score, post_metrics = evaluate_predictions(truth_frame, post_predictions)
            post_metrics = post_metrics.copy()
            post_metrics.insert(0, "mode", "postprocess")
            post_metrics.insert(0, "combination_size", combination_size)
            post_metrics.insert(0, "combination", combination_name)

            score_rows.append(
                build_score_row(
                    combination_name=combination_name,
                    combination_size=combination_size,
                    raw_score=raw_score,
                    raw_metrics=raw_metrics,
                    post_score=post_score,
                    post_metrics=post_metrics,
                )
            )
            metrics_rows.extend([raw_metrics, post_metrics])

    scores = pd.DataFrame(score_rows).sort_values(
        by=["post_oof_weighted_r2", "raw_oof_weighted_r2"],
        ascending=False,
    )
    metrics = pd.concat(metrics_rows, ignore_index=True)
    best_by_size = build_best_by_size(scores)

    scores.to_csv(output_dir / "combination_scores.csv", index=False)
    metrics.to_csv(output_dir / "combination_metrics.csv", index=False)
    best_by_size.to_csv(output_dir / "best_by_size.csv", index=False)

    display_columns = [
        "combination_size",
        "combination",
        "raw_oof_weighted_r2",
        "post_oof_weighted_r2",
        "post_delta",
        "raw_Dry_Dead_g_weighted_component",
        "post_Dry_Dead_g_weighted_component",
        "raw_Dry_Clover_g_weighted_component",
        "post_Dry_Clover_g_weighted_component",
    ]
    print(scores[display_columns].head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
