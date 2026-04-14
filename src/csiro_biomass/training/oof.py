"""OOF aggregation and teacher selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from csiro_biomass.data.constants import TARGET_COLUMNS
from csiro_biomass.training.metrics import (
    compute_per_target_metrics,
    compute_weighted_r2_from_frame,
    save_metrics_csv,
)
from csiro_biomass.utils.config import ensure_dir


def _run_dirs(experiment_root: Path) -> list[Path]:
    return sorted(path.parent for path in experiment_root.glob("**/valid_predictions.parquet"))


def aggregate_experiment_root(experiment_root: str | Path, train_manifest: str | Path) -> dict:
    experiment_root = Path(experiment_root)
    truth_frame = pd.read_parquet(train_manifest)[["image_id", *TARGET_COLUMNS]]

    run_dirs = _run_dirs(experiment_root)
    if not run_dirs:
        raise FileNotFoundError(f"No valid_predictions.parquet files found under {experiment_root}")

    run_frames = []
    run_summaries = []
    for run_dir in run_dirs:
        prediction_frame = pd.read_parquet(run_dir / "valid_predictions.parquet")
        run_frames.append(prediction_frame)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            run_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    combined = pd.concat(run_frames, ignore_index=True)
    pred_columns = [f"{target}_pred" for target in TARGET_COLUMNS]
    if not set(pred_columns).issubset(combined.columns) and set(TARGET_COLUMNS).issubset(combined.columns):
        combined = combined.rename(columns={target: f"{target}_pred" for target in TARGET_COLUMNS})
    combined = combined.groupby("image_id", as_index=False)[pred_columns].mean()
    oof = truth_frame.merge(combined, on="image_id", how="inner")
    oof = oof.rename(columns={target: f"{target}_true" for target in TARGET_COLUMNS})
    metrics = compute_per_target_metrics(oof)
    overall = {
        "experiment_root": str(experiment_root),
        "num_runs": len(run_dirs),
        "num_images": int(len(oof)),
        "oof_weighted_r2": float(compute_weighted_r2_from_frame(oof)),
    }

    oof.to_parquet(experiment_root / "oof_predictions.parquet", index=False)
    save_metrics_csv(experiment_root / "oof_metrics.csv", metrics)
    (experiment_root / "run_summaries.json").write_text(json.dumps(run_summaries, indent=2), encoding="utf-8")
    (experiment_root / "oof_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    return overall


def _load_oof_frame(experiment_root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(experiment_root / "oof_predictions.parquet")
    if set(TARGET_COLUMNS).issubset(frame.columns):
        return frame[["image_id", *TARGET_COLUMNS]].copy()
    rename_map = {f"{target}_pred": target for target in TARGET_COLUMNS}
    return frame[["image_id", *(f"{target}_pred" for target in TARGET_COLUMNS)]].rename(columns=rename_map)


def select_teachers(
    experiment_roots: list[str | Path],
    *,
    top_k: int = 4,
    correlation_threshold: float = 0.985,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    frames = {}
    for root in map(Path, experiment_roots):
        summary = json.loads((root / "oof_summary.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(root / "oof_metrics.csv")
        avg_corr = float(metrics["corr"].fillna(0.0).mean())
        row = {
            "experiment_root": str(root),
            "oof_weighted_r2": summary["oof_weighted_r2"],
            "avg_corr": avg_corr,
        }
        rows.append(row)
        frames[str(root)] = _load_oof_frame(root)

    ranking = pd.DataFrame(rows).sort_values(
        by=["oof_weighted_r2", "avg_corr"],
        ascending=False,
    )
    selected_rows = []
    correlation_records = []
    selected_frames = {}
    for _, row in ranking.iterrows():
        root = row["experiment_root"]
        current_frame = frames[root]
        keep = True
        for selected_root, selected_frame in selected_frames.items():
            merged = current_frame.merge(selected_frame, on="image_id", suffixes=("_a", "_b"))
            corr_values = []
            for target in TARGET_COLUMNS:
                corr_values.append(merged[f"{target}_a"].corr(merged[f"{target}_b"]))
            mean_corr = float(pd.Series(corr_values).fillna(0.0).mean())
            correlation_records.append(
                {
                    "candidate": root,
                    "selected": selected_root,
                    "mean_prediction_corr": mean_corr,
                }
            )
            if mean_corr >= correlation_threshold:
                keep = False
                break
        if keep:
            selected_rows.append(row.to_dict())
            selected_frames[root] = current_frame
        if len(selected_rows) >= top_k:
            break

    return pd.DataFrame(selected_rows), pd.DataFrame(correlation_records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate OOF runs or select teachers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate a single experiment root")
    aggregate.add_argument("--experiment-root", required=True)
    aggregate.add_argument("--train-manifest", required=True)

    select = subparsers.add_parser("select", help="Select top diverse teachers across experiment roots")
    select.add_argument("--experiment-root", action="append", required=True)
    select.add_argument("--output-dir", required=True)
    select.add_argument("--top-k", type=int, default=4)
    select.add_argument("--correlation-threshold", type=float, default=0.985)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate_experiment_root(args.experiment_root, args.train_manifest)
    elif args.command == "select":
        selected, correlations = select_teachers(
            args.experiment_root,
            top_k=args.top_k,
            correlation_threshold=args.correlation_threshold,
        )
        output_dir = ensure_dir(args.output_dir)
        selected.to_csv(output_dir / "teacher_selection.csv", index=False)
        correlations.to_csv(output_dir / "teacher_correlations.csv", index=False)
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
