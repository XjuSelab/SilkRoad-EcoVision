# Server Training

This repository now supports a score-first server workflow for the CSIRO dataset.

## Core commands

Run a full fold/seed sweep for a single experiment root:

```bash
bash scripts/train_modelscope_dinov3.sh 896
```

Each experiment root writes:

```text
artifacts/server/<experiment>/
  fold0_seed42/
  fold0_seed3407/
  fold1_seed42/
  ...
  oof_predictions.parquet
  oof_metrics.csv
  oof_summary.json
  run_summaries.json
```

## Teacher selection

After multiple experiment roots finish:

```bash
uv run csiro-biomass oof select \
  --experiment-root artifacts/server/dinov3-vitl-896-modelscope \
  --experiment-root artifacts/server/dinov3-vitl-1024-modelscope \
  --experiment-root artifacts/server/dinov2-vitl-518 \
  --experiment-root artifacts/server/dinov2-vitg-518 \
  --experiment-root artifacts/server/siglip-so400m-384 \
  --experiment-root artifacts/server/siglip-so400m-448 \
  --output-dir artifacts/server/teacher-selection \
  --top-k 4 \
  --correlation-threshold 0.985
```

This writes:

- `teacher_selection.csv`
- `teacher_correlations.csv`

## Inference

Use TTA and optional checkpoint averaging:

```yaml
infer:
  tta_policies: [identity, hflip, vflip, rot90]
  members:
    - checkpoints:
        - artifacts/server/dinov3-vitl-1024-modelscope/fold0_seed42/best.pt
        - artifacts/server/dinov3-vitl-1024-modelscope/fold0_seed3407/best.pt
      weight: 0.35
```

`checkpoints` averages multiple checkpoints into one ensemble member before model-level weighting.
