# CSIRO Biomass Reproduction Notes

## Local Development

- Dependency management: `uv`
- Local environment: `.venv`
- Code sync: local push to GitHub, server `git pull`
- Lock file policy: `uv.lock` not committed
- Hugging Face mirror: recommend `HF_ENDPOINT=https://hf-mirror.com` for local debugging

## Data Layout

The preparation command writes:

```text
data/raw/csiro-biomass.zip
data/raw/csiro-biomass/{train,test,train.csv,test.csv,sample_submission.csv}
data/processed/csiro-biomass/metadata/train_wide.parquet
data/processed/csiro-biomass/metadata/test_wide.parquet
data/processed/csiro-biomass/metadata/train_long.parquet
data/processed/csiro-biomass/metadata/test_long.parquet
data/processed/csiro-biomass/folds/folds_v1.parquet
```

`train_wide.parquet` keeps one row per image with metadata and the five biomass targets.

## Modeling Choices

- Input image is split into two `1000x1000` views by width.
- Active DINOv3 single-model routes are `timm` and local `ModelScope/transformers`.
- Legacy `torchhub` DINOv3 configs are archived under `configs/archive/torchhub/`.
- Outputs include five regression heads and five 7-class interval-classification heads.
- Loss is weighted `SmoothL1 + 0.3 * CrossEntropy`.
- Post-processing follows the exact rule set quoted in the winning write-up.

## Notes on Scope

- The local bundle currently exposes the public training set plus a visible one-image test sample.
- The code fully reproduces the training and pseudo-label pipeline structure, but hidden Kaggle leaderboard scores cannot be reproduced without hidden test access.
- Hugging Face upload is intentionally left as a later private-repo step; the current implementation prepares the manifests needed for that migration.
