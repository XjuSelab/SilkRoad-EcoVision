# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reproduction of the CSIRO `Image2Biomass` competition solutions. The package name is
`csiro_biomass` (installed as `silkroad-ecovision`) and exposes a single CLI entrypoint
`csiro-biomass`. Python is pinned to `>=3.12,<3.13`; dependencies are managed by `uv`
with an explicit `pytorch-cu121` index for torch/vision/audio.

## Common commands

```bash
# Install (local dev / server)
uv sync --dev
uv sync --no-dev

# Unified CLI
uv run csiro-biomass --help

# Data prep (expects csiro-biomass.zip at repo root or data/raw/)
uv run csiro-biomass prepare-data --zip-path csiro-biomass.zip --extract-images

# Supervised training — prefer the wrapper for ModelScope DINOv3 variants
bash scripts/train_modelscope_dinov3.sh 896        # or 1024
uv run csiro-biomass train-supervised --config configs/server/<name>.yaml

# Pseudo-label online training / ensemble inference
uv run csiro-biomass train-pseudo --config configs/pseudo-online.yaml
uv run csiro-biomass infer       --config configs/infer-ensemble.yaml

# OOF aggregation / teacher selection (args after `oof` are passed through)
uv run csiro-biomass oof aggregate --experiment-root artifacts/server/<exp> \
  --train-manifest data/processed/csiro-biomass/metadata/train_wide.parquet
uv run csiro-biomass oof select --experiment-root <a> --experiment-root <b> \
  --output-dir artifacts/server/teacher-selection

# Tests / lint
uv run pytest                          # testpaths=tests/
uv run pytest tests/test_oof.py::<name>
uv run ruff check .                    # line-length 100, select E,F,I,UP,B
```

Hugging Face mirror (weights): `export HF_ENDPOINT=https://hf-mirror.com`. The
ModelScope DINOv3 script requires a local snapshot at
`artifacts/pretrained/modelscope/facebook/dinov3-vitl16-pretrain-lvd1689m/` with
`config.json`, `model.safetensors`, `preprocessor_config.json` (override via
`MODEL_SCOPE_DINOV3_DIR`).

## Architecture

All subcommands go through `src/csiro_biomass/cli/main.py`, which lazily imports one
of five pipelines:

- `data.prepare.run_prepare_data` — unzips the competition dump, builds
  `train_wide.parquet` + fold splits + submission templates under
  `data/processed/csiro-biomass/metadata/`.
- `training.supervised.run_training` — supervised baseline. Builds a
  `DualStreamBiomassModel` (two views → shared ViT backbone → linear proj + cross
  attention + MLP trunk → regression head), assembles `CSIROBiomassDataset`
  DataLoaders, runs the `training.engine` loop, writes per-run
  `valid_predictions.parquet` + `summary.json` + checkpoints into an experiment dir,
  and calls `training.oof.aggregate_experiment_root` at the end.
- `training.pseudo.run_pseudo_training` — online pseudo-label training that reuses
  the supervised dataloader builders and can consume test metadata/predictions as
  pseudo manifests.
- `inference.predict.run_inference` — ensemble inference driven by YAML.
- `training.oof.main` — `aggregate` and `select` subcommands. Args after the top-level
  `oof` token are forwarded verbatim via `sys.argv` rewriting.

Key modules:

- `models/backbone.py` — backbone factory (`timm`, HF `transformers`, ModelScope
  snapshots); exposes `feature_dim` and `data_config` (mean/std/interpolation)
  consumed by the training pipeline to derive data transforms when the YAML omits
  them.
- `models/dual_stream.py` — defines two `target_head_mode`s: `FIVE_HEAD` (one output
  per target in `TARGET_COLUMNS`) and `THREE_HEAD_CONSTRAINED` (three base targets
  in `BASE_TARGET_COLUMNS`, derived totals enforced). The mainline DINOv3 configs
  now use the constrained variant.
- `training/losses.py` — `WeightedBiomassLoss` (per-target weighting used in both
  training and metric reporting).
- `training/metrics.py` + `training/oof.py` — weighted R² is the canonical metric;
  OOF aggregation scans `experiment_root/**/valid_predictions.parquet`, merges with
  the train manifest, and writes aggregate metrics + per-run summaries.
- `utils/distributed.py` — `init_distributed` / `destroy_distributed`; torchrun
  launches are the default for server configs.
- `utils/postprocess.py` — third-place calibration helpers shared with
  `scripts/calibrate_third_place_postprocess.py`.

## Configs

- `configs/supervised-local-*.yaml` — local CPU/GPU debug.
- `configs/server/supervised-*.yaml` — the sweep grid (DINOv2/DINOv3/SigLIP, 384–1024).
  Filename encodes backbone + input size + source (`timm` vs `modelscope`).
- `configs/pseudo-online.yaml`, `configs/infer-ensemble.yaml` — pseudo-label and
  ensemble stages; these reference artifacts produced by earlier supervised runs.

YAMLs are loaded with `utils.config.load_yaml_config`. Data-transform fields
(`mean`/`std`/`interpolation`) are optional and inherit from the backbone's
`data_config` when absent.

## Repository rules

- `uv` is the only dependency manager; do not commit `uv.lock` edits unrelated to a
  dependency change (per `README.md`).
- `data/`, `artifacts/`, `outputs/` are generated and gitignored.
- Commit messages must be prefixed with `feat:`, `fix:`, `refactor:`, `style:`,
  `chore:`, or `docs:` (`AGENTS.md`). Prefer the smallest matching type; use `docs:`
  for documentation-only changes.
- Server deploys pull from GitHub; treat `main` as the source of truth for any
  server-side training run.
