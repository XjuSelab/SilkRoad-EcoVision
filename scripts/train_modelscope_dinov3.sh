#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/train_modelscope_dinov3.sh <896|1024> [extra args...]"
  exit 1
fi

variant="$1"
shift

snapshot_dir="${MODEL_SCOPE_DINOV3_DIR:-artifacts/pretrained/modelscope/facebook/dinov3-vitl16-pretrain-lvd1689m}"

for required in config.json model.safetensors preprocessor_config.json; do
  if [[ ! -f "${snapshot_dir}/${required}" ]]; then
    echo "missing required ModelScope snapshot file: ${snapshot_dir}/${required}" >&2
    exit 1
  fi
done

case "${variant}" in
  896)
    config_path="configs/server/supervised-dinov3-vitl-896-modelscope.yaml"
    ;;
  1024)
    config_path="configs/server/supervised-dinov3-vitl-1024-modelscope.yaml"
    ;;
  *)
    echo "unsupported variant: ${variant}; expected 896 or 1024" >&2
    exit 1
    ;;
esac

exec uv run csiro-biomass train-supervised --config "${config_path}" "$@"
