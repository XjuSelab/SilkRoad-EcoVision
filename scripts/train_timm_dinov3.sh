#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/train_timm_dinov3.sh <896|1024|hplus-896> [extra args...]"
  exit 1
fi

variant="$1"
shift

case "${variant}" in
  896|vitl-896)
    config_path="configs/server/supervised-dinov3-vitl-896-timm.yaml"
    ;;
  1024|vitl-1024)
    config_path="configs/server/supervised-dinov3-vitl-1024-timm.yaml"
    ;;
  hplus-896|vithplus-896)
    config_path="configs/server/supervised-dinov3-vithplus-896-timm.yaml"
    ;;
  *)
    echo "unsupported variant: ${variant}; expected 896, 1024, or hplus-896" >&2
    exit 1
    ;;
esac

exec uv run csiro-biomass train-supervised --config "${config_path}" "$@"
