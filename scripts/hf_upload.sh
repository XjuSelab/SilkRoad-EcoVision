#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DATASET_REPO="XJU-SeLab/csiro-biomass-private"
DEFAULT_MODEL_REPO="XJU-SeLab/csiro-biomass-server-models"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/hf_upload.sh --target dataset --local-path ./src_dir --path-in-repo some/path
  bash scripts/hf_upload.sh --target model --local-path ./artifacts/server/dinov3-vitl-896 --path-in-repo server/dinov3-vitl-896

Options:
  --target <dataset|model>       Choose the default repo mapping. Default: dataset
  --repo-id <repo_id>            Override the default repo id
  --repo-type <dataset|model>    Override the inferred repo type
  --local-path <path>            Local file or directory to upload
  --path-in-repo <path>          Destination path inside the Hub repo
  --revision <rev>               Optional branch / revision
  --commit-message <msg>         Optional commit message
  --token <token>                Optional explicit token
  -h, --help                     Show this help message

Defaults:
  dataset -> XJU-SeLab/csiro-biomass-private
  model   -> XJU-SeLab/csiro-biomass-server-models

Notes:
  - The script prefers a global `hf` CLI. If unavailable, it falls back to `uv run hf`.
  - The script enables Xet high-performance mode by default with HF_XET_HIGH_PERFORMANCE=1.
EOF
}

target="dataset"
repo_id=""
repo_type=""
local_path=""
path_in_repo=""
revision=""
commit_message=""
token=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="${2:-}"
      shift 2
      ;;
    --repo-id)
      repo_id="${2:-}"
      shift 2
      ;;
    --repo-type)
      repo_type="${2:-}"
      shift 2
      ;;
    --local-path)
      local_path="${2:-}"
      shift 2
      ;;
    --path-in-repo)
      path_in_repo="${2:-}"
      shift 2
      ;;
    --revision)
      revision="${2:-}"
      shift 2
      ;;
    --commit-message)
      commit_message="${2:-}"
      shift 2
      ;;
    --token)
      token="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$target" in
  dataset|model) ;;
  *)
    echo "Unsupported --target: $target" >&2
    exit 1
    ;;
esac

case "${repo_type:-}" in
  ""|dataset|model) ;;
  *)
    echo "Unsupported --repo-type: $repo_type" >&2
    exit 1
    ;;
esac

if [[ -z "$local_path" ]]; then
  echo "--local-path is required" >&2
  exit 1
fi

if [[ ! -e "$local_path" ]]; then
  echo "Local path does not exist: $local_path" >&2
  exit 1
fi

if [[ -z "$path_in_repo" ]]; then
  echo "--path-in-repo is required" >&2
  exit 1
fi

while [[ "${path_in_repo#/}" != "$path_in_repo" ]]; do
  path_in_repo="${path_in_repo#/}"
done

if [[ -z "$path_in_repo" ]]; then
  echo "--path-in-repo must not resolve to repo root" >&2
  exit 1
fi

if [[ -z "$repo_id" ]]; then
  if [[ "$target" == "dataset" ]]; then
    repo_id="$DEFAULT_DATASET_REPO"
  else
    repo_id="$DEFAULT_MODEL_REPO"
  fi
fi

if [[ -z "$repo_type" ]]; then
  repo_type="$target"
fi

if command -v hf >/dev/null 2>&1; then
  cli=(hf)
elif command -v uv >/dev/null 2>&1; then
  cli=(uv run hf)
else
  echo "Neither \`hf\` nor \`uv\` is available. Install huggingface_hub CLI first." >&2
  exit 1
fi

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

cmd=("${cli[@]}" upload "$repo_id" "$local_path" "$path_in_repo" --repo-type "$repo_type")

if [[ -n "$revision" ]]; then
  cmd+=(--revision "$revision")
fi

if [[ -n "$commit_message" ]]; then
  cmd+=(--commit-message "$commit_message")
fi

if [[ -n "$token" ]]; then
  cmd+=(--token "$token")
fi

echo "Repo: $repo_id"
echo "Repo type: $repo_type"
echo "Local path: $local_path"
echo "Path in repo: $path_in_repo"
echo "CLI: ${cli[*]}"
echo "HF_XET_HIGH_PERFORMANCE=$HF_XET_HIGH_PERFORMANCE"
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
