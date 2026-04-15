#!/usr/bin/env python3
"""Download only files that match a repo path prefix."""

from __future__ import annotations

import argparse
import os
from pathlib import PurePosixPath

from huggingface_hub import HfApi, hf_hub_download

DEFAULT_DATASET_REPO = "XJU-SeLab/csiro-biomass-private"
DEFAULT_MODEL_REPO = "XJU-SeLab/csiro-biomass-server-models"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a Hugging Face repo subdirectory by prefix.")
    parser.add_argument("--target", choices=["dataset", "model"], default="dataset")
    parser.add_argument("--repo-id", help="Override the default repo id for the selected target.")
    parser.add_argument("--repo-type", choices=["dataset", "model"], help="Override the inferred repo type.")
    parser.add_argument("--prefix", required=True, help="Repo-relative directory prefix to download.")
    parser.add_argument("--local-dir", default=".", help="Local directory to store downloaded files.")
    parser.add_argument("--revision", help="Optional branch / revision / commit hash.")
    parser.add_argument(
        "--endpoint",
        help="Hub endpoint. Defaults to HF_ENDPOINT or https://hf-mirror.com.",
    )
    return parser


def resolve_repo_id(target: str, repo_id: str | None) -> str:
    if repo_id:
        return repo_id
    if target == "dataset":
        return DEFAULT_DATASET_REPO
    return DEFAULT_MODEL_REPO


def normalize_prefix(prefix: str) -> str:
    normalized = str(PurePosixPath(prefix.lstrip("/")))
    if normalized in {"", "."}:
        raise SystemExit("--prefix must point to a repo subdirectory")
    return normalized.rstrip("/") + "/"


def main() -> None:
    args = build_parser().parse_args()
    repo_id = resolve_repo_id(args.target, args.repo_id)
    repo_type = args.repo_type or args.target
    endpoint = args.endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT
    prefix = normalize_prefix(args.prefix)

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    api = HfApi(endpoint=endpoint)
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=args.revision)
    target_files = [path for path in files if path.startswith(prefix)]

    if not target_files:
        raise SystemExit(f"0 files matched prefix: {prefix}")

    print(f"Repo: {repo_id}")
    print(f"Repo type: {repo_type}")
    print(f"Endpoint: {endpoint}")
    print(f"Prefix: {prefix}")
    print(f"Local dir: {args.local_dir}")
    print(f"Will download {len(target_files)} files")

    for path in target_files:
        print(f"Downloading: {path}")
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=path,
            revision=args.revision,
            endpoint=endpoint,
            local_dir=args.local_dir,
        )

    print("Done")


if __name__ == "__main__":
    main()
