#!/usr/bin/env python3
"""Download an entire Hugging Face repo snapshot with sensible defaults."""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

DEFAULT_DATASET_REPO = "XJU-SeLab/csiro-biomass-private"
DEFAULT_MODEL_REPO = "XJU-SeLab/csiro-biomass-server-models"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a Hugging Face repo snapshot.")
    parser.add_argument("--target", choices=["dataset", "model"], default="dataset")
    parser.add_argument("--repo-id", help="Override the default repo id for the selected target.")
    parser.add_argument("--repo-type", choices=["dataset", "model"], help="Override the inferred repo type.")
    parser.add_argument("--local-dir", required=True, help="Local directory to store the downloaded snapshot.")
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


def main() -> None:
    args = build_parser().parse_args()
    repo_id = resolve_repo_id(args.target, args.repo_id)
    repo_type = args.repo_type or args.target
    endpoint = args.endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    print(f"Repo: {repo_id}")
    print(f"Repo type: {repo_type}")
    print(f"Endpoint: {endpoint}")
    print(f"Local dir: {args.local_dir}")
    if args.revision:
        print(f"Revision: {args.revision}")

    download_path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=args.local_dir,
        revision=args.revision,
        endpoint=endpoint,
    )
    print(f"Done: {download_path}")


if __name__ == "__main__":
    main()
