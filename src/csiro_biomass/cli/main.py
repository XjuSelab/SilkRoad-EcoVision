"""Unified command line entrypoint."""

from __future__ import annotations

import argparse

from csiro_biomass.utils.config import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSIRO biomass reproduction toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Create wide manifests, folds, and templates")
    prepare.add_argument("--zip-path", required=True)
    prepare.add_argument("--raw-dir", default="data/raw")
    prepare.add_argument("--processed-dir", default="data/processed/csiro-biomass")
    prepare.add_argument("--n-splits", type=int, default=3)
    prepare.add_argument("--extract-images", action="store_true")

    train = subparsers.add_parser("train-supervised", help="Train the supervised baseline")
    train.add_argument("--config", required=True)

    pseudo = subparsers.add_parser("train-pseudo", help="Run pseudo-label online training")
    pseudo.add_argument("--config", required=True)

    infer = subparsers.add_parser("infer", help="Run ensemble inference")
    infer.add_argument("--config", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare-data":
        from csiro_biomass.data.prepare import run_prepare_data

        run_prepare_data(args)
    elif args.command == "train-supervised":
        from csiro_biomass.training.supervised import run_training

        run_training(load_yaml_config(args.config))
    elif args.command == "train-pseudo":
        from csiro_biomass.training.pseudo import run_pseudo_training

        run_pseudo_training(load_yaml_config(args.config))
    elif args.command == "infer":
        from csiro_biomass.inference.predict import run_inference

        run_inference(load_yaml_config(args.config))
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
