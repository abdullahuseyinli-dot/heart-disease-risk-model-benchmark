"""Prepare an authorized multi-hospital table under a frozen data contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from heartshift.config import load_yaml
from heartshift.data.external import ExternalDatasetContract, prepare_external_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    contract = ExternalDatasetContract.from_mapping(load_yaml(config_path))
    outputs = prepare_external_dataset(repo_root, contract, output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
