"""Prepare the independent UCI diabetes-readmission shift task."""

from __future__ import annotations

import argparse
from pathlib import Path

from heartshift.config import load_yaml
from heartshift.data.readmission import prepare_readmission_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    outputs = prepare_readmission_dataset(repo_root, load_yaml(args.config.resolve()))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
