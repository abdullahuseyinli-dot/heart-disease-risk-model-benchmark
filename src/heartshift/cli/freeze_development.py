"""Freeze exact development bytes before descriptive outer inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from heartshift.config import load_yaml
from heartshift.research.development_freeze import freeze_development_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    output = args.output if args.output.is_absolute() else repo_root / args.output
    frozen = freeze_development_candidate(repo_root, load_yaml(config_path), output)
    print(f"development_freeze: {frozen}")


if __name__ == "__main__":
    main()
