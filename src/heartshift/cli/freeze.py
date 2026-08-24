"""Validate pre-outer gates and freeze the HeartShift candidate once."""

from __future__ import annotations

import argparse
from pathlib import Path

from heartshift.config import load_yaml
from heartshift.research.gates import freeze_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    config = load_yaml(args.config.resolve())
    output = args.output if args.output.is_absolute() else repo_root / args.output
    lock = freeze_candidate(repo_root, config, output.resolve())
    print(f"Candidate frozen: {output.resolve()}")
    print(f"Method tree SHA-256: {lock['method_tree_sha256']}")


if __name__ == "__main__":
    main()
