"""Build the statistical report for the locked readmission shift experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from heartshift.config import load_yaml
from heartshift.reporting.readmission_report import build_readmission_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = build_readmission_report(
        args.repo_root.resolve(),
        load_yaml(args.config.resolve()),
        args.output_dir.resolve(),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
