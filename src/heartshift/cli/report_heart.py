"""Build the statistical report from locked patient-level heart predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from heartshift.config import load_yaml
from heartshift.reporting.outer_report import build_heart_outer_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    outputs = build_heart_outer_report(
        repo_root,
        load_yaml(args.config.resolve()),
        args.output_dir.resolve(),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
