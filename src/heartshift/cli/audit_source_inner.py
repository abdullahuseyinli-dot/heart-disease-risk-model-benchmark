"""Create or validate an exact post-run source-only artifact-tree audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from heartshift.research.artifact_audit import (
    validate_source_inner_artifact_audit,
    write_source_inner_artifact_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    if args.validate_only:
        result = validate_source_inner_artifact_audit(run_dir)
        print(f"source_inner_audit: {result}")
    else:
        output = write_source_inner_artifact_audit(run_dir)
        print(f"source_inner_audit: {output}")


if __name__ == "__main__":
    main()
