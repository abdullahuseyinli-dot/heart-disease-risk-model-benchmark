"""Hash the exact source bytes associated with a completed experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from heartshift.research.provenance import write_post_run_source_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--file", type=Path, action="append", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    output = write_post_run_source_snapshot(repo_root, run_dir, list(args.file))
    print(f"source_snapshot: {output}")


if __name__ == "__main__":
    main()
