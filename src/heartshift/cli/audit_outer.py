"""Independently reconstruct and hash locked outer-evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from heartshift.research.outer_evidence import (
    audit_heart_classical_outer,
    audit_heart_neural_outer,
    audit_readmission_outer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=["heart-classical", "heart-neural", "readmission"],
        required=True,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    if args.kind == "heart-classical":
        outputs = audit_heart_classical_outer(
            repo_root, run_dir.resolve(), workers=args.workers, replace=args.replace
        )
    elif args.kind == "heart-neural":
        outputs = audit_heart_neural_outer(repo_root, run_dir.resolve(), replace=args.replace)
    else:
        outputs = audit_readmission_outer(
            repo_root, run_dir.resolve(), workers=args.workers, replace=args.replace
        )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
