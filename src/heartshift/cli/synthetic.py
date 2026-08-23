"""Run controlled PS-MaskDRO mechanism and failure-mode experiments."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from heartshift.config import config_hash, load_yaml
from heartshift.research.synthetic_experiment import run_synthetic_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    config = load_yaml(args.config.resolve())
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"synthetic-{timestamp}-{config_hash(config)[:12]}"
    run_dir = repo_root / "artifacts" / "runs" / run_name
    outputs = run_synthetic_experiment(repo_root, config, run_dir)
    print(f"run_dir: {run_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
