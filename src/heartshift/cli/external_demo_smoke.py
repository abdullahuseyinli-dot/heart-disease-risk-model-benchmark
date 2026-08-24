"""Run models end to end on the public eICU demo (smoke test only)."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from heartshift.config import config_hash, load_yaml
from heartshift.evaluation.external_smoke import run_external_demo_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = load_yaml(config_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"external-demo-smoke-{timestamp}-{config_hash(config)[:12]}"
    run_dir = repo_root / "artifacts" / "runs" / run_name
    outputs = run_external_demo_smoke(repo_root, config, run_dir)
    print(f"run_dir: {run_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
