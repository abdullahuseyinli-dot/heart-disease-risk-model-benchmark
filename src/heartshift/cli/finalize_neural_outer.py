"""Finalize frozen neural prediction shards after a disclosed aggregation failure."""

from __future__ import annotations

import argparse
from pathlib import Path

from heartshift.config import load_yaml
from heartshift.research.gates import verify_frozen_candidate
from heartshift.research.neural_recovery import finalize_fixed_neural_shards

FINALIZATION_CONFIRMATION = "FINALIZE_LOCKED_SHARDS_ONCE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirmation", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.confirmation != FINALIZATION_CONFIRMATION:
        raise SystemExit(
            "Locked shard finalization refused. Pass --confirmation "
            "FINALIZE_LOCKED_SHARDS_ONCE only for the frozen recovery configuration."
        )
    repo_root = args.repo_root.resolve()
    config = load_yaml(args.config.resolve())
    lock_path = repo_root / str(config["freeze_lock"])
    verify_frozen_candidate(repo_root, lock_path)
    run_dir = repo_root / "artifacts" / "runs" / str(config["locked_run_name"])
    outputs = finalize_fixed_neural_shards(repo_root, config, run_dir)
    print(f"run_dir: {run_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
