"""Run source-only tuning or a separately confirmed locked outer benchmark."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from heartshift.config import config_hash, load_yaml
from heartshift.evaluation.classical_benchmark import run_inner_benchmark, run_outer_benchmark
from heartshift.research.gates import verify_frozen_candidate

OUTER_CONFIRMATION = "RUN_LOCKED_OUTER_ONCE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("inner", "outer"), required=True)
    parser.add_argument("--inner-run-dir", type=Path)
    parser.add_argument("--confirmation")
    parser.add_argument("--run-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    config = load_yaml(args.config.resolve())
    short_hash = config_hash(config)[:12]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.phase == "outer":
        locked_run_name = config.get("locked_run_name")
        if locked_run_name is None:
            raise SystemExit("Outer configuration must name its locked_run_name")
        if args.run_name is not None and args.run_name != locked_run_name:
            raise SystemExit("Outer run name cannot override the frozen locked_run_name")
        run_name = str(locked_run_name)
    else:
        run_name = args.run_name or f"{args.phase}-{timestamp}-{short_hash}"
    run_dir = repo_root / "artifacts" / "runs" / run_name
    if args.phase == "inner":
        outputs = run_inner_benchmark(repo_root, config, run_dir)
    else:
        if args.confirmation != OUTER_CONFIRMATION:
            raise SystemExit(
                "Locked outer evaluation refused. Pass --confirmation RUN_LOCKED_OUTER_ONCE "
                "only after the candidate configuration is frozen."
            )
        if args.inner_run_dir is None:
            raise SystemExit("--inner-run-dir is required for outer evaluation")
        lock_path = config.get("freeze_lock")
        if lock_path is None:
            raise SystemExit("Outer configuration must name its freeze_lock")
        verify_frozen_candidate(repo_root, repo_root / str(lock_path))
        outputs = run_outer_benchmark(repo_root, config, args.inner_run_dir.resolve(), run_dir)
    print(f"run_dir: {run_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
