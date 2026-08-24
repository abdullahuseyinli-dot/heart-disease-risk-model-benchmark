"""Run endpoint-isolated neural outer evaluation on consumed development data."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from heartshift.config import config_hash, load_yaml
from heartshift.data.uci import sha256_file
from heartshift.evaluation.neural_outer import run_neural_outer


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
    status = str(config.get("status"))
    if status not in {
        "consumed_uci_development_only",
        "code_smoke_not_scientific_result",
    }:
        raise SystemExit(
            "Research neural outer evaluation requires status consumed_uci_development_only"
        )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or (f"research-neural-outer-{timestamp}-{config_hash(config)[:12]}")
    run_dir = repo_root / "artifacts" / "runs" / run_name
    outputs = run_neural_outer(repo_root, config, run_dir)
    hashes = {path.name: sha256_file(path) for path in outputs.values() if path.is_file()}
    audit_path = run_dir / "evidence_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": f"complete_{status}",
                "new_confirmatory_claim_allowed": False,
                "target_endpoint_loaded_after_all_endpoint_free_prediction_shards": True,
                "individual_seed_predictions_retained": True,
                "config_sha256": config_hash(config),
                "sha256": hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["audit"] = audit_path
    print(f"run_dir: {run_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
