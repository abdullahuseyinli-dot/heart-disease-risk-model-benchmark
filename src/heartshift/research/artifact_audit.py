"""Post-run exact-tree audits for source-only research evidence."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, cast

from heartshift.data.uci import sha256_file
from heartshift.research.gates import validate_source_only_run

AUDIT_NAME = "evidence_audit.json"
AUDIT_STATUS = "complete_source_only_inner_evidence_post_run_audit"


def _artifact_files(run_dir: Path) -> dict[str, Path]:
    return {
        path.relative_to(run_dir).as_posix(): path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != AUDIT_NAME
    }


def _resolve_audited_path(run_dir: Path, relative: str) -> Path:
    posix_path = PurePosixPath(relative)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise AssertionError(f"Unsafe path in source evidence audit: {relative}")
    path = (run_dir / Path(*posix_path.parts)).resolve()
    if not path.is_relative_to(run_dir):
        raise AssertionError(f"Source evidence path escapes its run: {relative}")
    return path


def write_source_inner_artifact_audit(run_dir: Path) -> Path:
    """Validate source isolation, then hash every existing non-audit artifact."""
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Source-only run directory is missing: {run_dir}")
    output = run_dir / AUDIT_NAME
    if output.exists():
        raise FileExistsError(f"Source-only evidence audit already exists: {output}")
    source_validation = validate_source_only_run(run_dir)
    files = _artifact_files(run_dir)
    hashes = {relative: sha256_file(path) for relative, path in files.items()}
    output.write_text(
        json.dumps(
            {
                "status": AUDIT_STATUS,
                "audit_timing": "post_run_before_repository_commit",
                "artifact_count": len(hashes),
                "source_only_validation": source_validation,
                "sha256": hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def validate_source_inner_artifact_audit(run_dir: Path) -> dict[str, Any]:
    """Require the current source-only artifact tree to match its exact audit."""
    run_dir = run_dir.resolve()
    audit_path = run_dir / AUDIT_NAME
    if not audit_path.is_file():
        raise FileNotFoundError(f"Source-only evidence audit is missing: {audit_path}")
    audit = cast(dict[str, Any], json.loads(audit_path.read_text(encoding="utf-8")))
    if audit.get("status") != AUDIT_STATUS:
        raise AssertionError("Unexpected source-only evidence audit status")
    expected = {
        str(relative): str(expected_hash)
        for relative, expected_hash in cast(dict[str, str], audit["sha256"]).items()
    }
    current = _artifact_files(run_dir)
    if set(current) != set(expected):
        missing = sorted(set(expected) - set(current))
        extra = sorted(set(current) - set(expected))
        raise AssertionError(f"Source-only artifact tree changed; missing={missing}, extra={extra}")
    for relative, expected_hash in expected.items():
        path = _resolve_audited_path(run_dir, relative)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise AssertionError(f"Source-only artifact hash failed: {path}")
    if int(audit.get("artifact_count", -1)) != len(expected):
        raise AssertionError("Source-only evidence artifact count is inconsistent")
    source_validation = validate_source_only_run(run_dir)
    return {
        "status": AUDIT_STATUS,
        "artifact_count": len(expected),
        "prediction_rows": int(source_validation["prediction_rows"]),
        "audit_sha256": sha256_file(audit_path),
    }
