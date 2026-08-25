"""Versioned JSON contracts for release-facing HeartShift records.

Historical experiment artifacts predate this module and remain byte-for-byte
immutable.  New manifests and release sidecars use these helpers so that
duplicate keys, non-finite values, schema drift, and self-hash mismatches fail
closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

SELF_HASH_FIELD = "record_sha256"
SCHEMA_BY_KIND = {
    "acquisition_receipt": "acquisition_receipt.schema.json",
    "candidate_tree_scan": "candidate_tree_scan.schema.json",
    "dataset_manifest": "dataset_manifest.schema.json",
    "failure_record": "failure_record.schema.json",
    "final_release_gate_report": "final_release_gate_report.schema.json",
    "freeze_record": "freeze_record.schema.json",
    "hardware_environment": "hardware_environment.schema.json",
    "method_registry": "method_registry.schema.json",
    "metric_table_contract": "metric_table.schema.json",
    "prediction_table_contract": "prediction_table.schema.json",
    "release_evidence_inventory": "release_evidence_inventory.schema.json",
    "release_gate_status": "release_gate_status.schema.json",
    "report_evidence_manifest": "report_manifest.schema.json",
    "run_manifest": "run_manifest.schema.json",
}


class ContractError(ValueError):
    """Raised when a release-facing record violates its contract."""


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON number is prohibited: {value}")


def load_json(path: Path) -> dict[str, Any]:
    """Load a strict UTF-8 JSON object."""
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError(f"invalid strict JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON contract must be an object: {path}")
    return value


def canonical_json_bytes(payload: object) -> bytes:
    """Return deterministic UTF-8 JSON bytes after checking finite numbers."""

    def check(value: Any, location: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ContractError(f"non-finite number at {location}")
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ContractError(f"non-string object key at {location}")
                check(child, f"{location}.{key}")
        elif isinstance(value, list | tuple):
            for index, child in enumerate(value):
                check(child, f"{location}[{index}]")

    check(payload, "$")
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def record_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a record while excluding its own hash field."""
    material = dict(payload)
    material.pop(SELF_HASH_FIELD, None)
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def bind_record_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with a deterministic self-hash."""
    result = dict(payload)
    result[SELF_HASH_FIELD] = record_sha256(result)
    return result


def validate_record_hash(payload: Mapping[str, Any]) -> None:
    observed = payload.get(SELF_HASH_FIELD)
    if not isinstance(observed, str) or len(observed) != 64:
        raise ContractError(f"{SELF_HASH_FIELD} must be a lowercase SHA-256 digest")
    if observed != observed.lower() or any(
        character not in "0123456789abcdef" for character in observed
    ):
        raise ContractError(f"{SELF_HASH_FIELD} is not lowercase hexadecimal")
    expected = record_sha256(payload)
    if observed != expected:
        raise ContractError(f"record self-hash mismatch: {observed} != {expected}")


def validate_schema(schema: Mapping[str, Any]) -> None:
    """Check that a document is itself a valid Draft 2020-12 schema."""
    Draft202012Validator.check_schema(dict(schema))


def validate_document(payload: Mapping[str, Any], schema_path: Path) -> None:
    """Validate a mapping against a local Draft 2020-12 schema."""
    schema = load_json(schema_path)
    validate_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(dict(payload)), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(
            f"${''.join(f'[{part!r}]' for part in error.path)}: {error.message}"
            for error in errors[:8]
        )
        raise ContractError(f"{schema_path.name} validation failed: {details}")


def schema_for_record(repo_root: Path, payload: Mapping[str, Any]) -> Path:
    kind = payload.get("record_kind")
    if not isinstance(kind, str) or kind not in SCHEMA_BY_KIND:
        raise ContractError(f"unsupported record_kind: {kind!r}")
    return repo_root / "configs" / "schema" / SCHEMA_BY_KIND[kind]


def _bound_path(repo_root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ContractError("bound evidence path must be a string")
    root = repo_root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ContractError(f"bound evidence path escapes repository: {relative}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_binding(repo_root: Path, binding: Mapping[str, Any]) -> None:
    path = _bound_path(repo_root, binding.get("path"))
    if not path.is_file():
        raise ContractError(f"bound evidence file is missing: {path}")
    expected_size = binding.get("size_bytes")
    if isinstance(expected_size, int) and path.stat().st_size != expected_size:
        raise ContractError(
            f"bound evidence size mismatch for {path}: {path.stat().st_size} != {expected_size}"
        )
    expected_hash = binding.get("sha256")
    if not isinstance(expected_hash, str) or _file_sha256(path) != expected_hash:
        raise ContractError(f"bound evidence SHA-256 mismatch: {path}")


def validate_record_bindings(repo_root: Path, payload: Mapping[str, Any]) -> None:
    """Verify repository files named by a release-facing record."""
    kind = payload.get("record_kind")
    if kind == "dataset_manifest":
        allow_absent = payload.get("status") in {"terms_gated", "protected_not_acquired"}
        for artifact in payload["artifacts"]:
            path = _bound_path(repo_root, artifact["storage_path"])
            if allow_absent and not path.exists():
                continue
            _validate_binding(
                repo_root,
                {
                    "path": artifact["storage_path"],
                    "sha256": artifact["expected_sha256"],
                    "size_bytes": artifact["expected_size_bytes"],
                },
            )
    elif kind == "report_evidence_manifest":
        for binding in (*payload["source_records"], *payload["headline_tables"]):
            _validate_binding(repo_root, binding)
        _validate_binding(repo_root, payload["validation"]["evidence"])
    elif kind == "hardware_environment":
        for binding in payload["bound_run_manifests"]:
            _validate_binding(repo_root, binding)
    elif kind == "acquisition_receipt":
        _validate_binding(repo_root, payload["manifest"])
        for artifact in payload["artifacts"]:
            if artifact["status"] == "verified":
                _validate_binding(repo_root, artifact)


def validate_contract_file(repo_root: Path, path: Path) -> dict[str, Any]:
    payload = load_json(path)
    validate_document(payload, schema_for_record(repo_root, payload))
    validate_record_hash(payload)
    return payload


def iter_contract_files(repo_root: Path) -> list[Path]:
    roots = (
        repo_root / "manifests" / "datasets",
        repo_root / "manifests" / "environment",
        repo_root / "manifests" / "evidence",
    )
    return sorted(path for root in roots if root.is_dir() for path in root.glob("*.json"))


def validate_contract_repository(
    repo_root: Path,
    *,
    verify_bindings: bool = False,
) -> list[Path]:
    """Validate every schema and contract, optionally materialized evidence bytes."""
    schema_root = repo_root / "configs" / "schema"
    schema_paths = sorted(schema_root.glob("*.schema.json"))
    if not schema_paths:
        raise ContractError("no JSON schemas were found")
    for path in schema_paths:
        validate_schema(load_json(path))
    contract_paths = iter_contract_files(repo_root)
    if not contract_paths:
        raise ContractError("no release-facing contract records were found")
    for path in contract_paths:
        payload = validate_contract_file(repo_root, path)
        if verify_bindings:
            validate_record_bindings(repo_root, payload)
    return contract_paths
