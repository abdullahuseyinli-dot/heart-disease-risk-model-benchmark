from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from heartshift.config import ConfigurationError, load_yaml
from heartshift.contracts import (
    ContractError,
    bind_record_hash,
    canonical_json_bytes,
    load_json,
    record_sha256,
    validate_contract_repository,
    validate_record_bindings,
    validate_record_hash,
)
from heartshift.immutable import (
    ImmutableWriteError,
    preflight_create_only,
    publish_staged_create_only,
    write_bytes_create_only,
    write_json_create_only,
    write_parquet_create_only,
)
from heartshift.registry import load_method_registry

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repository_contracts_bind_real_evidence() -> None:
    paths = validate_contract_repository(REPO_ROOT)
    assert len(paths) == 9
    assert any(path.name == "heart_outer_v5_report_v1.json" for path in paths)


def test_canonical_record_hash_is_order_independent_and_finite() -> None:
    first = bind_record_hash({"b": [2, 1], "a": "value"})
    second = bind_record_hash({"a": "value", "b": [2, 1]})
    assert first["record_sha256"] == second["record_sha256"]
    validate_record_hash(first)
    assert canonical_json_bytes(first).startswith(b'{"a":"value"')
    changed = dict(first)
    changed["a"] = "changed"
    with pytest.raises(ContractError, match="self-hash mismatch"):
        validate_record_hash(changed)
    with pytest.raises(ContractError, match="non-finite"):
        canonical_json_bytes({"value": float("nan")})


def test_strict_json_rejects_duplicate_keys_and_constants(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate"):
        load_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ContractError, match="non-finite"):
        load_json(nonfinite)


def test_binding_validation_detects_changed_file(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("original\n", encoding="utf-8")
    payload = {
        "record_kind": "hardware_environment",
        "bound_run_manifests": [
            {
                "path": "evidence.txt",
                "sha256": record_sha256({"wrong": True}),
                "size_bytes": evidence.stat().st_size,
            }
        ],
    }
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        validate_record_bindings(tmp_path, payload)


def test_strict_yaml_rejects_ambiguity_and_key_drift(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("value: 1\nvalue: 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate YAML key"):
        load_yaml(duplicate)
    alias = tmp_path / "alias.yaml"
    alias.write_text("first: &shared [1]\nsecond: *shared\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="aliases are prohibited"):
        load_yaml(alias)
    valid = tmp_path / "valid.yaml"
    valid.write_text("required: true\nunexpected: false\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unexpected keys"):
        load_yaml(valid, required_keys={"required"}, allowed_keys={"required"})
    with pytest.raises(ConfigurationError, match="missing required"):
        load_yaml(valid, required_keys={"absent"})


def test_typed_method_registry_has_unique_provenance_bound_entries() -> None:
    registry = load_method_registry(REPO_ROOT)
    methods = registry.by_id()
    assert len(methods) == 18
    assert methods["tabpfn_v2_v3"].provenance.license == "Prior-Labs-License-1.2"
    assert methods["shiftguard"].evidence_eligibility == "development_only"
    assert methods["distpfn"].provenance.implementation_fidelity == "not_implemented"


def test_create_only_writers_never_replace_existing_evidence(tmp_path: Path) -> None:
    binary = tmp_path / "nested" / "evidence.bin"
    assert write_bytes_create_only(binary, b"evidence") == binary
    with pytest.raises(ImmutableWriteError, match="refusing to overwrite"):
        write_bytes_create_only(binary, b"replacement")
    assert binary.read_bytes() == b"evidence"

    record = tmp_path / "record.json"
    write_json_create_only(record, {"status": "complete"})
    assert json.loads(record.read_text(encoding="utf-8"))["status"] == "complete"

    parquet = tmp_path / "table.parquet"
    write_parquet_create_only(parquet, pd.DataFrame({"sample_id": ["a", "b"]}))
    assert pd.read_parquet(parquet)["sample_id"].tolist() == ["a", "b"]


def test_create_only_preflight_and_publication_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    with pytest.raises(ImmutableWriteError, match="duplicate"):
        preflight_create_only((target, target))
    target.write_bytes(b"old")
    with pytest.raises(ImmutableWriteError, match="refusing to overwrite"):
        preflight_create_only((target,))

    staged = tmp_path / "staged.bin"
    staged.write_bytes(b"new")
    with pytest.raises(ImmutableWriteError):
        publish_staged_create_only(staged, target)
    assert staged.read_bytes() == b"new"
    assert target.read_bytes() == b"old"

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="share a directory"):
        publish_staged_create_only(staged, other / "target.bin")
