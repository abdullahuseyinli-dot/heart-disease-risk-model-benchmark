"""Manifest-driven, create-only acquisition and verification of public data."""

from __future__ import annotations

import hashlib
import os
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from heartshift.contracts import bind_record_hash, validate_contract_file, validate_document
from heartshift.immutable import (
    preflight_create_only,
    publish_staged_create_only,
    write_json_create_only,
)


class AcquisitionError(RuntimeError):
    """Raised when a dataset cannot be verified or acquired safely."""


@dataclass(frozen=True)
class ArtifactVerification:
    artifact_id: str
    path: str
    status: str
    size_bytes: int | None
    sha256: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_path(repo_root: Path, relative: str) -> Path:
    root = repo_root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise AcquisitionError(f"artifact path escapes repository: {relative}")
    return candidate


def _artifact_fields(value: object, *, index: int) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise AcquisitionError(f"artifacts[{index}] is not a string-keyed mapping")
    return {str(key): child for key, child in value.items()}


def _manifest_artifacts(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = [
        _artifact_fields(value, index=index) for index, value in enumerate(manifest["artifacts"])
    ]
    identifiers = [str(artifact["artifact_id"]) for artifact in artifacts]
    storage_paths = [str(artifact["storage_path"]) for artifact in artifacts]
    if len(identifiers) != len(set(identifiers)):
        raise AcquisitionError("dataset manifest contains duplicate artifact_id values")
    if len(storage_paths) != len(set(storage_paths)):
        raise AcquisitionError("dataset manifest contains duplicate storage_path values")
    return artifacts


def verify_dataset_manifest(
    repo_root: Path,
    manifest_path: Path,
    *,
    require_present: bool = True,
) -> list[ArtifactVerification]:
    """Verify all bytes named by a versioned dataset manifest."""
    manifest = validate_contract_file(repo_root, manifest_path)
    results: list[ArtifactVerification] = []
    errors: list[str] = []
    for artifact in _manifest_artifacts(manifest):
        relative = str(artifact["storage_path"])
        path = _repository_path(repo_root, relative)
        if not path.is_file():
            results.append(
                ArtifactVerification(
                    artifact_id=str(artifact["artifact_id"]),
                    path=relative,
                    status="missing",
                    size_bytes=None,
                    sha256=None,
                )
            )
            if require_present:
                errors.append(f"missing {relative}")
            continue
        size = path.stat().st_size
        digest = _sha256(path)
        expected_size = int(artifact["expected_size_bytes"])
        expected_digest = str(artifact["expected_sha256"])
        status = "verified" if size == expected_size and digest == expected_digest else "mismatch"
        results.append(
            ArtifactVerification(
                artifact_id=str(artifact["artifact_id"]),
                path=relative,
                status=status,
                size_bytes=size,
                sha256=digest,
            )
        )
        if status != "verified":
            errors.append(
                f"mismatch {relative}: bytes={size}/{expected_size}, "
                f"sha256={digest}/{expected_digest}"
            )
    if errors:
        raise AcquisitionError("dataset verification failed: " + "; ".join(errors))
    return results


def _download_to_staging(
    url: str,
    staged: Path,
    *,
    timeout_seconds: float,
    expected_size_bytes: int,
) -> None:
    if not url.startswith("https://"):
        raise AcquisitionError(f"only HTTPS acquisition is permitted: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "HeartShift/0.1 data-acquisition"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if getattr(response, "status", 200) != 200:
            raise AcquisitionError(f"download returned HTTP {response.status}: {url}")
        final_url = response.geturl()
        if not final_url.startswith("https://"):
            raise AcquisitionError(f"download redirected outside HTTPS: {final_url}")
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise AcquisitionError(
                    f"download returned an invalid Content-Length: {content_length!r}"
                ) from exc
            if declared_size > expected_size_bytes:
                raise AcquisitionError(
                    f"download exceeds manifest size: {declared_size} > {expected_size_bytes}"
                )
        with staged.open("xb") as output:
            remaining = expected_size_bytes + 1
            while remaining > 0:
                block = response.read(min(1024 * 1024, remaining))
                if not block:
                    break
                output.write(block)
                remaining -= len(block)
            if output.tell() > expected_size_bytes:
                raise AcquisitionError(
                    f"download exceeds manifest size: more than {expected_size_bytes} bytes"
                )
            output.flush()
            os.fsync(output.fileno())


def _receipt(
    *,
    repo_root: Path,
    manifest_path: Path,
    dataset_id: str,
    results: list[ArtifactVerification],
    status: str,
    network_explicitly_enabled: bool,
    failure: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    created = datetime.now(UTC)
    try:
        manifest_relative = manifest_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise AcquisitionError("dataset manifest must be stored inside the repository") from exc
    return bind_record_hash(
        {
            "schema_version": "1.0.0",
            "record_kind": "acquisition_receipt",
            "receipt_id": f"acquire-{dataset_id}-{created.strftime('%Y%m%dT%H%M%SZ').lower()}",
            "dataset_id": dataset_id,
            "created_at_utc": created.isoformat().replace("+00:00", "Z"),
            "manifest": {
                "path": manifest_relative,
                "sha256": _sha256(manifest_path),
                "size_bytes": manifest_path.stat().st_size,
            },
            "artifacts": [
                {
                    "artifact_id": result.artifact_id,
                    "path": result.path,
                    "status": result.status,
                    "sha256": result.sha256,
                    "size_bytes": result.size_bytes,
                }
                for result in results
            ],
            "network_explicitly_enabled": network_explicitly_enabled,
            "failure": dict(failure) if failure is not None else None,
            "status": status,
        }
    )


def acquire_dataset_manifest(
    repo_root: Path,
    manifest_path: Path,
    *,
    allow_network: bool,
    receipt_path: Path | None = None,
    timeout_seconds: float = 120.0,
) -> list[ArtifactVerification]:
    """Acquire absent public artifacts without replacing any existing path.

    Existing artifacts must match their manifest exactly. All absent destinations
    are preflighted before the first request. Partial downloads remain named as
    ``.partial.<uuid>`` if acquisition fails, preserving failure evidence.
    """
    manifest = validate_contract_file(repo_root, manifest_path)
    artifacts = _manifest_artifacts(manifest)
    existing = verify_dataset_manifest(repo_root, manifest_path, require_present=False)
    by_id = {result.artifact_id: result for result in existing}
    mismatches = [result.path for result in existing if result.status == "mismatch"]
    if mismatches:
        raise AcquisitionError(f"existing artifacts do not match the manifest: {mismatches}")

    absent = [
        artifact
        for artifact in artifacts
        if by_id[str(artifact["artifact_id"])].status == "missing"
    ]
    if not absent:
        results = verify_dataset_manifest(repo_root, manifest_path)
        if receipt_path is not None:
            preflight_create_only((receipt_path,))
            receipt = _receipt(
                repo_root=repo_root,
                manifest_path=manifest_path,
                dataset_id=str(manifest["dataset_id"]),
                results=results,
                status="complete",
                network_explicitly_enabled=allow_network,
            )
            validate_document(receipt, repo_root / "configs/schema/acquisition_receipt.schema.json")
            write_json_create_only(receipt_path, receipt)
        return results
    destinations = [
        _repository_path(repo_root, str(artifact["storage_path"])) for artifact in absent
    ]
    publication_targets = [*destinations]
    if receipt_path is not None:
        publication_targets.append(receipt_path)
    preflight_create_only(publication_targets)
    if not allow_network or os.environ.get("HEARTSHIFT_OFFLINE") == "1":
        raise AcquisitionError("network acquisition is disabled; pass --allow-network explicitly")
    staged: Path | None = None
    try:
        for artifact, destination in zip(absent, destinations, strict=True):
            acquisition = str(artifact["acquisition"])
            if acquisition != "download":
                raise AcquisitionError(
                    f"{artifact['artifact_id']} requires {acquisition}; "
                    "automatic download is prohibited"
                )
            source_url = artifact.get("source_url")
            if not isinstance(source_url, str):
                raise AcquisitionError(f"{artifact['artifact_id']} has no public source URL")
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged = destination.with_name(f".{destination.name}.partial.{uuid.uuid4().hex}")
            _download_to_staging(
                source_url,
                staged,
                timeout_seconds=timeout_seconds,
                expected_size_bytes=int(artifact["expected_size_bytes"]),
            )
            size = staged.stat().st_size
            digest = _sha256(staged)
            if size != int(artifact["expected_size_bytes"]) or digest != str(
                artifact["expected_sha256"]
            ):
                raise AcquisitionError(
                    f"download mismatch retained at {staged}: bytes={size}, sha256={digest}"
                )
            publish_staged_create_only(staged, destination)
            staged = None
            destination.chmod(0o444)
    except Exception as exc:
        if receipt_path is not None:
            results = verify_dataset_manifest(repo_root, manifest_path, require_present=False)
            partial_relative: str | None = None
            if staged is not None and staged.exists():
                partial_relative = staged.resolve().relative_to(repo_root.resolve()).as_posix()
            receipt = _receipt(
                repo_root=repo_root,
                manifest_path=manifest_path,
                dataset_id=str(manifest["dataset_id"]),
                results=results,
                status="failed",
                network_explicitly_enabled=allow_network,
                failure={
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "preserved_partial_path": partial_relative,
                },
            )
            validate_document(receipt, repo_root / "configs/schema/acquisition_receipt.schema.json")
            write_json_create_only(receipt_path, receipt)
        raise

    results = verify_dataset_manifest(repo_root, manifest_path)
    if receipt_path is not None:
        receipt = _receipt(
            repo_root=repo_root,
            manifest_path=manifest_path,
            dataset_id=str(manifest["dataset_id"]),
            results=results,
            status="complete",
            network_explicitly_enabled=allow_network,
        )
        validate_document(receipt, repo_root / "configs/schema/acquisition_receipt.schema.json")
        write_json_create_only(receipt_path, receipt)
    return results


def verification_as_dict(value: ArtifactVerification) -> Mapping[str, object]:
    """Return a stable JSON-ready representation for the CLI."""
    return {
        "artifact_id": value.artifact_id,
        "path": value.path,
        "status": value.status,
        "size_bytes": value.size_bytes,
        "sha256": value.sha256,
    }
