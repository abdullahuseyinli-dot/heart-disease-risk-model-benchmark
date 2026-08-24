"""Exact-candidate release scanning, attestations, and evidence inventories."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from heartshift.contracts import (
    bind_record_hash,
    canonical_json_bytes,
    load_json,
    validate_contract_file,
    validate_document,
    validate_record_hash,
)
from heartshift.immutable import write_json_create_only

COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"
LFS_OID = re.compile(rb"^oid sha256:([0-9a-f]{64})$", flags=re.MULTILINE)
LFS_SIZE = re.compile(rb"^size ([0-9]+)$", flags=re.MULTILINE)
WINDOWS_USER_PATH = re.compile(rb"[a-zA-Z]:[\\/]+Users[\\/]+[^\\/\s]+[\\/]")
GITHUB_WORKFLOW_URL = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repository>[A-Za-z0-9_.-]+)/actions/runs/(?P<run_id>[1-9][0-9]*)$"
)
POLICY_KEYS = {
    "schema_version",
    "max_git_blob_bytes",
    "prohibited_path_patterns",
    "prohibited_suffixes",
    "required_gates",
    "required_roles",
    "text_scan_exempt_prefixes",
    "text_suffixes",
}


class ReleaseGateError(RuntimeError):
    """Raised when release evidence is incomplete or candidate-bound checks fail."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, arguments: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseGateError(f"git {' '.join(arguments)} failed: {details}") from exc
    return completed.stdout


def resolve_candidate(repo_root: Path, candidate: str) -> str:
    resolved = (
        _git(repo_root, ["rev-parse", "--verify", f"{candidate}^{{commit}}"]).decode().strip()
    )
    if not COMMIT_PATTERN.fullmatch(resolved):
        raise ReleaseGateError(f"candidate did not resolve to a full commit: {candidate!r}")
    return resolved


def validate_release_policy(repo_root: Path, path: Path) -> dict[str, Any]:
    """Validate a release policy stored inside the repository."""
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ReleaseGateError("release policy must be stored inside the repository") from exc
    policy = load_json(path)
    missing = sorted(POLICY_KEYS - set(policy))
    unexpected = sorted(set(policy) - POLICY_KEYS)
    if missing or unexpected:
        raise ReleaseGateError(
            f"release policy key mismatch; missing={missing}, unexpected={unexpected}"
        )
    if policy["schema_version"] != "1.0.0":
        raise ReleaseGateError("unsupported release policy schema version")
    validate_document(policy, repo_root / "configs/schema/release_gate_policy.schema.json")
    try:
        for value in policy["prohibited_path_patterns"]:
            re.compile(str(value), flags=re.IGNORECASE)
    except re.error as exc:
        raise ReleaseGateError(f"invalid prohibited path pattern: {exc}") from exc
    role_paths = [str(value) for value in dict(policy["required_roles"]).values()]
    if len(role_paths) != len(set(role_paths)):
        raise ReleaseGateError("required release roles must map to unique paths")
    if "remote_ci" in policy["required_gates"]:
        raise ReleaseGateError("remote_ci is reserved for completed remote attestations")
    return policy


def _candidate_regular_file(repo_root: Path, candidate: str, relative: str) -> bytes:
    raw = _git(
        repo_root,
        ["ls-tree", "-l", "-z", "--full-tree", candidate, "--", relative],
    )
    records = [record for record in raw.split(b"\0") if record]
    if len(records) != 1:
        raise ReleaseGateError(f"candidate does not contain required file: {relative}")
    metadata, raw_path = records[0].split(b"\t", 1)
    mode, object_type, object_id, _raw_size = metadata.split(maxsplit=3)
    if raw_path.decode("utf-8", errors="strict") != relative:
        raise ReleaseGateError(f"candidate path mismatch for required file: {relative}")
    if mode != b"100644" or object_type != b"blob":
        raise ReleaseGateError(f"candidate path is not a regular file: {relative}")
    return _blob(repo_root, object_id.decode())


def _candidate_policy(
    repo_root: Path,
    candidate: str,
    policy_path: Path,
) -> tuple[dict[str, Any], str, str]:
    """Load a policy only when its bytes are part of the exact candidate tree."""
    try:
        relative = policy_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ReleaseGateError("release policy must be stored inside the repository") from exc
    candidate_bytes = _candidate_regular_file(repo_root, candidate, relative)
    if not policy_path.is_file() or policy_path.read_bytes() != candidate_bytes:
        raise ReleaseGateError(
            f"working release policy differs from exact candidate bytes: {relative}"
        )
    policy = validate_release_policy(repo_root, policy_path)
    return policy, relative, _sha256_bytes(candidate_bytes)


def _candidate_version(repo_root: Path, candidate: str) -> str:
    raw = _candidate_regular_file(repo_root, candidate, "pyproject.toml")
    try:
        document = tomllib.loads(raw.decode("utf-8"))
        version = document["project"]["version"]
    except (KeyError, TypeError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseGateError("candidate pyproject.toml has no valid project.version") from exc
    if not isinstance(version, str) or not version:
        raise ReleaseGateError("candidate pyproject.toml has no valid project.version")
    return version


def _tree_entries(repo_root: Path, candidate: str) -> list[dict[str, Any]]:
    raw = _git(repo_root, ["ls-tree", "-r", "-l", "-z", "--full-tree", candidate])
    entries: list[dict[str, Any]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id, raw_size = metadata.split(maxsplit=3)
        path = raw_path.decode("utf-8", errors="strict")
        entries.append(
            {
                "path": path,
                "mode": mode.decode(),
                "object_type": object_type.decode(),
                "git_object": object_id.decode(),
                "git_blob_bytes": int(raw_size) if raw_size != b"-" else 0,
            }
        )
    if not entries:
        raise ReleaseGateError("candidate tree is empty")
    return entries


def _blob(repo_root: Path, object_id: str) -> bytes:
    return _git(repo_root, ["cat-file", "blob", object_id])


def _blobs(repo_root: Path, object_ids: Sequence[str]) -> dict[str, bytes]:
    """Read selected Git objects in one process, preserving exact blob bytes."""
    unique = tuple(dict.fromkeys(object_ids))
    if not unique:
        return {}
    request = "".join(f"{object_id}\n" for object_id in unique).encode("ascii")
    try:
        completed = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=repo_root,
            input=request,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseGateError(f"git cat-file --batch failed: {details}") from exc
    stream = io.BytesIO(completed.stdout)
    result: dict[str, bytes] = {}
    for requested in unique:
        header = stream.readline().decode("ascii").strip().split()
        if len(header) != 3 or header[1] != "blob":
            raise ReleaseGateError(f"unexpected git cat-file response for {requested}: {header}")
        size = int(header[2])
        payload = stream.read(size)
        if stream.read(1) != b"\n" or len(payload) != size:
            raise ReleaseGateError(f"truncated git cat-file response for {requested}")
        result[requested] = payload
    return result


def _lfs_metadata(payload: bytes) -> tuple[str, int] | None:
    if not payload.startswith(LFS_HEADER):
        return None
    oid_match = LFS_OID.search(payload)
    size_match = LFS_SIZE.search(payload)
    if oid_match is None or size_match is None:
        raise ReleaseGateError("malformed Git LFS pointer")
    return oid_match.group(1).decode(), int(size_match.group(1))


def scan_candidate_tree(repo_root: Path, candidate: str, policy_path: Path) -> dict[str, Any]:
    """Scan the exact Git candidate, never the mutable working tree."""
    resolved = resolve_candidate(repo_root, candidate)
    policy, policy_relative, policy_sha256 = _candidate_policy(repo_root, resolved, policy_path)
    entries = _tree_entries(repo_root, resolved)
    findings: list[dict[str, str]] = []
    folded: dict[str, str] = {}
    lfs_objects = 0
    lfs_bytes = 0
    patterns = [
        re.compile(str(value), flags=re.IGNORECASE) for value in policy["prohibited_path_patterns"]
    ]
    prohibited_suffixes = {str(value).casefold() for value in policy["prohibited_suffixes"]}
    text_suffixes = {str(value).casefold() for value in policy["text_suffixes"]}
    text_exemptions = tuple(str(value) for value in policy["text_scan_exempt_prefixes"])
    max_blob = int(policy["max_git_blob_bytes"])
    content_ids = [
        str(entry["git_object"])
        for entry in entries
        if int(entry["git_blob_bytes"]) <= 512
        or (
            PurePosixPath(str(entry["path"])).suffix.casefold() in text_suffixes
            and not str(entry["path"]).startswith(text_exemptions)
        )
    ]
    content = _blobs(repo_root, content_ids)

    for entry in entries:
        path = str(entry["path"])
        normalized = PurePosixPath(path)
        folded_path = path.casefold()
        if folded_path in folded and folded[folded_path] != path:
            findings.append({"code": "case_collision", "path": path, "detail": folded[folded_path]})
        folded[folded_path] = path
        if entry["object_type"] != "blob" or entry["mode"] in {"120000", "160000"}:
            findings.append(
                {
                    "code": "unsupported_tree_entry",
                    "path": path,
                    "detail": f"mode={entry['mode']} type={entry['object_type']}",
                }
            )
            continue
        for pattern in patterns:
            if pattern.search(path):
                findings.append(
                    {"code": "prohibited_path", "path": path, "detail": pattern.pattern}
                )
        if normalized.suffix.casefold() in prohibited_suffixes:
            findings.append(
                {"code": "prohibited_suffix", "path": path, "detail": normalized.suffix}
            )
        payload = content.get(str(entry["git_object"]))
        try:
            lfs = _lfs_metadata(payload) if payload is not None else None
        except ReleaseGateError as exc:
            findings.append({"code": "invalid_lfs_pointer", "path": path, "detail": str(exc)})
            lfs = None
        if lfs is not None:
            entry["lfs_oid_sha256"], entry["lfs_size_bytes"] = lfs
            lfs_objects += 1
            lfs_bytes += lfs[1]
        elif int(entry["git_blob_bytes"]) > max_blob:
            findings.append(
                {
                    "code": "oversized_git_blob",
                    "path": path,
                    "detail": f"{entry['git_blob_bytes']} > {max_blob}",
                }
            )
        if (
            normalized.suffix.casefold() in text_suffixes
            and not path.startswith(text_exemptions)
            and payload is not None
            and WINDOWS_USER_PATH.search(payload)
        ):
            findings.append(
                {"code": "local_user_path", "path": path, "detail": "Windows user path"}
            )

    material = [
        {
            key: entry[key]
            for key in (
                "path",
                "mode",
                "git_object",
                "git_blob_bytes",
                "lfs_oid_sha256",
                "lfs_size_bytes",
            )
            if key in entry
        }
        for entry in entries
    ]
    return bind_record_hash(
        {
            "schema_version": "1.0.0",
            "record_kind": "candidate_tree_scan",
            "candidate_commit": resolved,
            "created_at_utc": _utc_now(),
            "policy": {
                "path": policy_relative,
                "sha256": policy_sha256,
            },
            "entry_count": len(entries),
            "tree_sha256": _sha256_bytes(canonical_json_bytes(material)),
            "lfs_object_count": lfs_objects,
            "lfs_bytes": lfs_bytes,
            "findings": findings,
            "status": "pass" if not findings else "fail",
        }
    )


def write_candidate_scan(
    repo_root: Path,
    candidate: str,
    policy_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    report = scan_candidate_tree(repo_root, candidate, policy_path)
    validate_document(report, repo_root / "configs/schema/candidate_tree_scan.schema.json")
    write_json_create_only(output_path, report)
    if report["status"] != "pass":
        raise ReleaseGateError(f"candidate tree scan failed; see {output_path}")
    return report


def record_gate_status(
    repo_root: Path,
    *,
    candidate: str,
    gate: str,
    status: str,
    evidence_path: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    """Bind one completed gate to an exact commit and immutable evidence file."""
    if status not in {"pass", "fail", "not_run"}:
        raise ReleaseGateError(f"unsupported gate status: {status}")
    if status == "pass" and evidence_path is None:
        raise ReleaseGateError("a passing release gate requires an evidence file")
    if status == "not_run" and evidence_path is not None:
        raise ReleaseGateError("a not-run release gate cannot bind evidence")
    resolved = resolve_candidate(repo_root, candidate)
    evidence: dict[str, object] | None = None
    if evidence_path is not None:
        if not evidence_path.is_file():
            raise ReleaseGateError(f"gate evidence does not exist: {evidence_path}")
        try:
            relative = evidence_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError as exc:
            raise ReleaseGateError("gate evidence must be stored inside the repository") from exc
        evidence = {
            "path": relative,
            "sha256": _sha256_file(evidence_path),
            "size_bytes": evidence_path.stat().st_size,
        }
    record = bind_record_hash(
        {
            "schema_version": "1.0.0",
            "record_kind": "release_gate_status",
            "candidate_commit": resolved,
            "gate": gate,
            "status": status,
            "evidence": evidence,
            "created_at_utc": _utc_now(),
        }
    )
    validate_document(record, repo_root / "configs/schema/release_gate_status.schema.json")
    write_json_create_only(output_path, record)
    return record


def _bound_repository_file(repo_root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str):
        raise ReleaseGateError(f"{label} path must be a string")
    root = repo_root.resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ReleaseGateError(f"{label} path escapes the repository: {relative}")
    return path


def _validated_gate_status(
    repo_root: Path,
    path: Path,
    *,
    candidate: str,
    gate: str,
) -> dict[str, Any]:
    record = load_json(path)
    validate_document(record, repo_root / "configs/schema/release_gate_status.schema.json")
    validate_record_hash(record)
    expected = {
        "schema_version",
        "record_kind",
        "candidate_commit",
        "gate",
        "status",
        "evidence",
        "created_at_utc",
        "record_sha256",
    }
    if set(record) != expected:
        raise ReleaseGateError(f"unexpected fields in gate record: {path}")
    if record["record_kind"] != "release_gate_status":
        raise ReleaseGateError(f"wrong gate record kind: {path}")
    if record["candidate_commit"] != candidate or record["gate"] != gate:
        raise ReleaseGateError(f"gate record is not bound to {candidate}/{gate}: {path}")
    evidence = record["evidence"]
    if evidence is not None:
        if not isinstance(evidence, Mapping):
            raise ReleaseGateError(f"invalid evidence binding in {path}")
        evidence_file = _bound_repository_file(
            repo_root, evidence["path"], label="release-gate evidence"
        )
        if not evidence_file.is_file():
            raise ReleaseGateError(f"missing evidence bound by {path}: {evidence_file}")
        if (
            _sha256_file(evidence_file) != evidence["sha256"]
            or evidence_file.stat().st_size != evidence["size_bytes"]
        ):
            raise ReleaseGateError(f"evidence binding mismatch in {path}")
    return record


def _gate_evidence_file(repo_root: Path, record: Mapping[str, Any], *, gate: str) -> Path:
    evidence = record.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ReleaseGateError(f"passing {gate} gate has no evidence binding")
    return _bound_repository_file(repo_root, evidence.get("path"), label=f"{gate} evidence")


def _validate_candidate_tree_gate(
    repo_root: Path,
    record: Mapping[str, Any],
    *,
    candidate: str,
    policy_relative: str,
    policy_sha256: str,
) -> None:
    scan_path = _gate_evidence_file(repo_root, record, gate="candidate_tree")
    scan = validate_contract_file(repo_root, scan_path)
    if (
        scan["record_kind"] != "candidate_tree_scan"
        or scan["candidate_commit"] != candidate
        or scan["status"] != "pass"
        or scan["policy"]["path"] != policy_relative
        or scan["policy"]["sha256"] != policy_sha256
    ):
        raise ReleaseGateError("candidate_tree gate does not bind a passing exact-policy scan")


def _validate_remote_ci_gate(
    repo_root: Path,
    record: Mapping[str, Any],
    *,
    candidate: str,
    workflow_url: str,
) -> None:
    match = GITHUB_WORKFLOW_URL.fullmatch(workflow_url)
    if match is None:
        raise ReleaseGateError("completed attestation requires a canonical GitHub workflow URL")
    evidence_path = _gate_evidence_file(repo_root, record, gate="remote_ci")
    evidence = load_json(evidence_path)
    validate_document(evidence, repo_root / "configs/schema/github_workflow_run.schema.json")
    expected_repository = f"{match.group('owner')}/{match.group('repository')}"
    if (
        evidence["head_sha"] != candidate
        or evidence["html_url"] != workflow_url
        or int(evidence["id"]) != int(match.group("run_id"))
        or str(evidence["repository"]["full_name"]).casefold() != expected_repository.casefold()
    ):
        raise ReleaseGateError("remote_ci evidence does not match the candidate workflow run")


def assemble_release_gate_report(
    repo_root: Path,
    *,
    candidate: str,
    version: str,
    policy_path: Path,
    status_directory: Path,
    output_path: Path,
    mode: str,
    workflow_url: str | None = None,
) -> dict[str, Any]:
    """Assemble candidate or completed attestation from individually hashed gates."""
    if mode not in {"candidate_attestation", "completed_remote_attestation"}:
        raise ReleaseGateError(f"unsupported attestation mode: {mode}")
    resolved = resolve_candidate(repo_root, candidate)
    candidate_version = _candidate_version(repo_root, resolved)
    if version != candidate_version:
        raise ReleaseGateError(
            f"release version does not match candidate package: {version} != {candidate_version}"
        )
    policy, policy_relative, policy_sha256 = _candidate_policy(repo_root, resolved, policy_path)
    required = [str(value) for value in policy["required_gates"]]
    if mode == "completed_remote_attestation":
        if workflow_url is None or GITHUB_WORKFLOW_URL.fullmatch(workflow_url) is None:
            raise ReleaseGateError("completed attestation requires a canonical GitHub workflow URL")
        required.append("remote_ci")
    gates: dict[str, dict[str, object]] = {}
    for gate in required:
        path = status_directory / f"{gate}.status.json"
        if not path.is_file():
            gates[gate] = {"status": "not_run", "evidence": None}
            continue
        record = _validated_gate_status(repo_root, path, candidate=resolved, gate=gate)
        if gate == "candidate_tree" and record["status"] == "pass":
            _validate_candidate_tree_gate(
                repo_root,
                record,
                candidate=resolved,
                policy_relative=policy_relative,
                policy_sha256=policy_sha256,
            )
        if gate == "remote_ci" and record["status"] == "pass":
            if workflow_url is None:
                raise ReleaseGateError("remote_ci evidence requires a workflow URL")
            _validate_remote_ci_gate(
                repo_root,
                record,
                candidate=resolved,
                workflow_url=workflow_url,
            )
        gates[gate] = {"status": record["status"], "evidence": record["evidence"]}
    all_pass = all(value["status"] == "pass" for value in gates.values())
    status = (
        "pass"
        if mode == "completed_remote_attestation" and all_pass
        else "pending_remote_ci"
        if mode == "candidate_attestation" and all_pass
        else "fail"
    )
    report = bind_record_hash(
        {
            "schema_version": "1.0.0",
            "record_kind": "final_release_gate_report",
            "mode": mode,
            "candidate_commit": resolved,
            "created_at_utc": _utc_now(),
            "version": version,
            "workflow_url": workflow_url,
            "gates": gates,
            "status": status,
        }
    )
    validate_document(report, repo_root / "configs/schema/final_release_gate_report.schema.json")
    write_json_create_only(output_path, report)
    if status == "fail":
        raise ReleaseGateError(f"release gate report failed; see {output_path}")
    return report


def create_release_inventory(
    repo_root: Path,
    *,
    candidate: str,
    version: str,
    tag: str,
    policy_path: Path,
    gate_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Inventory every exact-candidate blob after an external gate has passed."""
    resolved = resolve_candidate(repo_root, candidate)
    candidate_version = _candidate_version(repo_root, resolved)
    if version != candidate_version:
        raise ReleaseGateError(
            f"release version does not match candidate package: {version} != {candidate_version}"
        )
    gate = validate_contract_file(repo_root, gate_report_path)
    if (
        gate["mode"] != "completed_remote_attestation"
        or gate["status"] != "pass"
        or gate["candidate_commit"] != resolved
        or gate["version"] != version
    ):
        raise ReleaseGateError(
            "release inventory requires a passing completed candidate attestation"
        )
    if tag != f"v{version}":
        raise ReleaseGateError(f"release tag must exactly match the package version: v{version}")
    policy, _policy_relative, _policy_sha256 = _candidate_policy(repo_root, resolved, policy_path)
    entries = _tree_entries(repo_root, resolved)
    required_paths = {str(path) for path in dict(policy["required_roles"]).values()}
    content = _blobs(
        repo_root,
        [
            str(entry["git_object"])
            for entry in entries
            if int(entry["git_blob_bytes"]) <= 512 or str(entry["path"]) in required_paths
        ],
    )
    inventory_entries: list[dict[str, object]] = []
    by_path: dict[str, dict[str, Any]] = {}
    lfs_count = 0
    lfs_bytes = 0
    for entry in entries:
        if entry["object_type"] != "blob":
            raise ReleaseGateError(
                f"non-blob candidate entry cannot be inventoried: {entry['path']}"
            )
        payload = content.get(str(entry["git_object"]))
        item: dict[str, object] = {
            "path": entry["path"],
            "mode": entry["mode"],
            "git_object": entry["git_object"],
            "git_blob_bytes": entry["git_blob_bytes"],
        }
        lfs = _lfs_metadata(payload) if payload is not None else None
        if lfs is not None:
            item["lfs_oid_sha256"], item["lfs_size_bytes"] = lfs
            lfs_count += 1
            lfs_bytes += lfs[1]
        inventory_entries.append(item)
        by_path[str(entry["path"])] = entry
    tree_sha256 = _sha256_bytes(canonical_json_bytes(inventory_entries))
    required_roles: list[dict[str, object]] = []
    for role, raw_path in sorted(dict(policy["required_roles"]).items()):
        path = str(raw_path)
        if path not in by_path:
            raise ReleaseGateError(f"candidate is missing required release role {role}: {path}")
        entry = by_path[path]
        payload = content.get(str(entry["git_object"]))
        if payload is None:
            payload = _blob(repo_root, str(entry["git_object"]))
        required_roles.append(
            {
                "role": str(role),
                "path": path,
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    try:
        gate_relative = gate_report_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ReleaseGateError("gate report must be stored inside the repository") from exc
    inventory = bind_record_hash(
        {
            "schema_version": "1.0.0",
            "record_kind": "release_evidence_inventory",
            "candidate_commit": resolved,
            "version": version,
            "tag": tag,
            "created_at_utc": _utc_now(),
            "release_gate": {
                "path": gate_relative,
                "sha256": _sha256_file(gate_report_path),
                "status": "pass",
            },
            "tree": {
                "entry_count": len(inventory_entries),
                "tree_sha256": tree_sha256,
                "lfs_object_count": lfs_count,
                "lfs_bytes": lfs_bytes,
                "entries": inventory_entries,
            },
            "required_roles": required_roles,
            "status": "ready",
        }
    )
    validate_document(
        inventory, repo_root / "configs/schema/release_evidence_inventory.schema.json"
    )
    write_json_create_only(output_path, inventory)
    return inventory


def dump_json(payload: Mapping[str, Any]) -> str:
    """Render concise deterministic CLI output."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
