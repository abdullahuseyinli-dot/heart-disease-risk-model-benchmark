from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from heartshift.contracts import load_json, validate_record_hash
from heartshift.release import (
    ReleaseGateError,
    assemble_release_gate_report,
    create_release_inventory,
    record_gate_status,
    resolve_candidate,
    scan_candidate_tree,
    validate_release_policy,
    write_candidate_scan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _release_repository(tmp_path: Path, *, max_blob: int = 1024 * 1024) -> tuple[Path, str, Path]:
    root = tmp_path / "release-repository"
    root.mkdir()
    shutil.copytree(REPO_ROOT / "configs/schema", root / "configs/schema")
    (root / "README.md").write_text("# Release fixture\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "release-fixture"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    policy_path = root / "release-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "max_git_blob_bytes": max_blob,
                "prohibited_path_patterns": ["(^|/)\\.env($|\\.)"],
                "prohibited_suffixes": [".pem"],
                "required_gates": ["candidate_tree", "tests"],
                "required_roles": {"readme": "README.md"},
                "text_scan_exempt_prefixes": [],
                "text_suffixes": [".md", ".json"],
            }
        ),
        encoding="utf-8",
    )
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "HeartShift Test")
    _git(root, "config", "user.email", "test@example.org")
    _git(
        root,
        "add",
        "README.md",
        "pyproject.toml",
        "release-policy.json",
        "configs/schema",
    )
    _git(root, "commit", "-q", "-m", "candidate")
    candidate = _git(root, "rev-parse", "HEAD")
    return root, candidate, policy_path


def test_candidate_scan_reads_exact_git_tree_and_is_create_only(tmp_path: Path) -> None:
    root, candidate, policy = _release_repository(tmp_path)
    (root / "README.md").write_text("uncommitted local change\n", encoding="utf-8")
    report = scan_candidate_tree(root, candidate, policy)
    validate_record_hash(report)
    assert report["status"] == "pass"
    assert report["candidate_commit"] == candidate
    assert report["entry_count"] > 1

    output = root / ".audit/release/candidate-tree.json"
    write_candidate_scan(root, candidate, policy, output)
    with pytest.raises(FileExistsError):
        write_candidate_scan(root, candidate, policy, output)


def test_candidate_scan_reports_oversized_blob(tmp_path: Path) -> None:
    root, candidate, policy = _release_repository(tmp_path, max_blob=3)
    report = scan_candidate_tree(root, candidate, policy)
    assert report["status"] == "fail"
    assert any(item["code"] == "oversized_git_blob" for item in report["findings"])


def test_candidate_scan_requires_policy_bytes_from_candidate(tmp_path: Path) -> None:
    root, candidate, policy = _release_repository(tmp_path)
    payload = json.loads(policy.read_text(encoding="utf-8"))
    payload["max_git_blob_bytes"] = 10_000_000
    policy.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReleaseGateError, match="differs from exact candidate"):
        scan_candidate_tree(root, candidate, policy)


def test_repository_release_policy_is_valid() -> None:
    policy = validate_release_policy(
        REPO_ROOT, REPO_ROOT / "configs/release/release_gate_policy_v1.json"
    )
    assert policy["required_gates"][0] == "candidate_tree"


def test_two_stage_gate_and_inventory_are_candidate_bound(tmp_path: Path) -> None:
    root, candidate, policy = _release_repository(tmp_path)
    audit = root / ".audit/release" / candidate
    audit.mkdir(parents=True)
    scan = audit / "candidate-tree.json"
    write_candidate_scan(root, candidate, policy, scan)
    record_gate_status(
        root,
        candidate=candidate,
        gate="candidate_tree",
        status="pass",
        evidence_path=scan,
        output_path=audit / "candidate_tree.status.json",
    )
    test_log = audit / "tests.log"
    test_log.write_text("tests passed\n", encoding="utf-8")
    record_gate_status(
        root,
        candidate=candidate,
        gate="tests",
        status="pass",
        evidence_path=test_log,
        output_path=audit / "tests.status.json",
    )

    candidate_report = assemble_release_gate_report(
        root,
        candidate=candidate,
        version="1.2.3",
        policy_path=policy,
        status_directory=audit,
        output_path=audit / "candidate-attestation.json",
        mode="candidate_attestation",
    )
    assert candidate_report["status"] == "pending_remote_ci"

    remote_log = audit / "remote-ci.json"
    remote_log.write_text(
        json.dumps(
            {
                "id": 1,
                "head_sha": candidate,
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://github.com/example/project/actions/runs/1",
                "run_attempt": 1,
                "repository": {"full_name": "example/project"},
            }
        ),
        encoding="utf-8",
    )
    record_gate_status(
        root,
        candidate=candidate,
        gate="remote_ci",
        status="pass",
        evidence_path=remote_log,
        output_path=audit / "remote_ci.status.json",
    )
    completed_path = audit / "completed-attestation.json"
    completed = assemble_release_gate_report(
        root,
        candidate=candidate,
        version="1.2.3",
        policy_path=policy,
        status_directory=audit,
        output_path=completed_path,
        mode="completed_remote_attestation",
        workflow_url="https://github.com/example/project/actions/runs/1",
    )
    assert completed["status"] == "pass"

    inventory_path = audit / "release-inventory.json"
    inventory = create_release_inventory(
        root,
        candidate=candidate,
        version="1.2.3",
        tag="v1.2.3",
        policy_path=policy,
        gate_report_path=completed_path,
        output_path=inventory_path,
    )
    assert inventory["status"] == "ready"
    assert inventory["candidate_commit"] == candidate
    assert inventory["required_roles"][0]["role"] == "readme"
    validate_record_hash(load_json(inventory_path))


def test_gate_rejects_missing_status_and_wrong_candidate(tmp_path: Path) -> None:
    root, candidate, policy = _release_repository(tmp_path)
    audit = root / ".audit/release" / candidate
    audit.mkdir(parents=True)
    with pytest.raises(ReleaseGateError, match="failed"):
        assemble_release_gate_report(
            root,
            candidate=candidate,
            version="1.2.3",
            policy_path=policy,
            status_directory=audit,
            output_path=audit / "failed-attestation.json",
            mode="candidate_attestation",
        )
    failed = load_json(audit / "failed-attestation.json")
    assert failed["gates"]["tests"]["status"] == "not_run"
    with pytest.raises(ReleaseGateError):
        resolve_candidate(root, "not-a-commit")


def test_gate_rejects_unsubstantiated_pass_and_wrong_version(tmp_path: Path) -> None:
    root, candidate, policy = _release_repository(tmp_path)
    with pytest.raises(ReleaseGateError, match="requires an evidence file"):
        record_gate_status(
            root,
            candidate=candidate,
            gate="tests",
            status="pass",
            evidence_path=None,
            output_path=root / ".audit/tests.status.json",
        )
    audit = root / ".audit/release" / candidate
    audit.mkdir(parents=True)
    with pytest.raises(ReleaseGateError, match="does not match candidate package"):
        assemble_release_gate_report(
            root,
            candidate=candidate,
            version="9.9.9",
            policy_path=policy,
            status_directory=audit,
            output_path=audit / "wrong-version.json",
            mode="candidate_attestation",
        )
