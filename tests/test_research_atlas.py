"""Guard against lost trials, hidden failures, and corrupted recovered evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_inventory_keeps_failed_gate_even_when_audit_is_complete(tmp_path: Path) -> None:
    atlas = tool("build_research_atlas")
    for kind in ("runs", "reports", "failures"):
        (tmp_path / "artifacts" / kind).mkdir(parents=True)
    run = tmp_path / "artifacts/runs/fixture-smoke"
    write_json(run / "acceptance_gate.json", {"passed": False})
    write_json(run / "evidence_audit.json", {"status": "complete"})
    reference = tmp_path / "interpretation.md"
    reference.write_text("Smoke only", encoding="utf-8")
    catalog = {
        "entries": [
            {
                "path": "artifacts/runs/fixture-smoke",
                "family": "Fixture",
                "scope": "pipeline_smoke",
                "note": "Failed smoke gate",
                "reference": "interpretation.md",
            }
        ]
    }
    rows = atlas.inspect_trials(tmp_path, catalog)
    assert rows[0]["scope"] == "pipeline_smoke"
    assert rows[0]["recorded_signals"] == ['acceptance_gate.json: {"passed": false}']
    assert {record["path"] for record in rows[0]["status_records"]} == {
        "artifacts/runs/fixture-smoke/acceptance_gate.json",
        "artifacts/runs/fixture-smoke/evidence_audit.json",
    }
    (tmp_path / "artifacts/runs/unclassified-result").mkdir()
    with pytest.raises(ValueError, match="unclassified-result"):
        atlas.inspect_trials(tmp_path, catalog)


def test_recovery_accepts_declared_git_line_endings_but_rejects_changed_values(
    tmp_path: Path,
) -> None:
    atlas = tool("build_research_atlas")
    original, git_blob = b"value\r\n1\r\n", b"value\n1\n"
    target = tmp_path / "result.csv"
    target.write_bytes(original)
    write_json(
        tmp_path / atlas.AUDIT / "coursework_manifest.json",
        {
            "counts": {"already_tracked_byte_identical": 1},
            "files": [
                {
                    "path": "source.csv",
                    "disposition": "already_tracked_byte_identical",
                    "repository_path": "result.csv",
                    "bytes": len(original),
                    "sha256": hashlib.sha256(original).hexdigest(),
                    "repository_normalization": "git_text_lf",
                    "repository_bytes": len(git_blob),
                    "repository_sha256": hashlib.sha256(git_blob).hexdigest(),
                }
            ],
        },
    )
    write_json(tmp_path / atlas.AUDIT / "notebook_source_index.json", {"notebooks": []})
    atlas.validate_recovery(tmp_path)
    target.write_bytes(git_blob)
    atlas.validate_recovery(tmp_path)
    target.write_bytes(b"value\n2\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        atlas.validate_recovery(tmp_path)


def test_seed_figure_rejects_incomplete_variant_selection(tmp_path: Path) -> None:
    plots = tool("plot_research_overview")
    path = tmp_path / plots.SOURCES[0]
    path.parent.mkdir(parents=True)
    path.write_text("experiment\nv4_structured_policy_bank\n", encoding="utf-8")
    with pytest.raises(ValueError, match="all ten frozen variants"):
        plots.figure_data(tmp_path)


def test_figure_manifest_cannot_omit_an_output(tmp_path: Path) -> None:
    plots = tool("plot_research_overview")
    for relative in plots.SOURCES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    manifest = json.loads((ROOT / plots.OUTPUT / "figure_manifest.json").read_text())
    manifest["outputs"] = []
    write_json(tmp_path / plots.OUTPUT / "figure_manifest.json", manifest)
    with pytest.raises(ValueError, match="Figure output set changed"):
        plots.check(tmp_path)


def test_document_links_support_angle_bracket_paths_with_spaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = tool("validate_repository")
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    (tmp_path / "a directory").mkdir()
    (tmp_path / "a directory/data.csv").write_text("value\n1\n", encoding="utf-8")
    doc = tmp_path / "README.md"
    doc.write_text("[Source](<a directory/data.csv>)", encoding="utf-8")
    validator.validate_document_links("README.md")
    doc.write_text("[Source](<a directory/missing.csv>)", encoding="utf-8")
    with pytest.raises(SystemExit, match="broken link"):
        validator.validate_document_links("README.md")


def test_workflow_validator_rejects_malformed_action_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = tool("validate_repository")
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    workflow = tmp_path / ".github/workflows/check.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("steps:\n  - uses: actions/checkout@" + "a" * 40 + "\n")
    validator.validate_workflow_pins()
    workflow.write_text("steps:\n  - uses: actions/checkout@" + "a" * 41 + "\n")
    with pytest.raises(SystemExit, match="40-character commit"):
        validator.validate_workflow_pins()
