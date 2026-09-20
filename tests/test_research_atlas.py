"""Guard against lost trials, hidden failures, and corrupted recovered evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
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


@pytest.fixture
def results_workspace(tmp_path: Path) -> Path:
    results = tool("build_results_document")
    paths = [
        *results.TABLES,
        *results.JSON_SOURCES,
        results.DOCUMENT,
        results.MANIFEST,
        "README.md",
    ]
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


@pytest.mark.parametrize("duplicate", [False, True])
def test_result_tables_reject_missing_or_duplicated_methods(
    results_workspace: Path, duplicate: bool
) -> None:
    results = tool("build_results_document")
    path = results_workspace / results.HEART / "primary_estimands.csv"
    lines = path.read_text().splitlines()
    if duplicate:
        lines[-1] = lines[-2]
    else:
        lines.pop()
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="Incomplete or duplicate result set"):
        results.read_tables(results_workspace)


def test_result_table_method_set_must_match_frozen_report(results_workspace: Path) -> None:
    results = tool("build_results_document")
    path = results_workspace / results.HEART / "primary_estimands.csv"
    path.write_text(path.read_text().replace("psmask:v2_prior_separated", "unknown:method"))
    with pytest.raises(ValueError, match="Method set differs from the frozen report"):
        results.read_tables(results_workspace)


def test_result_tables_cannot_silently_include_an_adaptation_track(results_workspace: Path) -> None:
    results = tool("build_results_document")
    path = results_workspace / results.HEART / "report_manifest.json"
    payload = json.loads(path.read_text())
    payload["tracks"].append("uda")
    write_json(path, payload)
    with pytest.raises(ValueError, match="Automatic adaptation cannot be pooled"):
        results.read_tables(results_workspace)


def test_comparison_reference_must_match_caption(results_workspace: Path) -> None:
    results = tool("build_results_document")
    path = results_workspace / results.READMISSION / "patient_cluster_bootstrap_intervals.csv"
    path.write_text(path.read_text().replace("logistic_environment_class_balanced", "pooled_erm"))
    with pytest.raises(ValueError, match="Registered comparison reference changed"):
        results.read_tables(results_workspace)


@pytest.mark.parametrize("report", ["DEVELOPMENT", "STABILITY"])
def test_post_outcome_results_cannot_be_relabelled_confirmatory(
    results_workspace: Path, report: str
) -> None:
    results = tool("build_results_document")
    path = results_workspace / getattr(results, report) / "report_manifest.json"
    payload = json.loads(path.read_text())
    payload["new_confirmatory_claim_allowed"] = True
    write_json(path, payload)
    with pytest.raises(ValueError, match="Development scope or method count changed"):
        results.read_tables(results_workspace)


def test_result_presentation_rejects_stale_headlines_and_changed_source_bytes(
    results_workspace: Path,
) -> None:
    results = tool("build_results_document")
    results.build(results_workspace, check=True)
    readme = results_workspace / "README.md"
    original = readme.read_text(encoding="utf-8")
    readme.write_text(original.replace("0.602509", "0.123456", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="README findings do not match"):
        results.build(results_workspace, check=True)
    readme.write_text(original, encoding="utf-8")
    report = results_workspace / results.HEART / "primary_estimands.csv"
    report.write_bytes(report.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source bindings changed"):
        results.build(results_workspace, check=True)


def test_comparison_table_uses_bootstrap_mean_not_point_estimate_subtraction() -> None:
    results = tool("build_results_document")
    rows = [
        {
            "method": "candidate",
            "mean_difference": "-0.04",
            "observed_difference": "-0.01",
            "ci_025": "-0.06",
            "ci_975": "-0.02",
        }
    ]
    rendered = results.comparisons(rows, "method")
    assert "-0.040000" in rendered
    assert "-0.010000" not in rendered


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_result_is_not_presented_as_a_completed_metric(value: str) -> None:
    with pytest.raises(ValueError, match="Non-finite metric"):
        tool("build_results_document").number(value)
