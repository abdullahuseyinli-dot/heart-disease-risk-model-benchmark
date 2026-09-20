from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from heartshift.config import config_hash
from heartshift.research.outer_evidence import (
    _assert_frames_match,
    _recompute_binary_metric_table,
    _resolved_config,
    _validate_manifest,
    _write_evidence,
)


def test_frame_reconstruction_is_order_independent_and_detects_drift() -> None:
    expected = pd.DataFrame({"sample_id": ["a", "b"], "score": [0.25, 0.75]})
    observed = expected.iloc[::-1].reset_index(drop=True)
    assert _assert_frames_match(observed, expected, sort_by=["sample_id"]) == 0.0
    with pytest.raises(AssertionError, match="columns differ"):
        _assert_frames_match(observed.drop(columns="score"), expected, sort_by=["sample_id"])
    changed = observed.copy()
    changed.loc[0, "score"] = 0.5
    with pytest.raises(AssertionError):
        _assert_frames_match(changed, expected, sort_by=["sample_id"])


def test_metric_reconstruction_uses_declared_groups() -> None:
    predictions = pd.DataFrame(
        {
            "site": ["a", "a", "b", "b"],
            "target": [0, 1, 0, 1],
            "score": [0.1, 0.9, 0.2, 0.8],
        }
    )
    metrics = _recompute_binary_metric_table(
        predictions,
        grouping=["site"],
        score_column="score",
        workers=2,
        dropna=False,
    )
    assert set(metrics["site"]) == {"a", "b"}
    assert metrics["roc_auc"].eq(1.0).all()


def test_resolved_config_manifest_and_evidence_writes_are_audited(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):
        _resolved_config(run)
    config = {"protocol": "fixture", "seed": 1}
    (run / "config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
    assert _resolved_config(run) == config
    manifest = {"config_sha256": config_hash(config), "status": "complete"}
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _validate_manifest(run, config) == manifest

    outputs = _write_evidence(
        run,
        {"status": "passed", "prediction_rows": 4},
        kind="fixture_outer",
        replace=False,
    )
    audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
    assert audit["audit_status"] == "complete_locked_outer_evidence"
    assert "independent_validation.json" in audit["sha256"]
    with pytest.raises(FileExistsError, match="already exists"):
        _write_evidence(run, {}, kind="fixture_outer", replace=False)

    wrong = dict(config)
    wrong["seed"] = 2
    with pytest.raises(AssertionError, match="does not reconstruct"):
        _validate_manifest(run, wrong)
