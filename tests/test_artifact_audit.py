from __future__ import annotations

import json

import pandas as pd
import pytest

from heartshift.research.artifact_audit import (
    validate_source_inner_artifact_audit,
    write_source_inner_artifact_audit,
)


def _source_only_run(tmp_path):
    run_dir = tmp_path / "source-run"
    run_dir.mkdir()
    config_hash = "config-hash"
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {"phase": "psmask_inner_source_only", "config_sha256": config_hash}
        )
        + "\n",
        encoding="utf-8",
    )
    sites = ("cleveland", "hungary", "switzerland", "va_long_beach")
    rows = []
    for index, outer_target in enumerate(sites):
        site = sites[(index + 1) % len(sites)]
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "site": site,
                "target": index % 2,
                "outer_target": outer_target,
                "inner_validation": site,
                "record_sha256": f"record-{index}",
                "config_sha256": config_hash,
                "y_score": 0.25 + index / 10,
            }
        )
    pd.DataFrame(rows).to_parquet(run_dir / "inner_predictions.parquet", index=False)
    pd.DataFrame({"outer_target": sites}).to_csv(
        run_dir / "selected_configurations.csv", index=False
    )
    (run_dir / "config.resolved.json").write_text("{}\n", encoding="utf-8")
    shard_dir = run_dir / "prediction_shards"
    shard_dir.mkdir()
    (shard_dir / "preserved.txt").write_text("evidence\n", encoding="utf-8")
    return run_dir


def test_source_inner_artifact_audit_hashes_the_exact_tree(tmp_path) -> None:
    run_dir = _source_only_run(tmp_path)
    output = write_source_inner_artifact_audit(run_dir)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_count"] == 5
    assert "prediction_shards/preserved.txt" in payload["sha256"]
    result = validate_source_inner_artifact_audit(run_dir)
    assert result["prediction_rows"] == 4
    with pytest.raises(FileExistsError):
        write_source_inner_artifact_audit(run_dir)


def test_source_inner_artifact_audit_detects_changed_evidence(tmp_path) -> None:
    run_dir = _source_only_run(tmp_path)
    write_source_inner_artifact_audit(run_dir)
    (run_dir / "prediction_shards/preserved.txt").write_text(
        "changed\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="hash failed"):
        validate_source_inner_artifact_audit(run_dir)
