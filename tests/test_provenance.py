from __future__ import annotations

import json
from pathlib import Path

import pytest

from heartshift.research.provenance import write_post_run_source_snapshot


def test_post_run_source_snapshot_hashes_declared_bytes_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "artifacts" / "runs" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "config.resolved.json").write_text("{}\n", encoding="utf-8")
    source = tmp_path / "method.py"
    source.write_text("METHOD = 1\n", encoding="utf-8")
    output = write_post_run_source_snapshot(tmp_path, run_dir, [source])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "post_run_exact_source_snapshot"
    assert {row["path"] for row in payload["files"]} == {
        "artifacts/runs/run/run_manifest.json",
        "artifacts/runs/run/config.resolved.json",
        "method.py",
    }
    with pytest.raises(FileExistsError):
        write_post_run_source_snapshot(tmp_path, run_dir, [source])
