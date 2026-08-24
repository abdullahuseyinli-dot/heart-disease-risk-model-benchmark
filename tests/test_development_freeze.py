from __future__ import annotations

import json

import pytest

from heartshift.research.development_freeze import freeze_development_candidate


def test_development_freeze_hashes_declared_files_and_refuses_overwrite(
    tmp_path,
) -> None:
    source = tmp_path / "method.py"
    source.write_text("METHOD = 1\n", encoding="utf-8")
    config = {
        "protocol_version": "test-v1",
        "status": "consumed_test",
        "files": ["method.py"],
    }
    output = tmp_path / "artifacts" / "locks" / "test.json"
    freeze_development_candidate(tmp_path, config, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["new_confirmatory_claim_allowed"] is False
    assert payload["files"][0]["path"] == "method.py"
    with pytest.raises(FileExistsError):
        freeze_development_candidate(tmp_path, config, output)
