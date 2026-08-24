from __future__ import annotations

from pathlib import Path

from heartshift.evaluation.classical_benchmark import _portable_argv


def test_portable_argv_removes_workstation_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repository"
    repo_root.mkdir()
    internal = repo_root / "configs" / "study.yaml"
    external = tmp_path / "private" / "credentials.json"

    portable = _portable_argv(
        repo_root,
        [
            str(repo_root),
            str(internal),
            "--config",
            str(external),
            "relative/path",
        ],
    )

    assert portable == [
        ".",
        "configs/study.yaml",
        "--config",
        "<external>/credentials.json",
        "relative/path",
    ]
    assert str(tmp_path) not in " ".join(portable)
