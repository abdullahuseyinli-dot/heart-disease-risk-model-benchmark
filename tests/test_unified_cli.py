from __future__ import annotations

import sys
from pathlib import Path

import pytest

from heartshift.cli.main import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_unified_cli_parser_exposes_release_surface() -> None:
    args = build_parser().parse_args(
        [
            "release",
            "assemble",
            "--candidate",
            "a" * 40,
            "--version",
            "0.1.0",
            "--status-directory",
            ".audit/release",
            "--output",
            ".audit/report.json",
        ]
    )
    assert args.command == "release"
    assert args.release_command == "assemble"
    assert args.mode == "candidate_attestation"


@pytest.mark.parametrize(
    ("arguments", "marker"),
    [
        (["methods", "validate"], "Method registry valid: 18 entries"),
        (["methods", "list"], "tabpfn_v2_v3"),
        (
            [
                "contracts",
                "validate",
                "--path",
                "manifests/evidence/heart_prediction_table_v1.json",
            ],
            '"status": "pass"',
        ),
        (
            [
                "data",
                "verify",
                "--manifest",
                "manifests/datasets/uci_heart_v1.json",
            ],
            '"status": "verified"',
        ),
    ],
)
def test_unified_cli_read_only_commands(
    arguments: list[str],
    marker: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["heartshift", *arguments, "--repo-root", str(REPO_ROOT)],
    )
    main()
    assert marker in capsys.readouterr().out


def test_unified_cli_version(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["heartshift", "--version"])
    with pytest.raises(SystemExit, match="0"):
        main()
    assert "HeartShift 0.1.0" in capsys.readouterr().out


def test_unified_cli_complete_validation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["heartshift", "validate", "--repo-root", str(REPO_ROOT)],
    )
    main()
    assert "data, study, and contract validation passed" in capsys.readouterr().out
