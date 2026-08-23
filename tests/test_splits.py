from __future__ import annotations

from pathlib import Path

from heartshift.data.splits import (
    build_inner_manifest,
    build_outer_manifest,
    validate_inner_manifest,
    validate_outer_manifest,
)
from heartshift.data.uci import load_uci_heart

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = REPO_ROOT / "data/raw/uci_heart/doi-10.24432-C52P4X/extracted"


def test_outer_manifest_is_hospital_disjoint() -> None:
    data = load_uci_heart(EXTRACTED)
    manifest = build_outer_manifest(data)
    validate_outer_manifest(data, manifest)
    assert len(manifest) == 4 * len(data)


def test_inner_manifest_never_contains_outer_target() -> None:
    data = load_uci_heart(EXTRACTED)
    manifest = build_inner_manifest(data)
    validate_inner_manifest(data, manifest)
    for outer_target, fold in manifest.groupby("outer_target"):
        assert outer_target not in set(fold["site"])
