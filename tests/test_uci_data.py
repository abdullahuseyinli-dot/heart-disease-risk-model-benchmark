from __future__ import annotations

from pathlib import Path

import pandas as pd

from heartshift.data.uci import EXPECTED_SITE_ROWS, FEATURE_COLUMNS, load_uci_heart

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = REPO_ROOT / "data/raw/uci_heart/doi-10.24432-C52P4X/extracted"


def test_official_uci_reconstruction() -> None:
    data = load_uci_heart(EXTRACTED)
    assert len(data) == 920
    assert data["sample_id"].is_unique
    assert data.groupby("site").size().to_dict() == EXPECTED_SITE_ROWS
    assert set(data["target"].unique()) == {0, 1}
    assert not data["target"].isna().any()
    assert set(FEATURE_COLUMNS).issubset(data.columns)


def test_zero_sentinels_are_explicit_not_silent_values() -> None:
    data = load_uci_heart(EXTRACTED)
    for column in ("trestbps", "chol", "thalach"):
        assert not data[column].eq(0).any()
        assert data.loc[data[f"{column}_zero_sentinel"], column].isna().all()


def test_profile_expected_prevalence_counts() -> None:
    data = load_uci_heart(EXTRACTED)
    counts = data.groupby("site")["target"].agg(["size", "sum"])
    assert counts.loc["cleveland"].to_dict() == {"size": 303, "sum": 139}
    assert counts.loc["hungary"].to_dict() == {"size": 294, "sum": 106}
    assert counts.loc["switzerland"].to_dict() == {"size": 123, "sum": 115}
    assert counts.loc["va_long_beach"].to_dict() == {"size": 200, "sum": 149}


def test_canonical_table_round_trip_after_prepare() -> None:
    canonical = REPO_ROOT / "data/processed/uci_heart_canonical_v1.parquet"
    if canonical.exists():
        frame = pd.read_parquet(canonical)
        assert len(frame) == 920
        assert frame["sample_id"].is_unique
