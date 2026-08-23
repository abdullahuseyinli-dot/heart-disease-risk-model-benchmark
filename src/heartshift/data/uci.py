"""Deterministic reconstruction of the four-centre UCI Heart dataset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "uci-heart-canonical-v1"
UCI_DOI = "10.24432/C52P4X"
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

RAW_COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target_ordinal",
)
FEATURE_COLUMNS = RAW_COLUMNS[:-1]
CONTINUOUS_COLUMNS = ("age", "trestbps", "chol", "thalach", "oldpeak")
CATEGORICAL_COLUMNS = ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")
CORE_COLUMNS = ("age", "sex", "cp")
FEATURE_PANELS: dict[str, tuple[str, ...]] = {
    "core": CORE_COLUMNS,
    "routine": ("trestbps", "chol", "fbs", "restecg"),
    "exercise": ("thalach", "exang", "oldpeak", "slope"),
    "advanced": ("ca", "thal"),
}

SITE_FILES: dict[str, str] = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va_long_beach": "processed.va.data",
}
EXPECTED_SITE_ROWS: dict[str, int] = {
    "cleveland": 303,
    "hungary": 294,
    "switzerland": 123,
    "va_long_beach": 200,
}
EXPECTED_PROCESSED_SHA256: dict[str, str] = {
    "processed.cleveland.data": "A74B7EFA387BC9D108D7D0115D831FE9B414B29AE7124F331B622B4EFA0427C8",
    "processed.hungarian.data": "D1AD108F785768CD3D7E82DC522E6F5A61EEA93CCCFB3A46EE8076F73FC3D796",
    "processed.switzerland.data": (
        "834A405CCF5B66AB4056BB77794ADC8DF0B7125186454C0A1D002D33C6C3B314"
    ),
    "processed.va.data": "E7C93D8D0D2ACDADFA4C5E8DE768E2191E7F618B952E29623F1F0D5949FF6B8F",
}
EXPECTED_ARCHIVE_SHA256 = "B17CD273DA9CE1CAA4710FCE80227EA454D4DBF9FCBC8E6A9121672751563ADC"

# These zero values are physiologically invalid source placeholders. The original
# token is retained through the corresponding indicator and immutable raw files.
ZERO_SENTINEL_COLUMNS = ("trestbps", "chol", "thalach")


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 digest without modifying the file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_value(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.12g}"


def _row_fingerprint(values: Iterable[float]) -> str:
    payload = "|".join(_canonical_value(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_site(extracted_dir: Path, site: str, filename: str) -> pd.DataFrame:
    path = extracted_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing official UCI source file: {path}")
    actual_hash = sha256_file(path)
    expected_hash = EXPECTED_PROCESSED_SHA256[filename]
    if actual_hash != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {filename}: {actual_hash} != {expected_hash}")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    frame = pd.read_csv(
        path,
        header=None,
        names=list(RAW_COLUMNS),
        na_values=["?"],
        dtype="string",
    )
    if len(frame) != EXPECTED_SITE_ROWS[site]:
        raise ValueError(f"Unexpected row count for {site}: {len(frame)}")
    if len(raw_lines) != len(frame):
        raise ValueError(f"Raw-line count mismatch for {site}")

    for column in RAW_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    parsed_for_fingerprint = frame.loc[:, RAW_COLUMNS].copy()
    frame.insert(0, "site", site)
    frame.insert(0, "source_row", np.arange(1, len(frame) + 1, dtype=np.int32))
    line_hashes = [hashlib.sha256(line.encode("utf-8")).hexdigest() for line in raw_lines]
    frame.insert(0, "source_line_sha256", line_hashes)
    frame.insert(
        0,
        "sample_id",
        [f"uci:{site}:{row:04d}:{line_hash[:12]}" for row, line_hash in enumerate(line_hashes, 1)],
    )
    frame["source_file"] = filename
    frame["source_file_sha256"] = actual_hash

    frame["record_sha256"] = [
        _row_fingerprint(row) for row in parsed_for_fingerprint.itertuples(index=False, name=None)
    ]

    for column in ZERO_SENTINEL_COLUMNS:
        sentinel = frame[column].eq(0).fillna(False).astype(bool)
        frame[f"{column}_zero_sentinel"] = sentinel
        frame.loc[sentinel, column] = np.nan

    frame["target"] = frame["target_ordinal"].gt(0).astype(np.int8)
    frame["schema_version"] = SCHEMA_VERSION
    return frame


def load_uci_heart(extracted_dir: Path) -> pd.DataFrame:
    """Load all four official processed cohorts with immutable provenance."""
    frames = [_read_site(extracted_dir, site, filename) for site, filename in SITE_FILES.items()]
    data = pd.concat(frames, ignore_index=True)
    if len(data) != sum(EXPECTED_SITE_ROWS.values()):
        raise AssertionError("Combined UCI Heart row count changed")
    if not data["sample_id"].is_unique:
        raise AssertionError("sample_id is not unique")

    duplicate_sizes = data.groupby("record_sha256")["sample_id"].transform("size")
    data["duplicate_cluster_size"] = duplicate_sizes.astype(np.int16)
    data["is_exact_duplicate"] = duplicate_sizes.gt(1)
    return data


def build_profile(data: pd.DataFrame) -> dict[str, Any]:
    """Build a JSON-serializable data profile used as an acceptance artifact."""
    site_profiles: dict[str, Any] = {}
    for site, site_frame in data.groupby("site", sort=False):
        site_profiles[str(site)] = {
            "n": len(site_frame),
            "positives": int(site_frame["target"].sum()),
            "prevalence": float(site_frame["target"].mean()),
            "missing_fraction": {
                column: float(site_frame[column].isna().mean()) for column in FEATURE_COLUMNS
            },
            "zero_sentinel_count": {
                column: int(site_frame[f"{column}_zero_sentinel"].sum())
                for column in ZERO_SENTINEL_COLUMNS
            },
        }
    duplicate_clusters = (
        data.loc[data["is_exact_duplicate"]]
        .groupby("record_sha256", sort=True)["sample_id"]
        .apply(list)
        .to_dict()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "uci_doi": UCI_DOI,
        "n": len(data),
        "n_features": len(FEATURE_COLUMNS),
        "target_definition": "target = 1[target_ordinal > 0]",
        "sites": site_profiles,
        "exact_duplicate_clusters": duplicate_clusters,
    }


def _raw_manifest(raw_root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(path for path in raw_root.rglob("*") if path.is_file()):
        files.append(
            {
                "path": path.relative_to(raw_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "dataset": "UCI Heart Disease",
        "doi": UCI_DOI,
        "source_url": UCI_ARCHIVE_URL,
        "schema_version": SCHEMA_VERSION,
        "files": files,
    }


def prepare_uci_heart(repo_root: Path) -> dict[str, Path]:
    """Create canonical data, profiles, checksums, and immutable split manifests."""
    from heartshift.data.splits import build_inner_manifest, build_outer_manifest

    raw_root = repo_root / "data" / "raw" / "uci_heart" / "doi-10.24432-C52P4X"
    archive = raw_root / "heart-disease.zip"
    extracted = raw_root / "extracted"
    if sha256_file(archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Official UCI Heart archive checksum mismatch")

    data = load_uci_heart(extracted)
    processed_dir = repo_root / "data" / "processed"
    manifests_dir = repo_root / "data" / "manifests"
    splits_dir = repo_root / "data" / "splits"
    for directory in (processed_dir, manifests_dir, splits_dir):
        directory.mkdir(parents=True, exist_ok=True)

    canonical_path = processed_dir / "uci_heart_canonical_v1.parquet"
    profile_path = processed_dir / "uci_heart_profile_v1.json"
    raw_manifest_path = manifests_dir / "uci_heart_raw_manifest_v1.json"
    outer_path = splits_dir / "uci_heart_outer_loho_v1.parquet"
    inner_path = splits_dir / "uci_heart_inner_loho_v1.parquet"

    data.to_parquet(canonical_path, index=False)
    profile = build_profile(data)
    profile["canonical_sha256"] = sha256_file(canonical_path)
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_manifest_path.write_text(
        json.dumps(_raw_manifest(raw_root), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    build_outer_manifest(data).to_parquet(outer_path, index=False)
    build_inner_manifest(data).to_parquet(inner_path, index=False)

    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "outer_manifest": {"path": outer_path.as_posix(), "sha256": sha256_file(outer_path)},
        "inner_manifest": {"path": inner_path.as_posix(), "sha256": sha256_file(inner_path)},
        "rule": "outer leave-one-hospital-out; inner leave-one-source-hospital-out",
    }
    split_json = splits_dir / "uci_heart_loho_v1.json"
    split_json.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "canonical": canonical_path,
        "profile": profile_path,
        "raw_manifest": raw_manifest_path,
        "outer_manifest": outer_path,
        "inner_manifest": inner_path,
        "split_manifest": split_json,
    }
