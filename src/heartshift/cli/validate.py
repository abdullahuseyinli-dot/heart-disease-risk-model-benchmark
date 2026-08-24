"""Validate HeartShift raw evidence, canonical data, and split isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from heartshift.config import load_yaml
from heartshift.data.readmission import READMISSION_ARCHIVE_SHA256
from heartshift.data.splits import validate_inner_manifest, validate_outer_manifest
from heartshift.data.uci import EXPECTED_ARCHIVE_SHA256, sha256_file
from heartshift.research.artifact_audit import validate_source_inner_artifact_audit
from heartshift.research.study_contract import (
    validate_historical_seed_extension_contract,
    validate_router_development_contract,
    validate_shiftguard_study_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def validate_repository(repo_root: Path) -> None:
    archive = repo_root / "data/raw/uci_heart/doi-10.24432-C52P4X/heart-disease.zip"
    canonical = repo_root / "data/processed/uci_heart_canonical_v1.parquet"
    profile_path = repo_root / "data/processed/uci_heart_profile_v1.json"
    outer_path = repo_root / "data/splits/uci_heart_outer_loho_v1.parquet"
    inner_path = repo_root / "data/splits/uci_heart_inner_loho_v1.parquet"
    if sha256_file(archive) != EXPECTED_ARCHIVE_SHA256:
        raise AssertionError("Raw UCI archive checksum failed")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if sha256_file(canonical) != profile["canonical_sha256"]:
        raise AssertionError("Canonical table checksum failed")
    data = pd.read_parquet(canonical)
    if len(data) != 920 or data["sample_id"].nunique() != 920:
        raise AssertionError("Canonical row or sample-ID invariant failed")
    if set(data["site"]) != {"cleveland", "hungary", "switzerland", "va_long_beach"}:
        raise AssertionError("Hospital provenance invariant failed")
    validate_outer_manifest(data, pd.read_parquet(outer_path))
    validate_inner_manifest(data, pd.read_parquet(inner_path))

    readmission_archive = (
        repo_root
        / "data/raw/uci_diabetes_readmission/doi-10.24432-C5230J/diabetes-130-us-hospitals.zip"
    )
    readmission_canonical = repo_root / "data/processed/uci_readmission_canonical_v1.parquet"
    readmission_profile_path = repo_root / "data/processed/uci_readmission_profile_v1.json"
    readmission_split_path = repo_root / "data/splits/uci_readmission_patient_grouped_v1.parquet"
    if sha256_file(readmission_archive) != READMISSION_ARCHIVE_SHA256:
        raise AssertionError("Raw UCI readmission archive checksum failed")
    readmission_profile = json.loads(readmission_profile_path.read_text(encoding="utf-8"))
    if sha256_file(readmission_canonical) != readmission_profile["canonical_sha256"]:
        raise AssertionError("Readmission canonical table checksum failed")
    readmission = pd.read_parquet(
        readmission_canonical,
        columns=["sample_id", "patient_nbr"],
    )
    if len(readmission) != 101_766 or readmission["sample_id"].nunique() != 101_766:
        raise AssertionError("Readmission row or sample-ID invariant failed")
    readmission_split = pd.read_parquet(readmission_split_path)
    evaluated = readmission_split.loc[readmission_split["split"].ne("quarantined_cross_domain")]
    patient_sets = {
        str(role): set(group["patient_nbr"])
        for role, group in evaluated.groupby("split", observed=True)
    }
    roles = sorted(patient_sets)
    if any(
        patient_sets[first] & patient_sets[second]
        for index, first in enumerate(roles)
        for second in roles[index + 1 :]
    ):
        raise AssertionError("Readmission patient-disjoint split invariant failed")

    shiftguard_config = repo_root / "configs/research/shiftguard_v1.yaml"
    if shiftguard_config.is_file():
        validate_shiftguard_study_config(repo_root, load_yaml(shiftguard_config))

    router_configs = {
        "control": repo_root / "configs/research/heart_controls_v1.yaml",
        "neural_inner": repo_root / "configs/research/observed_backbones_v1.yaml",
        "neural_outer": repo_root / "configs/research/observed_backbones_outer_v1.yaml",
        "router": repo_root / "configs/research/support_router_outer_v3.yaml",
        "report": repo_root / "configs/reporting/heart_research_development_v1.yaml",
    }
    if all(path.is_file() for path in router_configs.values()):
        validate_router_development_contract(*(load_yaml(path) for path in router_configs.values()))
        for superseded_name in ("support_router_v1.yaml", "support_router_v2.yaml"):
            superseded = load_yaml(repo_root / "configs/research" / superseded_name)
            if not str(superseded.get("status", "")).startswith("superseded_before_full_run"):
                raise AssertionError(f"Unsafe source-meta router remains active: {superseded_name}")

    seed_extension_configs = {
        "control": repo_root / "configs/research/heart_controls_v1.yaml",
        "extension": (repo_root / "configs/research/historical_psmask_seed_extension_v1.yaml"),
        "report": (repo_root / "configs/reporting/historical_psmask_seed_extension_v1.yaml"),
    }
    if all(path.is_file() for path in seed_extension_configs.values()):
        validate_historical_seed_extension_contract(
            *(load_yaml(path) for path in seed_extension_configs.values())
        )

    neural_inner_run = repo_root / "artifacts/runs/observed-backbones-development-v1"
    if (neural_inner_run / "evidence_audit.json").is_file():
        validate_source_inner_artifact_audit(neural_inner_run)


def main() -> None:
    args = build_parser().parse_args()
    validate_repository(args.repo_root.resolve())
    print("HeartShift evidence validation passed.")


if __name__ == "__main__":
    main()
