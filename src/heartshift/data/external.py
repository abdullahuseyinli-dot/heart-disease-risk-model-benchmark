"""Contracts and deterministic hospital-disjoint splits for external confirmation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heartshift.data.uci import sha256_file

EICU_DEMO_FILES = (
    "patient.csv.gz",
    "apacheApsVar.csv.gz",
    "apachePatientResult.csv.gz",
)
EICU_APACHE_MISSING_SENTINEL = -1


@dataclass(frozen=True)
class ExternalDatasetContract:
    name: str
    raw_path: str
    encounter_id_column: str
    patient_id_column: str
    hospital_id_column: str
    target_column: str
    continuous_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    split_salt: str
    development_fraction: float
    selection_fraction: float
    confirmation_fraction: float
    minimum_hospital_rows: int
    minimum_class_rows: int

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.continuous_columns + self.categorical_columns

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ExternalDatasetContract:
        split = payload["hospital_split"]
        contract = cls(
            name=str(payload["name"]),
            raw_path=str(payload["raw_path"]),
            encounter_id_column=str(payload["encounter_id_column"]),
            patient_id_column=str(payload["patient_id_column"]),
            hospital_id_column=str(payload["hospital_id_column"]),
            target_column=str(payload["target_column"]),
            continuous_columns=tuple(str(value) for value in payload["continuous_columns"]),
            categorical_columns=tuple(str(value) for value in payload["categorical_columns"]),
            split_salt=str(split["salt"]),
            development_fraction=float(split["development_fraction"]),
            selection_fraction=float(split["selection_fraction"]),
            confirmation_fraction=float(split["confirmation_fraction"]),
            minimum_hospital_rows=int(payload.get("minimum_hospital_rows", 1)),
            minimum_class_rows=int(payload.get("minimum_class_rows", 1)),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        identifiers = {
            self.encounter_id_column,
            self.patient_id_column,
            self.hospital_id_column,
            self.target_column,
        }
        if len(identifiers) != 4:
            raise ValueError("External identifier and endpoint columns must be distinct")
        if set(self.continuous_columns) & set(self.categorical_columns):
            raise ValueError("External continuous and categorical features overlap")
        if len(set(self.feature_columns)) != len(self.feature_columns) or not self.feature_columns:
            raise ValueError("External feature schema is empty or duplicated")
        if identifiers & set(self.feature_columns):
            raise ValueError(
                "Hospital, patient, encounter, and endpoint columns cannot be features"
            )
        fractions = (
            self.development_fraction,
            self.selection_fraction,
            self.confirmation_fraction,
        )
        if any(value <= 0.0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
            raise ValueError("Hospital split fractions must be positive and sum to one")
        if self.minimum_hospital_rows < 1 or self.minimum_class_rows < 1:
            raise ValueError("External hospital eligibility thresholds must be positive")


def _hospital_role(contract: ExternalDatasetContract, hospital_id: Any) -> str:
    digest = hashlib.sha256(
        f"{contract.split_salt}|{hospital_id}".encode()
    ).digest()
    uniform = int.from_bytes(digest[:8], "big") / float(2**64)
    if uniform < contract.development_fraction:
        return "development"
    if uniform < contract.development_fraction + contract.selection_fraction:
        return "architecture_selection"
    return "locked_confirmation"


def build_hospital_disjoint_manifest(
    frame: pd.DataFrame,
    contract: ExternalDatasetContract,
) -> pd.DataFrame:
    """Create a hash-based hospital split and quarantine cross-role patients."""
    required = {
        contract.encounter_id_column,
        contract.patient_id_column,
        contract.hospital_id_column,
        contract.target_column,
    }
    missing = required - set(frame)
    if missing:
        raise KeyError(f"External split columns are missing: {sorted(missing)}")
    if frame[contract.encounter_id_column].isna().any():
        raise ValueError("External encounter identifiers contain missing values")
    if frame[contract.encounter_id_column].duplicated().any():
        raise ValueError("External encounter identifiers are not unique")
    if frame[[contract.patient_id_column, contract.hospital_id_column]].isna().any().any():
        raise ValueError("External patient or hospital identifiers contain missing values")
    target = frame[contract.target_column]
    if not target.isin([0, 1, False, True]).all():
        raise ValueError("External endpoint must be binary")

    hospital_roles = {
        hospital: _hospital_role(contract, hospital)
        for hospital in frame[contract.hospital_id_column].unique()
    }
    manifest = frame.loc[
        :,
        [
            contract.encounter_id_column,
            contract.patient_id_column,
            contract.hospital_id_column,
            contract.target_column,
        ],
    ].copy()
    manifest.columns = ["encounter_id", "patient_id", "hospital_id", "target"]
    manifest["split"] = manifest["hospital_id"].map(hospital_roles)

    patient_role_count = manifest.groupby("patient_id")["split"].nunique()
    cross_role_patients = set(patient_role_count.loc[patient_role_count.gt(1)].index)
    manifest["quarantine_reason"] = ""
    selected = manifest["patient_id"].isin(cross_role_patients)
    manifest.loc[selected, "quarantine_reason"] = "patient_crosses_hospital_split_roles"
    manifest.loc[selected, "split"] = "quarantined_cross_role_patient"

    eligible = manifest.loc[~selected]
    hospital_profile = eligible.groupby("hospital_id").agg(
        n=("encounter_id", "size"),
        negative=("target", lambda values: int((values == 0).sum())),
        positive=("target", lambda values: int((values == 1).sum())),
    )
    ineligible = set(
        hospital_profile.loc[
            hospital_profile["n"].lt(contract.minimum_hospital_rows)
            | hospital_profile["negative"].lt(contract.minimum_class_rows)
            | hospital_profile["positive"].lt(contract.minimum_class_rows)
        ].index
    )
    selected = manifest["hospital_id"].isin(ineligible) & manifest["quarantine_reason"].eq("")
    manifest.loc[selected, "quarantine_reason"] = "prespecified_hospital_eligibility_failure"
    manifest.loc[selected, "split"] = "quarantined_ineligible_hospital"
    return manifest.sort_values("encounter_id").reset_index(drop=True)


def validate_hospital_disjoint_manifest(
    manifest: pd.DataFrame,
    contract: ExternalDatasetContract,
) -> dict[str, Any]:
    """Validate patient/hospital isolation and report locked class counts."""
    required = {"encounter_id", "patient_id", "hospital_id", "target", "split"}
    if not required <= set(manifest):
        raise KeyError(f"External manifest columns are missing: {sorted(required - set(manifest))}")
    active = manifest.loc[
        manifest["split"].isin(
            ["development", "architecture_selection", "locked_confirmation"]
        )
    ]
    roles = {
        role: group
        for role, group in active.groupby("split", sort=True)
    }
    expected_roles = {"development", "architecture_selection", "locked_confirmation"}
    if set(roles) != expected_roles:
        raise AssertionError(f"External split lacks roles: {sorted(expected_roles - set(roles))}")
    for first_index, first in enumerate(sorted(roles)):
        for second in sorted(roles)[first_index + 1 :]:
            if set(roles[first]["hospital_id"]) & set(roles[second]["hospital_id"]):
                raise AssertionError("A hospital crosses external split roles")
            if set(roles[first]["patient_id"]) & set(roles[second]["patient_id"]):
                raise AssertionError("A patient crosses external split roles")
    if active["encounter_id"].duplicated().any():
        raise AssertionError("Active external encounters are duplicated")
    profile = []
    for role, group in roles.items():
        profile.append(
            {
                "split": role,
                "encounters": len(group),
                "patients": int(group["patient_id"].nunique()),
                "hospitals": int(group["hospital_id"].nunique()),
                "prevalence": float(group["target"].mean()),
                "negative": int(group["target"].eq(0).sum()),
                "positive": int(group["target"].eq(1).sum()),
            }
        )
    return {
        "status": "passed_hospital_and_patient_disjoint_validation",
        "dataset": contract.name,
        "active_encounters": len(active),
        "quarantined_encounters": len(manifest) - len(active),
        "splits": profile,
    }


def prepare_external_dataset(
    repo_root: Path,
    contract: ExternalDatasetContract,
    output_dir: Path,
) -> dict[str, Path]:
    """Canonicalize an authorized CSV/CSV.GZ without silently imputing values."""
    raw_path = Path(contract.raw_path)
    if not raw_path.is_absolute():
        raw_path = repo_root / raw_path
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Authorized external raw file is unavailable: {raw_path}. "
            "Credentialed data must be obtained under its DUA before preparation."
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(raw_path, low_memory=False)
    required = {
        contract.encounter_id_column,
        contract.patient_id_column,
        contract.hospital_id_column,
        contract.target_column,
        *contract.feature_columns,
    }
    missing = required - set(frame)
    if missing:
        raise KeyError(f"External raw file misses contracted columns: {sorted(missing)}")
    selected = frame.loc[:, list(required)].copy()
    selected[contract.target_column] = selected[contract.target_column].astype(np.int8)
    selected["sample_id"] = (
        contract.name + ":" + selected[contract.encounter_id_column].astype(str)
    )
    selected["environment"] = selected[contract.hospital_id_column].astype(str)
    selected["target"] = selected[contract.target_column]
    manifest = build_hospital_disjoint_manifest(selected, contract)
    validation = validate_hospital_disjoint_manifest(manifest, contract)
    canonical_path = output_dir / "canonical.parquet"
    manifest_path = output_dir / "hospital_split.parquet"
    profile_path = output_dir / "profile.json"
    selected.to_parquet(canonical_path, index=False)
    manifest.to_parquet(manifest_path, index=False)
    profile_path.write_text(
        json.dumps(
            {
                **validation,
                "raw_path": str(raw_path.relative_to(repo_root)),
                "raw_sha256": sha256_file(raw_path),
                "canonical_sha256": sha256_file(canonical_path),
                "manifest_sha256": sha256_file(manifest_path),
                "feature_columns": list(contract.feature_columns),
                "hospital_used_only_for_split_and_audit": True,
                "imputation_applied": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "canonical": canonical_path,
        "manifest": manifest_path,
        "profile": profile_path,
    }


def load_eicu_demo_relational_table(raw_directory: Path) -> pd.DataFrame:
    """Build an auditable first-day demo table without using APACHE predictions.

    The public eICU demo is only an execution/schema smoke test.  APACHE IVa is
    selected deterministically when duplicate result versions exist; mortality
    predictions and scores are intentionally excluded because they encode a
    privileged clinical model and are not baseline covariates.
    """
    missing_files = [name for name in EICU_DEMO_FILES if not (raw_directory / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Public eICU demo files are missing: {missing_files}")
    patient = pd.read_csv(raw_directory / "patient.csv.gz", low_memory=False)
    physiology = pd.read_csv(raw_directory / "apacheApsVar.csv.gz", low_memory=False)
    result = pd.read_csv(raw_directory / "apachePatientResult.csv.gz", low_memory=False)
    required_patient = {
        "patientunitstayid",
        "uniquepid",
        "hospitalid",
        "age",
        "gender",
        "ethnicity",
        "admissionheight",
        "admissionweight",
        "hospitaladmitsource",
        "unittype",
        "unitadmitsource",
        "unitstaytype",
    }
    required_physiology = {
        "patientunitstayid",
        "intubated",
        "vent",
        "dialysis",
        "eyes",
        "motor",
        "verbal",
        "urine",
        "wbc",
        "temperature",
        "respiratoryrate",
        "sodium",
        "heartrate",
        "meanbp",
        "ph",
        "hematocrit",
        "creatinine",
        "albumin",
        "pao2",
        "pco2",
        "bun",
        "glucose",
        "bilirubin",
        "fio2",
    }
    required_result = {
        "patientunitstayid",
        "apachepatientresultsid",
        "apacheversion",
        "actualhospitalmortality",
    }
    for name, frame, required in (
        ("patient", patient, required_patient),
        ("apacheApsVar", physiology, required_physiology),
        ("apachePatientResult", result, required_result),
    ):
        missing = required - set(frame)
        if missing:
            raise KeyError(f"eICU demo {name} columns are missing: {sorted(missing)}")
    if patient["patientunitstayid"].duplicated().any():
        raise ValueError("eICU demo patient table duplicates unit stays")
    if physiology["patientunitstayid"].duplicated().any():
        raise ValueError("eICU demo APACHE physiology table duplicates unit stays")
    outcome_consistency = result.groupby("patientunitstayid")[
        "actualhospitalmortality"
    ].nunique(dropna=True)
    if outcome_consistency.gt(1).any():
        raise ValueError("eICU demo APACHE versions disagree on actual hospital mortality")
    result = result.copy()
    result["version_priority"] = result["apacheversion"].map({"IVa": 0, "IV": 1}).fillna(2)
    result = (
        result.sort_values(
            ["patientunitstayid", "version_priority", "apachepatientresultsid"]
        )
        .drop_duplicates("patientunitstayid", keep="first")
        .loc[:, ["patientunitstayid", "actualhospitalmortality", "apacheversion"]]
    )
    joined = patient.merge(physiology, on="patientunitstayid", validate="one_to_one").merge(
        result, on="patientunitstayid", validate="one_to_one"
    )
    joined = joined.loc[
        joined["actualhospitalmortality"].isin(["ALIVE", "EXPIRED"])
    ].copy()
    joined["hospital_death"] = joined["actualhospitalmortality"].eq("EXPIRED").astype("int8")
    joined["age"] = pd.to_numeric(joined["age"].replace({"> 89": "90"}), errors="coerce")
    physiology_features = required_physiology - {"patientunitstayid"}
    for feature in physiology_features:
        # Explicit promotion avoids pandas 3's lossless-assignment guard for
        # integer-valued APACHE columns while preserving every missing sentinel.
        numeric = pd.to_numeric(joined[feature], errors="coerce").astype(float)
        joined[feature] = numeric.mask(numeric.eq(EICU_APACHE_MISSING_SENTINEL))
    return joined


def prepare_eicu_demo_dataset(
    repo_root: Path,
    contract: ExternalDatasetContract,
    output_dir: Path,
) -> dict[str, Path]:
    """Prepare the public demo as a non-scientific pipeline smoke test."""
    raw_directory = Path(contract.raw_path)
    if not raw_directory.is_absolute():
        raw_directory = repo_root / raw_directory
    if not raw_directory.is_dir():
        raise FileNotFoundError(f"Public eICU demo directory is unavailable: {raw_directory}")
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = load_eicu_demo_relational_table(raw_directory)
    required = {
        contract.encounter_id_column,
        contract.patient_id_column,
        contract.hospital_id_column,
        contract.target_column,
        *contract.feature_columns,
    }
    missing = required - set(frame)
    if missing:
        raise KeyError(f"Prepared eICU demo table misses contracted columns: {sorted(missing)}")
    selected = frame.loc[:, list(required)].copy()
    selected["sample_id"] = contract.name + ":" + selected[
        contract.encounter_id_column
    ].astype(str)
    selected["environment"] = selected[contract.hospital_id_column].astype(str)
    selected["target"] = selected[contract.target_column].astype("int8")
    manifest = build_hospital_disjoint_manifest(selected, contract)
    validation = validate_hospital_disjoint_manifest(manifest, contract)
    canonical_path = output_dir / "canonical.parquet"
    manifest_path = output_dir / "hospital_split.parquet"
    profile_path = output_dir / "profile.json"
    selected.to_parquet(canonical_path, index=False)
    manifest.to_parquet(manifest_path, index=False)
    raw_hashes = {name: sha256_file(raw_directory / name) for name in EICU_DEMO_FILES}
    profile_path.write_text(
        json.dumps(
            {
                **validation,
                "status": "passed_public_demo_pipeline_smoke_only",
                "scientific_evaluation_allowed": False,
                "raw_directory": str(raw_directory.relative_to(repo_root)),
                "raw_sha256": raw_hashes,
                "canonical_sha256": sha256_file(canonical_path),
                "manifest_sha256": sha256_file(manifest_path),
                "feature_columns": list(contract.feature_columns),
                "hospital_used_only_for_split_and_audit": True,
                "apache_prediction_columns_excluded": True,
                "apache_minus_one_converted_to_missing": True,
                "age_above_89_encoded_as_90": True,
                "imputation_applied": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"canonical": canonical_path, "manifest": manifest_path, "profile": profile_path}
