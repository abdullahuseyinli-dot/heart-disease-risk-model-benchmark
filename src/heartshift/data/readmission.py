"""Canonical parsing and leakage-audited splits for UCI diabetes readmission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from heartshift.data.uci import sha256_file

READMISSION_ARCHIVE_SHA256 = "F82AC129DA2DDD2299391FF6FBAE3A6A58B3EDCF59AC9D7BD480C00FE453112A"
READMISSION_DOI = "10.24432/C5230J"
TABLESHIFT_COMMIT = "fca9429814703a07e3902d005d46563a207b7f0a"
TABLESHIFT_RANDOM_STATE = 264738
OOD_ADMISSION_SOURCE = 7

READMISSION_CONTINUOUS_COLUMNS = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)
READMISSION_ID_COLUMNS = ("encounter_id", "patient_nbr")
READMISSION_EXCLUDED_MODEL_COLUMNS = (*READMISSION_ID_COLUMNS, "admission_source_id", "readmitted")


def _line_hashes(csv_path: Path) -> list[str]:
    with csv_path.open("rb") as handle:
        next(handle)
        return [hashlib.sha256(line.rstrip(b"\r\n")).hexdigest() for line in handle]


def load_readmission(csv_path: Path) -> pd.DataFrame:
    """Parse the untouched UCI CSV while preserving row and patient provenance."""
    frame = pd.read_csv(csv_path, na_values="?", low_memory=False)
    if len(frame) != 101_766 or frame["encounter_id"].nunique() != 101_766:
        raise AssertionError("Unexpected UCI diabetes readmission row/encounter count")
    line_hashes = _line_hashes(csv_path)
    if len(line_hashes) != len(frame):
        raise AssertionError("Raw line hashes do not align with parsed encounters")
    frame.insert(0, "sample_id", "uci296:" + frame["encounter_id"].astype(str))
    frame["source_row"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    frame["source_line_sha256"] = line_hashes
    frame["source_file_sha256"] = sha256_file(csv_path)
    frame["target_tableshift_any_readmission"] = frame["readmitted"].ne("NO").astype("int8")
    frame["target_30d_readmission"] = frame["readmitted"].eq("<30").astype("int8")
    frame["admission_source_id"] = frame["admission_source_id"].astype("int16")
    frame["schema_version"] = "uci-readmission-canonical-v1"
    return frame


def readmission_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    metadata = {
        "sample_id",
        "source_row",
        "source_line_sha256",
        "source_file_sha256",
        "schema_version",
        "target_tableshift_any_readmission",
        "target_30d_readmission",
    }
    return tuple(
        column
        for column in frame.columns
        if column not in metadata and column not in READMISSION_EXCLUDED_MODEL_COLUMNS
    )


def readmission_categorical_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    features = readmission_feature_columns(frame)
    return tuple(column for column in features if column not in READMISSION_CONTINUOUS_COLUMNS)


def tableshift_reproduction_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce TableShift's encounter-level split, including its patient overlap."""
    cohort = frame.loc[frame["race"].notna()].reset_index(drop=True)
    domain = cohort["admission_source_id"]
    ood_indices = np.flatnonzero(domain.eq(OOD_ADMISSION_SOURCE).to_numpy())
    id_indices = np.flatnonzero(domain.ne(OOD_ADMISSION_SOURCE).to_numpy())
    train_indices, id_valid_test = train_test_split(
        id_indices,
        test_size=0.2,
        random_state=TABLESHIFT_RANDOM_STATE,
    )
    validation_indices, id_test_indices = train_test_split(
        id_valid_test,
        test_size=0.5,
        random_state=TABLESHIFT_RANDOM_STATE,
    )
    ood_test_indices, ood_validation_indices = train_test_split(
        ood_indices,
        test_size=0.1,
        random_state=TABLESHIFT_RANDOM_STATE,
    )
    roles = np.full(len(cohort), "unassigned", dtype=object)
    for indices, role in (
        (train_indices, "train"),
        (validation_indices, "validation"),
        (id_test_indices, "id_test"),
        (ood_validation_indices, "ood_validation"),
        (ood_test_indices, "ood_test"),
    ):
        roles[indices] = role
    result = cohort.loc[
        :, ["sample_id", "encounter_id", "patient_nbr", "admission_source_id"]
    ].copy()
    result["split"] = roles
    result["protocol"] = "tableshift_reproduction_fca9429"
    result["target_definition"] = "readmitted != NO"
    return result


def _stratified_patient_partition(
    patient_labels: pd.Series,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    patients = patient_labels.index.to_numpy()
    labels = patient_labels.to_numpy()
    training, testing = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return np.asarray(training), np.asarray(testing)


def patient_grouped_shift_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a patient-disjoint source/ER-target protocol with overlap quarantine."""
    cohort = frame.loc[frame["race"].notna()].copy()
    ood_rows = cohort["admission_source_id"].eq(OOD_ADMISSION_SOURCE)
    target_patients = set(cohort.loc[ood_rows, "patient_nbr"])
    cross_domain = cohort["patient_nbr"].isin(target_patients) & ~ood_rows
    eligible_source = ~ood_rows & ~cross_domain

    source_patient_labels = (
        cohort.loc[eligible_source]
        .groupby("patient_nbr")["target_tableshift_any_readmission"]
        .max()
    )
    source_train_patients, source_holdout_patients = _stratified_patient_partition(
        source_patient_labels,
        test_size=0.2,
        seed=TABLESHIFT_RANDOM_STATE,
    )
    holdout_labels = source_patient_labels.loc[source_holdout_patients]
    validation_patients, id_test_patients = _stratified_patient_partition(
        holdout_labels,
        test_size=0.5,
        seed=TABLESHIFT_RANDOM_STATE,
    )
    target_patient_labels = (
        cohort.loc[ood_rows].groupby("patient_nbr")["target_tableshift_any_readmission"].max()
    )
    ood_test_patients, ood_validation_patients = _stratified_patient_partition(
        target_patient_labels,
        test_size=0.1,
        seed=TABLESHIFT_RANDOM_STATE,
    )

    roles = pd.Series("quarantined_cross_domain", index=cohort.index, dtype="string")
    source_patient = cohort["patient_nbr"]
    roles.loc[eligible_source & source_patient.isin(source_train_patients)] = "train"
    roles.loc[eligible_source & source_patient.isin(validation_patients)] = "validation"
    roles.loc[eligible_source & source_patient.isin(id_test_patients)] = "id_test"
    roles.loc[ood_rows & source_patient.isin(ood_validation_patients)] = "ood_validation"
    roles.loc[ood_rows & source_patient.isin(ood_test_patients)] = "ood_test"
    result = cohort.loc[
        :, ["sample_id", "encounter_id", "patient_nbr", "admission_source_id"]
    ].copy()
    result["split"] = roles
    result["protocol"] = "patient_grouped_admission_source_shift_v1"
    result["target_definition"] = "readmitted != NO"

    evaluated = result.loc[result["split"].ne("quarantined_cross_domain")]
    patient_sets = {
        role: set(group["patient_nbr"]) for role, group in evaluated.groupby("split", observed=True)
    }
    roles_to_check = sorted(patient_sets)
    for index, first in enumerate(roles_to_check):
        for second in roles_to_check[index + 1 :]:
            if patient_sets[first] & patient_sets[second]:
                raise AssertionError(f"Patient leakage between {first} and {second}")
    return result


def split_profile(frame: pd.DataFrame, split: pd.DataFrame) -> dict[str, Any]:
    joined = split.merge(
        frame[
            [
                "sample_id",
                "target_tableshift_any_readmission",
                "target_30d_readmission",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    records = []
    for role, group in joined.groupby("split", observed=True):
        records.append(
            {
                "split": str(role),
                "encounters": len(group),
                "patients": int(group["patient_nbr"].nunique()),
                "any_readmission_prevalence": float(
                    group["target_tableshift_any_readmission"].mean()
                ),
                "readmission_30d_prevalence": float(group["target_30d_readmission"].mean()),
            }
        )
    evaluated = joined.loc[~joined["split"].astype(str).str.startswith("quarantined")]
    patient_sets = {
        str(role): set(group["patient_nbr"])
        for role, group in evaluated.groupby("split", observed=True)
    }
    overlaps = []
    roles = sorted(patient_sets)
    for index, first in enumerate(roles):
        for second in roles[index + 1 :]:
            count = len(patient_sets[first] & patient_sets[second])
            if count:
                overlaps.append({"first": first, "second": second, "patients": count})
    return {
        "protocol": str(split["protocol"].iloc[0]),
        "splits": records,
        "evaluated_patient_overlaps": overlaps,
    }


def prepare_readmission_dataset(
    repo_root: Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    archive = repo_root / config["raw_archive"]
    csv_path = repo_root / config["raw_csv"]
    if sha256_file(archive) != READMISSION_ARCHIVE_SHA256:
        raise AssertionError("UCI diabetes readmission archive checksum failed")
    frame = load_readmission(csv_path)
    table_split = tableshift_reproduction_split(frame)
    patient_split = patient_grouped_shift_split(frame)
    outputs = {key: repo_root / value for key, value in config["outputs"].items()}
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(outputs["canonical"], index=False)
    table_split.to_parquet(outputs["tableshift_split"], index=False)
    patient_split.to_parquet(outputs["patient_split"], index=False)
    profile = {
        "doi": READMISSION_DOI,
        "archive_sha256": READMISSION_ARCHIVE_SHA256,
        "csv_sha256": sha256_file(csv_path),
        "canonical_sha256": sha256_file(outputs["canonical"]),
        "rows": len(frame),
        "patients": int(frame["patient_nbr"].nunique()),
        "features": list(readmission_feature_columns(frame)),
        "continuous_features": list(READMISSION_CONTINUOUS_COLUMNS),
        "categorical_features": list(readmission_categorical_columns(frame)),
        "tableshift_commit": TABLESHIFT_COMMIT,
        "tableshift": split_profile(frame, table_split),
        "patient_grouped": split_profile(frame, patient_split),
    }
    outputs["profile"].write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
