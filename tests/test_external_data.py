from __future__ import annotations

from pathlib import Path

import pandas as pd

from heartshift.data.external import (
    ExternalDatasetContract,
    build_hospital_disjoint_manifest,
    load_eicu_demo_relational_table,
    validate_hospital_disjoint_manifest,
)


def _contract() -> ExternalDatasetContract:
    return ExternalDatasetContract.from_mapping(
        {
            "name": "test",
            "raw_path": "unused.csv",
            "encounter_id_column": "encounter",
            "patient_id_column": "patient",
            "hospital_id_column": "hospital",
            "target_column": "outcome",
            "continuous_columns": ["age"],
            "categorical_columns": ["sex"],
            "minimum_hospital_rows": 2,
            "minimum_class_rows": 1,
            "hospital_split": {
                "salt": "unit-test",
                "development_fraction": 0.34,
                "selection_fraction": 0.33,
                "confirmation_fraction": 0.33,
            },
        }
    )


def test_hospital_split_is_deterministic_and_patient_disjoint() -> None:
    contract = _contract()
    records = []
    # Supply enough hospitals that the deterministic hash covers every role.
    for hospital in range(60):
        for row in range(4):
            records.append(
                {
                    "encounter": f"e-{hospital}-{row}",
                    "patient": f"p-{hospital}-{row}",
                    "hospital": f"h-{hospital}",
                    "outcome": row % 2,
                    "age": 50 + row,
                    "sex": row % 2,
                }
            )
    frame = pd.DataFrame(records)
    first = build_hospital_disjoint_manifest(frame, contract)
    second = build_hospital_disjoint_manifest(frame, contract)
    assert first.equals(second)
    validation = validate_hospital_disjoint_manifest(first, contract)
    assert validation["status"] == "passed_hospital_and_patient_disjoint_validation"
    assert {row["split"] for row in validation["splits"]} == {
        "development",
        "architecture_selection",
        "locked_confirmation",
    }


def test_cross_role_patient_is_quarantined_without_moving_hospitals() -> None:
    contract = _contract()
    records = []
    roles: dict[str, str] = {}
    hospital = 0
    while len(set(roles.values())) < 3:
        name = f"h-{hospital}"
        probe = pd.DataFrame(
            [
                {
                    "encounter": f"probe-{hospital}-{row}",
                    "patient": f"probe-p-{hospital}-{row}",
                    "hospital": name,
                    "outcome": row % 2,
                }
                for row in range(2)
            ]
        )
        manifest = build_hospital_disjoint_manifest(probe, contract)
        roles[name] = str(manifest.loc[0, "split"])
        hospital += 1
    active: dict[str, str] = {}
    for name, role in roles.items():
        if role in {"development", "architecture_selection", "locked_confirmation"}:
            active.setdefault(role, name)
    for role, name in active.items():
        for row in range(2):
            records.append(
                {
                    "encounter": f"{role}-{row}",
                    "patient": (
                        "shared"
                        if row == 0 and role != "architecture_selection"
                        else f"{role}-{row}"
                    ),
                    "hospital": name,
                    "outcome": row % 2,
                }
            )
    manifest = build_hospital_disjoint_manifest(pd.DataFrame(records), contract)
    shared = manifest.loc[manifest["patient_id"].eq("shared")]
    assert shared["split"].eq("quarantined_cross_role_patient").all()
    assert shared["quarantine_reason"].eq("patient_crosses_hospital_split_roles").all()


def test_eicu_demo_join_selects_iva_and_preserves_missingness(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "patientunitstayid": [1],
            "uniquepid": ["p1"],
            "hospitalid": [10],
            "age": ["> 89"],
            "gender": ["Female"],
            "ethnicity": ["Caucasian"],
            "admissionheight": [160.0],
            "admissionweight": [60.0],
            "hospitaladmitsource": ["ED"],
            "unittype": ["ICU"],
            "unitadmitsource": ["ED"],
            "unitstaytype": ["admit"],
        }
    ).to_csv(tmp_path / "patient.csv.gz", index=False, compression="gzip")
    physiology = {
        "patientunitstayid": [1],
        "intubated": [0],
        "vent": [0],
        "dialysis": [0],
        "eyes": [4],
        "motor": [6],
        "verbal": [5],
        "urine": [-1.0],
        "wbc": [10.0],
        "temperature": [36.5],
        "respiratoryrate": [20],
        "sodium": [140.0],
        "heartrate": [80],
        "meanbp": [75],
        "ph": [-1.0],
        "hematocrit": [40.0],
        "creatinine": [1.0],
        "albumin": [-1.0],
        "pao2": [-1.0],
        "pco2": [-1.0],
        "bun": [20.0],
        "glucose": [100],
        "bilirubin": [-1.0],
        "fio2": [-1],
    }
    pd.DataFrame(physiology).to_csv(
        tmp_path / "apacheApsVar.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(
        {
            "apachepatientresultsid": [2, 1],
            "patientunitstayid": [1, 1],
            "apacheversion": ["IV", "IVa"],
            "actualhospitalmortality": ["EXPIRED", "EXPIRED"],
        }
    ).to_csv(tmp_path / "apachePatientResult.csv.gz", index=False, compression="gzip")
    result = load_eicu_demo_relational_table(tmp_path)
    assert len(result) == 1
    assert result.loc[result.index[0], "apacheversion"] == "IVa"
    assert result.loc[result.index[0], "age"] == 90
    assert pd.isna(result.loc[result.index[0], "urine"])
    assert result.loc[result.index[0], "hospital_death"] == 1
