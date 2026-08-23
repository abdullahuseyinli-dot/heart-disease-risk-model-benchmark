from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from heartshift.config import load_yaml
from heartshift.data.readmission import (
    patient_grouped_shift_split,
    tableshift_reproduction_split,
)
from heartshift.evaluation.readmission_benchmark import (
    _load_readmission_outer_frames,
    run_readmission_inner,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "data/processed/uci_readmission_canonical_v1.parquet"


def test_readmission_endpoint_definitions_remain_distinct() -> None:
    frame = pd.read_parquet(CANONICAL)
    assert len(frame) == 101_766
    assert frame["sample_id"].is_unique
    assert int(frame["target_tableshift_any_readmission"].sum()) == 46_902
    assert int(frame["target_30d_readmission"].sum()) == 11_357


def test_tableshift_reproduction_exposes_patient_overlap() -> None:
    split = tableshift_reproduction_split(pd.read_parquet(CANONICAL))
    counts = split["split"].value_counts().to_dict()
    assert counts == {
        "ood_test": 50_968,
        "train": 34_288,
        "ood_validation": 5_664,
        "id_test": 4_287,
        "validation": 4_286,
    }
    train_patients = set(split.loc[split["split"].eq("train"), "patient_nbr"])
    ood_patients = set(split.loc[split["split"].eq("ood_test"), "patient_nbr"])
    assert len(train_patients & ood_patients) == 4_983


def test_hardened_readmission_split_is_patient_disjoint() -> None:
    split = patient_grouped_shift_split(pd.read_parquet(CANONICAL))
    evaluated = split.loc[split["split"].ne("quarantined_cross_domain")]
    patient_sets = {
        role: set(group["patient_nbr"]) for role, group in evaluated.groupby("split", observed=True)
    }
    roles = sorted(patient_sets)
    assert all(
        not patient_sets[first] & patient_sets[second]
        for index, first in enumerate(roles)
        for second in roles[index + 1 :]
    )
    assert int(split["split"].eq("quarantined_cross_domain").sum()) == 8_746


def test_outer_loader_excludes_locked_endpoint_until_scoring() -> None:
    config = load_yaml(REPO_ROOT / "configs/independent/readmission_outer_v1.yaml")
    source, target_unlabelled, _, label_locator = _load_readmission_outer_frames(REPO_ROOT, config)
    assert len(source) == 27_341 + 3_433
    assert len(target_unlabelled) == 3_341 + 50_946
    assert "target_tableshift_any_readmission" not in target_unlabelled
    assert target_unlabelled["target"].eq(0).all()
    assert set(label_locator["split"]) == {"id_test", "ood_test"}


def test_readmission_classical_inner_runner_writes_prediction_evidence(tmp_path: Path) -> None:
    records = []
    for split_name, n_rows in (("train", 40), ("validation", 20)):
        for index in range(n_rows):
            target = index % 2
            records.append(
                {
                    "sample_id": f"{split_name}-{index}",
                    "patient_nbr": index,
                    "admission_source_id": 1 + index % 2,
                    "source_line_sha256": f"hash-{split_name}-{index}",
                    "numeric": float(index) if index % 5 else np.nan,
                    "category": "positive" if target else "negative",
                    "endpoint": target,
                }
            )
    frame = pd.DataFrame(records)
    split = frame.loc[:, ["sample_id"]].copy()
    split["split"] = ["train"] * 40 + ["validation"] * 20
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame.to_parquet(data_dir / "canonical.parquet", index=False)
    split.to_parquet(data_dir / "split.parquet", index=False)
    (data_dir / "profile.json").write_text(
        json.dumps(
            {
                "features": ["numeric", "category"],
                "continuous_features": ["numeric"],
                "categorical_features": ["category"],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "device": "cpu",
        "seeds": [3],
        "data": {
            "canonical_path": "data/canonical.parquet",
            "profile_path": "data/profile.json",
            "split_path": "data/split.parquet",
            "target": "endpoint",
            "source_splits": ["train"],
            "validation_split": "validation",
        },
        "mask_replicates": 1,
        "mask_policies": [
            {"name": "natural", "kind": "natural"},
            {"name": "mcar_30", "kind": "mcar", "rate": 0.3},
        ],
        "common_parameters": {},
        "experiments": [
            {
                "name": "logistic",
                "backend": "classical",
                "model": "logistic",
                "weighting": "environment_class_balanced",
                "grid": {"C": [1.0]},
            }
        ],
    }
    outputs = run_readmission_inner(tmp_path, config, tmp_path / "run")
    predictions = pd.read_parquet(outputs["predictions"])
    selections = pd.read_csv(outputs["selections"])
    assert set(predictions["policy"]) == {"natural", "mcar_30"}
    assert predictions["source_line_sha256"].notna().all()
    assert selections["experiment"].tolist() == ["logistic"]
