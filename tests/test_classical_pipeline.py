from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heartshift.data.uci import FEATURE_COLUMNS, load_uci_heart
from heartshift.evaluation.classical_benchmark import _fit_source_oof_calibrators
from heartshift.models.classical import (
    build_classical_pipeline,
    build_generic_classical_pipeline,
    group_sample_weights,
    sample_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = REPO_ROOT / "data/raw/uci_heart/doi-10.24432-C52P4X/extracted"


def test_fold_pipeline_predicts_without_nan_or_row_reordering() -> None:
    data = load_uci_heart(EXTRACTED)
    train = data.loc[data["site"].isin(["cleveland", "hungary"])]
    validation = data.loc[data["site"].eq("switzerland")]
    pipeline = build_classical_pipeline("logistic", {"C": 1.0, "l1_ratio": 0.0}, 7)
    pipeline.fit(
        train.loc[:, FEATURE_COLUMNS],
        train["target"],
        model__sample_weight=sample_weights(train, "site_class_balanced"),
    )
    probabilities = pipeline.predict_proba(validation.loc[:, FEATURE_COLUMNS])[:, 1]
    assert len(probabilities) == len(validation)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_site_class_weights_equalize_each_site_class_cell() -> None:
    data = load_uci_heart(EXTRACTED)
    source = data.loc[data["site"].isin(["cleveland", "hungary", "switzerland"])]
    weights = sample_weights(source, "site_class_balanced")
    totals = {}
    for site in source["site"].unique():
        for label in (0, 1):
            selected = source["site"].eq(site) & source["target"].eq(label)
            totals[(site, label)] = float(weights[selected].sum())
    assert max(totals.values()) - min(totals.values()) < 1e-9


def test_generic_pipeline_and_environment_class_weights() -> None:
    frame = pd.DataFrame(
        {
            "numeric": [0.0, 1.0, np.nan, 2.0, 3.0, 4.0, 5.0, np.nan],
            "category": ["a", "a", "b", None, "a", "b", "c", "c"],
            "environment": ["x", "x", "x", "x", "y", "y", "y", "y"],
            "target": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    weights = group_sample_weights(frame, "environment_class_balanced", group_column="environment")
    totals = [
        float(weights[frame["environment"].eq(environment) & frame["target"].eq(label)].sum())
        for environment in ("x", "y")
        for label in (0, 1)
    ]
    pipeline = build_generic_classical_pipeline(
        "logistic",
        {"C": 1.0},
        11,
        feature_columns=("numeric", "category"),
        continuous_columns=("numeric",),
        categorical_columns=("category",),
    )
    pipeline.fit(
        frame.loc[:, ["numeric", "category"]],
        frame["target"],
        model__sample_weight=weights,
    )
    probabilities = pipeline.predict_proba(frame.loc[:, ["numeric", "category"]])[:, 1]
    assert max(totals) - min(totals) < 1e-12
    assert np.isfinite(probabilities).all()


def test_source_oof_calibrator_uses_only_hospital_held_out_rows() -> None:
    rows = []
    for site in ("a", "b"):
        for index in range(8):
            target = index % 2
            rows.append(
                {
                    "sample_id": f"{site}-{index}",
                    "outer_target": "locked",
                    "inner_validation": site,
                    "site": site,
                    "target": target,
                    "model": "test_model",
                    "weighting": "pooled",
                    "parameters_json": "{}",
                    "seed": 3,
                    "y_score": 0.8 if target else 0.2,
                }
            )
    predictions = pd.DataFrame(rows)
    calibrators, records = _fit_source_oof_calibrators(
        predictions,
        outer_target="locked",
        model_name="test_model",
        weighting="pooled",
        parameters_json="{}",
        seeds=(3,),
    )
    assert set(calibrators) == {3}
    assert calibrators[3].slope > 0.0
    assert records[0]["source_sites"] == 2
    assert records[0]["source_samples"] == 16

    contaminated = predictions.copy()
    contaminated.loc[0, ["site", "inner_validation"]] = "locked"
    with pytest.raises(AssertionError, match="Outer-target"):
        _fit_source_oof_calibrators(
            contaminated,
            outer_target="locked",
            model_name="test_model",
            weighting="pooled",
            parameters_json="{}",
            seeds=(3,),
        )

    duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="duplicated"):
        _fit_source_oof_calibrators(
            duplicated,
            outer_target="locked",
            model_name="test_model",
            weighting="pooled",
            parameters_json="{}",
            seeds=(3,),
        )
