from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heartshift.reporting.outer_report import (
    _normalise_classical,
    cell_metrics,
    paired_primary_bootstrap,
    primary_estimands,
)
from heartshift.reporting.readmission_report import (
    paired_readmission_bootstrap,
    readmission_cell_metrics,
    readmission_primary_estimands,
)


def _paired_predictions() -> pd.DataFrame:
    records = []
    policies = {"natural": 0.0, "mcar_10": 0.1, "mcar_30": 0.3, "mcar_50": 0.5}
    for site in ("a", "b"):
        for index in range(20):
            target = index % 2
            for policy, rate in policies.items():
                for replicate in range(2):
                    for method, positive_score in (("reference", 0.7), ("candidate", 0.8)):
                        score = positive_score if target else 1.0 - positive_score
                        records.append(
                            {
                                "sample_id": f"{site}-{index}",
                                "outer_target": site,
                                "target": target,
                                "policy": policy,
                                "mask_replicate": replicate,
                                "observed_fraction": 1.0 - rate,
                                "method": method,
                                "track": "dg_zero_shot",
                                "score": score,
                            }
                        )
    return pd.DataFrame(records)


def test_primary_estimands_and_paired_bootstrap_are_prediction_derived() -> None:
    predictions = _paired_predictions()
    metrics = cell_metrics(predictions)
    primary = primary_estimands(metrics).set_index("method")
    replicates, intervals = paired_primary_bootstrap(
        predictions,
        reference_method="reference",
        comparison_methods=["candidate"],
        repetitions=20,
        seed=4,
    )
    assert (
        primary.loc["candidate", "macro_site_worst_mask_balanced_log_loss"]
        < primary.loc["reference", "macro_site_worst_mask_balanced_log_loss"]
    )
    assert len(replicates) == 20
    assert np.isfinite(replicates["difference_candidate_minus_reference"]).all()
    assert float(intervals.loc[0, "ci_975"]) < 0.0


def test_classical_calibration_track_is_a_separate_method(tmp_path: Path) -> None:
    path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a", "a"],
            "outer_target": ["site", "site"],
            "target": [1, 1],
            "policy": ["natural", "natural"],
            "mask_replicate": [0, 0],
            "observed_fraction": [1.0, 1.0],
            "model": ["tabm", "tabm"],
            "weighting": ["pooled", "pooled"],
            "calibration": ["raw", "source_oof_platt"],
            "y_score": [0.9, 0.8],
        }
    ).to_parquet(path, index=False)
    normalized = _normalise_classical(path, "modern")
    assert set(normalized["method"]) == {
        "modern:tabm:pooled",
        "modern:tabm:pooled:source_oof_platt",
    }


def test_readmission_report_uses_patient_cluster_bootstrap() -> None:
    records = []
    for split in ("id_test", "ood_test"):
        for patient in range(20):
            target = patient % 2
            for encounter in range(2):
                for policy in ("natural", "mcar_50"):
                    for replicate in range(2):
                        for experiment, positive_score in (
                            ("reference", 0.7),
                            ("candidate", 0.8),
                        ):
                            records.append(
                                {
                                    "sample_id": f"{split}-{patient}-{encounter}",
                                    "patient_nbr": patient,
                                    "target": target,
                                    "experiment": experiment,
                                    "backend": "neural",
                                    "variant": "test",
                                    "model": "test",
                                    "weighting": "test",
                                    "split": split,
                                    "policy": policy,
                                    "mask_replicate": replicate,
                                    "y_score": (positive_score if target else 1.0 - positive_score),
                                }
                            )
    predictions = pd.DataFrame(records)
    metrics = readmission_cell_metrics(predictions)
    primary = readmission_primary_estimands(metrics).set_index("experiment")
    replicates, intervals = paired_readmission_bootstrap(
        predictions,
        reference_experiment="reference",
        comparison_experiments=["candidate"],
        repetitions=20,
        seed=8,
    )
    assert (
        primary.loc["candidate", "ood_worst_mask_balanced_log_loss"]
        < primary.loc["reference", "ood_worst_mask_balanced_log_loss"]
    )
    assert len(replicates) == 20
    assert float(intervals.loc[0, "ci_975"]) < 0.0
