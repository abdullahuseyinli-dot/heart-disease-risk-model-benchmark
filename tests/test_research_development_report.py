from __future__ import annotations

import pandas as pd
import pytest

from heartshift.reporting.research_development import _audit_paired_predictions


def _paired_predictions() -> pd.DataFrame:
    records = []
    for method in ("a", "b"):
        for index, target in enumerate((0, 1)):
            records.append(
                {
                    "sample_id": f"s{index}",
                    "outer_target": "hospital",
                    "policy": "natural",
                    "mask_replicate": 0,
                    "target": target,
                    "observed_fraction": 1.0,
                    "observed_mask_code": 8191,
                    "method": method,
                    "track": "dg_zero_shot",
                    "score": 0.8 if target else 0.2,
                }
            )
    return pd.DataFrame(records)


def test_development_report_requires_exact_cross_method_masks() -> None:
    predictions = _paired_predictions()
    _audit_paired_predictions(predictions)
    predictions.loc[
        predictions["method"].eq("b") & predictions["sample_id"].eq("s0"),
        "observed_mask_code",
    ] = 4095
    with pytest.raises(AssertionError, match="exact masks"):
        _audit_paired_predictions(predictions)


def test_development_report_rejects_incomplete_method_keys() -> None:
    predictions = _paired_predictions().iloc[:-1]
    with pytest.raises(AssertionError, match="complete paired keys"):
        _audit_paired_predictions(predictions)
