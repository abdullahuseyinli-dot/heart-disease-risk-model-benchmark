from __future__ import annotations

from pathlib import Path

import pandas as pd

from heartshift.data.uci import FEATURE_COLUMNS
from heartshift.reporting.seed_extension import exact_seed_reproduction, multiplicity_sensitivity


def _seed_frame() -> pd.DataFrame:
    rows = []
    for seed in (1, 2):
        row = {
            "sample_id": "sample-1",
            "outer_target": "site-a",
            "experiment": "candidate",
            "policy": "natural",
            "mask_replicate": 0,
            "training_seed": seed,
            "record_sha256": "abc",
            "observed_fraction": 1.0,
            "observed_mask_code": (1 << len(FEATURE_COLUMNS)) - 1,
            "target": 1,
            "fit_seed": 100 + seed,
            "epochs": 5,
            "y_score": 0.75 + seed / 100,
        }
        row.update({f"observed__{feature}": True for feature in FEATURE_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_exact_seed_reproduction_accepts_exact_extension(tmp_path: Path) -> None:
    historical = _seed_frame().iloc[[0]].copy()
    extension = _seed_frame()
    historical_path = tmp_path / "historical.parquet"
    extension_path = tmp_path / "extension.parquet"
    historical.to_parquet(historical_path, index=False)
    extension.to_parquet(extension_path, index=False)

    result = exact_seed_reproduction(historical_path, extension_path, (1,))

    assert result["status"] == "exact_value_reproduction"
    assert result["rows"] == 1
    assert result["maximum_probability_difference"] == 0.0


def test_exact_seed_reproduction_rejects_changed_probability(tmp_path: Path) -> None:
    historical = _seed_frame().iloc[[0]].copy()
    extension = _seed_frame()
    extension.loc[extension["training_seed"].eq(1), "y_score"] = 0.1
    historical_path = tmp_path / "historical.parquet"
    extension_path = tmp_path / "extension.parquet"
    historical.to_parquet(historical_path, index=False)
    extension.to_parquet(extension_path, index=False)

    try:
        exact_seed_reproduction(historical_path, extension_path, (1,))
    except AssertionError as error:
        assert "probabilities" in str(error)
    else:
        raise AssertionError("Changed historical probability was accepted")


def test_multiplicity_sensitivity_returns_familywise_intervals() -> None:
    replicates = pd.DataFrame(
        {
            "bootstrap_replicate": [0, 0, 1, 1, 2, 2],
            "method": ["a", "b", "a", "b", "a", "b"],
            "difference_candidate_minus_reference": [-0.2, 0.1, -0.1, 0.2, 0.0, 0.3],
            "contrast_id": ["family"] * 6,
        }
    )
    intervals = pd.DataFrame(
        {
            "method": ["a", "b"],
            "observed_difference": [-0.1, 0.2],
            "contrast_id": ["family", "family"],
        }
    )

    result = multiplicity_sensitivity(replicates, intervals, contrast_id="family", alpha=0.05)

    assert set(result["method"]) == {"a", "b"}
    assert result["family_size"].eq(2).all()
    assert result["simultaneous_max_error_critical_value"].gt(0).all()
