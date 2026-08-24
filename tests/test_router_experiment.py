from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heartshift.data.uci import FEATURE_COLUMNS
from heartshift.research.router_experiment import (
    _fixed_blend_selection,
    _simplex_grid,
    assemble_router_table,
)


def test_simplex_grid_and_fixed_blend_selection() -> None:
    grid = _simplex_grid(3, 0.5)
    assert np.allclose(grid.sum(axis=1), 1.0)
    labels = np.tile([0, 1], 20)
    frame = pd.DataFrame(
        {
            "site": ["a"] * 40,
            "policy": ["natural"] * 40,
            "mask_replicate": [0] * 40,
            "target": labels,
        }
    )
    good = np.where(labels == 1, 0.9, 0.1)
    bad = 1.0 - good
    probabilities = np.column_stack([good, bad, np.full(40, 0.5)])
    weights, selected = _fixed_blend_selection(frame, probabilities, step=0.5)
    assert selected >= 0
    assert weights[0] == 1.0


def test_router_alignment_rejects_equal_fraction_but_different_exact_masks(
    tmp_path,
) -> None:
    neural_dir = tmp_path / "neural"
    control_dir = tmp_path / "control"
    neural_dir.mkdir()
    control_dir.mkdir()
    common = {
        "sample_id": "row-1",
        "outer_target": "outer",
        "inner_validation": "inner",
        "policy": "mcar",
        "mask_replicate": 0,
        "site": "inner",
        "target": 1,
        "observed_fraction": 6 / 13,
        "y_score": 0.7,
        "seed": 5062,
        "parameter_id": 0,
        "parameters_json": "{}",
    }
    neural = {**common, "experiment": "candidate", "observed_mask_code": 63}
    anchor = {
        **common,
        "control": "anchor",
        "observed_mask_code": 126,
    }
    for index, feature in enumerate(FEATURE_COLUMNS):
        neural[f"observed__{feature}"] = index < 6
        anchor[f"observed__{feature}"] = 1 <= index < 7
    pd.DataFrame([neural]).to_parquet(neural_dir / "inner_predictions.parquet")
    pd.DataFrame(
        [{"outer_target": "outer", "experiment": "candidate", "parameter_id": 0}]
    ).to_csv(neural_dir / "selected_configurations.csv", index=False)
    pd.DataFrame([anchor]).to_parquet(control_dir / "inner_predictions.parquet")
    pd.DataFrame(
        [
            {
                "outer_target": "outer",
                "selected_anchor": "anchor",
                "parameter_id": 0,
            }
        ]
    ).to_csv(control_dir / "selected_anchors.csv", index=False)

    with pytest.raises(AssertionError, match="exact mask"):
        assemble_router_table(
            neural_dir,
            control_dir,
            neural_experiments={"prior": "candidate"},
            prediction_seeds=(5062,),
        )
