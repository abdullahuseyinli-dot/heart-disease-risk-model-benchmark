from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heartshift.data.uci import FEATURE_COLUMNS
from heartshift.research.router_outer_experiment import (
    OUTER_KEY_COLUMNS,
    _align_expert_frames,
    _router_extra_features,
    _router_gate,
    select_neural_experiments,
)


def test_source_selection_chooses_score_then_smaller_backbone() -> None:
    selections = pd.DataFrame(
        [
            {
                "outer_target": "cleveland",
                "experiment": "attention",
                "variant": "v2",
                "parameter_id": 0,
                "selection_score": 0.4,
                "mean_parameter_count": 4000,
            },
            {
                "outer_target": "cleveland",
                "experiment": "deepsets",
                "variant": "v2",
                "parameter_id": 0,
                "selection_score": 0.4,
                "mean_parameter_count": 3000,
            },
        ]
    )
    selected = select_neural_experiments(selections, {"prior": ("attention", "deepsets")})
    assert selected.loc[0, "selected_experiment"] == "deepsets"
    assert selected.loc[0, "expert_name"] == "prior"


def test_router_context_modes_are_finite_and_schema_stable() -> None:
    probabilities = np.asarray([[0.1, 0.6, 0.8], [0.9, 0.4, 0.2]])
    assert _router_extra_features("support_only", probabilities) is None
    context = _router_extra_features("expert_logit_context", probabilities)
    relative = _router_extra_features("anchor_relative_context", probabilities)
    assert context is not None and context.shape == probabilities.shape
    assert relative is not None and relative.shape == probabilities.shape
    assert np.isfinite(context).all() and np.isfinite(relative).all()
    with pytest.raises(KeyError, match="Unknown router feature mode"):
        _router_extra_features("invalid", probabilities)


def test_outer_alignment_rejects_equal_fraction_but_different_exact_masks() -> None:
    base = {
        "sample_id": "sample-1",
        "outer_target": "cleveland",
        "policy": "mcar_50",
        "mask_replicate": 0,
        "site": "cleveland",
        "record_sha256": "abc",
        "observed_fraction": 6 / 13,
        "y_score": 0.7,
    }
    first = {**base, "observed_mask_code": 63}
    second = {**base, "observed_mask_code": 126}
    for index, feature in enumerate(FEATURE_COLUMNS):
        first[f"observed__{feature}"] = index < 6
        second[f"observed__{feature}"] = 1 <= index < 7
    with pytest.raises(AssertionError, match="observed_mask_code"):
        _align_expert_frames(
            {"first": pd.DataFrame([first]), "second": pd.DataFrame([second])},
            key_columns=OUTER_KEY_COLUMNS,
            endpoint_required=False,
        )


def test_router_gate_uses_strongest_fixed_comparator() -> None:
    records = []
    labels = np.tile([0, 1], 10)
    for method, confidence in (
        ("support_aware_router", 0.8),
        ("fixed_weak", 0.65),
        ("fixed_strong", 0.9),
    ):
        for policy in ("natural", "mcar_10", "mcar_30", "mcar_50"):
            for index, target in enumerate(labels):
                records.append(
                    {
                        "sample_id": f"s{index}",
                        "outer_target": "site-a",
                        "target": target,
                        "policy": policy,
                        "mask_replicate": 0,
                        "observed_fraction": 1.0,
                        "observed_mask_code": 8191,
                        "method": method,
                        "track": "dg_zero_shot",
                        "score": confidence if target else 1 - confidence,
                    }
                )
    weights = pd.DataFrame({"weight__a": [0.6, 0.4], "weight__b": [0.4, 0.6]})
    gate = _router_gate(pd.DataFrame(records), weights, natural_auroc_margin=0.01)
    robust = gate["checks"]["descriptive_robust_improvement_over_strongest_fixed"]
    assert robust["strongest_method"] == "fixed_strong"
    assert not robust["passed"]
