from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heartshift.adaptation import CompositeMixtureDiagnostic, MaskSupportAudit
from heartshift.data.uci import FEATURE_COLUMNS
from heartshift.evaluation.neural_benchmark import _parameter_grid
from heartshift.evaluation.neural_outer import (
    _adapt_predictions,
    _exact_shard_files,
    _load_heart_outer_frames,
    evaluation_units,
)
from heartshift.models.ps_maskdro import (
    CORE_PREDICTION_COLUMNS,
    MASK_PREDICTION_COLUMNS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parameter_grid_is_deterministic_and_preserves_base() -> None:
    combinations = _parameter_grid(
        {"fixed": 7, "learning_rate": 0.1},
        {"learning_rate": [0.01, 0.02], "dro_lambda": [0.25, 0.5]},
    )
    assert combinations == [
        {"fixed": 7, "learning_rate": 0.01, "dro_lambda": 0.25},
        {"fixed": 7, "learning_rate": 0.02, "dro_lambda": 0.25},
        {"fixed": 7, "learning_rate": 0.01, "dro_lambda": 0.5},
        {"fixed": 7, "learning_rate": 0.02, "dro_lambda": 0.5},
    ]


def test_outer_evaluation_units_match_policy_replication_rules() -> None:
    units = evaluation_units(3)
    assert len(units) == 19
    assert sum(policy.name == "natural" for policy, _ in units) == 1
    assert sum(policy.name == "mcar_30" for policy, _ in units) == 3
    assert sum(policy.name == "drop_advanced" for policy, _ in units) == 1


def test_exact_shard_file_selection_excludes_endpoint_free_prefix(tmp_path: Path) -> None:
    shard = tmp_path / "site__method"
    shard.mkdir()
    labelled = shard / "outer_predictions.parquet"
    unlabelled = shard / "unlabelled_outer_predictions.parquet"
    labelled.write_bytes(b"labelled")
    unlabelled.write_bytes(b"endpoint-free")

    assert _exact_shard_files(tmp_path, "outer_predictions.parquet") == [labelled]


def test_neural_outer_loader_excludes_locked_endpoint_until_scoring() -> None:
    source, target_unlabelled = _load_heart_outer_frames(
        REPO_ROOT / "data/processed/uci_heart_canonical_v1.parquet",
        outer_target="cleveland",
        feature_columns=FEATURE_COLUMNS,
    )
    assert len(source) == 920 - 303
    assert len(target_unlabelled) == 303
    assert "target" in source
    assert "target" not in target_unlabelled
    assert source["site"].ne("cleveland").all()
    assert target_unlabelled["site"].eq("cleveland").all()


def test_non_adaptable_outer_does_not_require_source_predictions() -> None:
    target = pd.DataFrame(
        {
            "sample_id": ["target-0", "target-1"],
            "site": ["held-out", "held-out"],
            "record_sha256": ["sha-0", "sha-1"],
            "policy": ["natural", "natural"],
            "mask_replicate": [0, 0],
            "evidence_logit": [-0.5, 0.5],
            "y_score_zero_shot": [0.25, 0.75],
            "observed_fraction": [1.0, 1.0],
        }
    )

    adapted, diagnostics, summary = _adapt_predictions(
        target,
        pd.DataFrame(),
        adaptable=False,
        diagnostic_config={"mode": "composite_multiview_v3"},
        base_seed=5062,
    )

    assert adapted["adaptation_status"].eq("not_applicable").all()
    assert not adapted["adaptation_allowed"].any()
    assert adapted[
        [
            "calibrated_evidence_logit",
            "y_score_calibrated_equal_prior",
            "estimated_target_prevalence_mlls",
            "estimated_target_prevalence_soft_bbse",
            "y_score_uda_mlls_research",
            "y_score_uda_soft_bbse_research",
            "y_score_uda_mlls",
            "y_score_uda_soft_bbse",
        ]
    ].isna().all().all()
    assert diagnostics.empty
    assert summary.empty


@pytest.mark.parametrize("accepted", [False, True])
def test_v3_outer_adaptation_separates_research_from_gated_scores(
    monkeypatch: pytest.MonkeyPatch,
    accepted: bool,
) -> None:
    source_size = 60
    target_size = 30
    source_labels = np.tile([0, 1], source_size // 2)
    source = pd.DataFrame(
        {
            "sample_id": [f"source-{index}" for index in range(source_size)],
            "site": np.tile(["a", "b", "c"], source_size // 3),
            "target": source_labels,
            "evaluation_policy": "natural",
            "policy_replicate": 0,
            "evidence_logit": np.where(source_labels == 1, 1.2, -1.2)
            + np.linspace(-0.2, 0.2, source_size),
        }
    )
    target = pd.DataFrame(
        {
            "sample_id": [f"target-{index}" for index in range(target_size)],
            "site": "held-out",
            "policy": "natural",
            "mask_replicate": 0,
            "evidence_logit": np.linspace(-1.5, 1.5, target_size),
            "y_score_zero_shot": np.linspace(0.1, 0.9, target_size),
        }
    )
    for index, column in enumerate(CORE_PREDICTION_COLUMNS):
        source[column] = np.linspace(index, index + 1.0, source_size)
        target[column] = np.linspace(index, index + 1.0, target_size)
    for index, column in enumerate(MASK_PREDICTION_COLUMNS):
        source[column] = (np.arange(source_size) + index) % 3 != 0
        target[column] = (np.arange(target_size) + index) % 3 != 0

    observed_calls: list[tuple[list[str], list[str]]] = []

    def fake_diagnostic(*args: object, **kwargs: object) -> CompositeMixtureDiagnostic:
        observed_calls.append(
            (
                list(kwargs["source_sample_ids"]),  # type: ignore[arg-type]
                list(kwargs["target_sample_ids"]),  # type: ignore[arg-type]
            )
        )
        support = MaskSupportAudit(
            passed=True,
            all_feature_states_supported=True,
            exact_pattern_support_rate=1.0,
            nearest_hamming_fraction_q95=0.0,
            nearest_hamming_fraction_max=0.0,
            maximum_allowed_nearest_hamming_fraction=0.25,
        )
        table = pd.DataFrame(
            {
                "view": ["evidence", "core", "mask", "composite"],
                "best_prior": [0.5] * 4,
                "mmd2_statistic": [0.01] * 4,
                "p_value": [0.5 if accepted else 0.01] * 4,
                "accepted": [accepted] * 4,
            }
        )
        return CompositeMixtureDiagnostic(
            accepted=accepted,
            best_prior=0.5,
            best_statistic=0.01,
            p_value=0.5 if accepted else 0.01,
            table=table,
            mask_support=support,
        )

    monkeypatch.setattr(
        "heartshift.evaluation.neural_outer.acquisition_aware_label_shift_diagnostic",
        fake_diagnostic,
    )
    adapted, diagnostics, summary = _adapt_predictions(
        target,
        source,
        adaptable=True,
        diagnostic_config={
            "mode": "composite_multiview_v3",
            "prior_grid": [0.25, 0.5, 0.75],
            "bootstrap_repetitions": 19,
            "rff_features_per_view": 16,
            "alpha": 0.05,
            "maximum_nearest_hamming_fraction": 0.25,
        },
        base_seed=5062,
    )

    assert observed_calls == [(source["sample_id"].tolist(), target["sample_id"].tolist())]
    assert np.isfinite(adapted["y_score_calibrated_equal_prior"]).all()
    assert np.isfinite(adapted["y_score_uda_mlls_research"]).all()
    assert np.isfinite(adapted["y_score_uda_soft_bbse_research"]).all()
    assert bool(adapted["adaptation_allowed"].iloc[0]) is accepted
    assert summary.loc[0, "diagnostic_mode"] == "composite_multiview_v3"
    assert bool(summary.loc[0, "adaptation_allowed"]) is accepted
    assert diagnostics["automatic_adaptation_allowed"].eq(accepted).all()
    if accepted:
        pd.testing.assert_series_equal(
            adapted["y_score_uda_mlls"],
            adapted["y_score_uda_mlls_research"],
            check_names=False,
        )
    else:
        assert adapted["y_score_uda_mlls"].isna().all()
        assert adapted["y_score_uda_soft_bbse"].isna().all()
        assert adapted["adaptation_status"].eq("diagnostic_rejected").all()
