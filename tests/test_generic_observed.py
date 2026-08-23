from __future__ import annotations

import numpy as np
import pandas as pd

from heartshift.models.generic_observed import (
    FeatureMaskPolicy,
    apply_feature_mask_policy,
    fit_generic_observed_fixed_epochs,
    fit_generic_observed_model,
    predict_generic_policy_bank,
)


def _generic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(25)
    rows = []
    for environment in ("a", "b"):
        for label in (0, 1):
            for index in range(30):
                rows.append(
                    {
                        "sample_id": f"{environment}-{label}-{index}",
                        "target": label,
                        "split": "train",
                        "environment": environment,
                        "numeric": rng.normal(label, 1.0),
                        "category": "positive" if label else "negative",
                        "sometimes_missing": np.nan if index % 4 == 0 else rng.normal(),
                    }
                )
    return pd.DataFrame(rows)


def test_generic_mask_never_reveals_natural_missingness() -> None:
    natural = np.array([[True, False, True], [True, True, False]])
    masked = apply_feature_mask_policy(
        natural,
        ("a", "b", "c"),
        FeatureMaskPolicy("mcar", "mcar", rate=0.5),
        base_seed=4,
        replicate=0,
    )
    assert not np.any(masked & ~natural)
    assert masked.any(axis=1).all()


def test_generic_model_handles_string_categories() -> None:
    data = _generic_frame()
    training = data.groupby(["environment", "target"], group_keys=False).head(20)
    validation = data.groupby(["environment", "target"], group_keys=False).tail(10).copy()
    validation["split"] = "validation"
    policies = (
        FeatureMaskPolicy("natural", "natural"),
        FeatureMaskPolicy("mcar_30", "mcar", rate=0.3),
    )
    result = fit_generic_observed_model(
        training,
        validation,
        feature_columns=("numeric", "category", "sometimes_missing"),
        continuous_columns=("numeric", "sometimes_missing"),
        categorical_columns=("category",),
        evaluation_policies=policies,
        variant="prior_separated",
        parameters={
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "dropout": 0.0,
            "batch_size": 16,
            "max_epochs": 2,
            "evaluation_interval": 1,
            "patience_evaluations": 2,
            "validation_mask_replicates": 1,
            "minimum_environment_size": 1,
        },
        seed=8,
        device="cpu",
    )
    predictions = predict_generic_policy_bank(
        result,
        validation,
        policies,
        device="cpu",
        base_seed=9,
        replicates=1,
        batch_size=64,
    )
    assert result.best_epoch >= 1
    assert np.isfinite(predictions["y_score"]).all()
    assert set(predictions["policy"]) == {"natural", "mcar_30"}


def test_generic_fixed_epoch_refit_does_not_require_validation() -> None:
    training = _generic_frame()
    policies = (FeatureMaskPolicy("natural", "natural"),)
    result = fit_generic_observed_fixed_epochs(
        training,
        feature_columns=("numeric", "category", "sometimes_missing"),
        continuous_columns=("numeric", "sometimes_missing"),
        categorical_columns=("category",),
        evaluation_policies=policies,
        variant="prior_separated",
        parameters={
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "dropout": 0.0,
            "batch_size": 32,
            "evaluation_interval": 1,
            "minimum_environment_size": 1,
        },
        seed=12,
        device="cpu",
        epochs=1,
    )
    assert result.best_epoch == 1
    assert np.isnan(result.validation_score)
    assert result.history[-1]["epoch"] == 1.0
