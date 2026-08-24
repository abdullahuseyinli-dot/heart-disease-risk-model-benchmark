from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from heartshift.data.uci import CORE_COLUMNS, FEATURE_COLUMNS, load_uci_heart
from heartshift.masks import MaskPolicy
from heartshift.models.ps_maskdro import (
    CORE_MISSING_SENTINEL,
    CORE_PREDICTION_COLUMNS,
    MASK_PREDICTION_COLUMNS,
    fit_ps_maskdro,
    fit_ps_maskdro_fixed_epochs,
    mirrams_objective,
    predict_policy_bank,
    ps_maskdro_objective,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = REPO_ROOT / "data/raw/uci_heart/doi-10.24432-C52P4X/extracted"


def test_entropic_objective_is_at_least_mean_group_risk() -> None:
    logits = torch.tensor([[0.0, 0.0, -2.0, 2.0], [0.5, -0.5, -1.0, 1.0]])
    labels = torch.tensor([0, 1, 0, 1])
    sites = torch.tensor([0, 0, 1, 1])
    objective, components = ps_maskdro_objective(
        logits,
        labels,
        sites,
        variant="v5",
        dro_lambda=1.0,
        dro_tau=0.2,
        brier_beta=0.0,
    )
    assert float(objective) >= components["mean_group_risk"] - 1e-7


def test_site_mask_and_joint_dro_have_distinct_group_axes() -> None:
    labels = torch.tensor([0, 1, 0, 1])
    sites = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor([[4.0, -4.0, 1.0, -1.0], [2.0, -2.0, 2.0, -2.0]])
    objectives = {}
    for variant in ("v5_site", "v5_mask", "v5"):
        objective, _ = ps_maskdro_objective(
            logits,
            labels,
            sites,
            variant=variant,
            dro_lambda=1.0,
            dro_tau=0.2,
            brier_beta=0.0,
        )
        objectives[variant] = float(objective)
    assert not np.isclose(objectives["v5_site"], objectives["v5_mask"])
    assert objectives["v5"] > max(objectives["v5_site"], objectives["v5_mask"])


def test_mirrams_equation_has_all_three_loss_terms() -> None:
    labels = torch.tensor([0, 1, 0, 1])
    logits = torch.tensor([[-4.0, 4.0, 0.0, 0.0], [-2.0, 2.0, 1.0, -1.0]])
    objective, components = mirrams_objective(
        logits,
        labels,
        lambda_supervised_mask=2.0,
        lambda_consistency=3.0,
        confidence_threshold=0.9,
    )
    expected = (
        components["natural_supervised_risk"]
        + 2.0 * components["masked_supervised_risk"]
        + 3.0 * components["consistency_risk"]
    )
    assert np.isclose(float(objective), expected)
    assert np.isclose(components["confident_fraction"], 0.5)


def test_short_cpu_training_is_finite() -> None:
    data = load_uci_heart(EXTRACTED)
    cleveland = data.loc[data["site"].eq("cleveland")]
    hungary = data.loc[data["site"].eq("hungary")]
    training = cleveland.groupby("target", group_keys=False).head(60).copy()
    validation = hungary.groupby("target", group_keys=False).head(30).copy()
    result = fit_ps_maskdro(
        training,
        validation,
        variant="v2",
        parameters={
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "dropout": 0.0,
            "max_epochs": 2,
            "evaluation_interval": 1,
            "patience_evaluations": 2,
            "validation_mask_replicates": 1,
        },
        seed=3,
        device="cpu",
    )
    assert result.best_epoch >= 1
    assert np.isfinite(result.validation_score)
    assert result.parameter_count > 0


def test_fixed_epoch_refit_does_not_require_validation_data() -> None:
    data = load_uci_heart(EXTRACTED)
    training = (
        data.loc[data["site"].isin(["cleveland", "hungary"])]
        .groupby(["site", "target"], group_keys=False)
        .head(20)
    )
    result = fit_ps_maskdro_fixed_epochs(
        training,
        variant="v2",
        parameters={
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "dropout": 0.0,
            "evaluation_interval": 1,
        },
        seed=7,
        device="cpu",
        epochs=2,
    )
    assert result.best_epoch == 2
    assert np.isnan(result.validation_score)
    assert len(result.history) == 2

    mask_pool = training.loc[:, FEATURE_COLUMNS].notna().to_numpy()
    predictions = predict_policy_bank(
        result.model,
        result.preprocessor,
        training,
        (MaskPolicy("drop_core", "panel", panel="core"),),
        device=torch.device("cpu"),
        base_seed=11,
        replicates=1,
        empirical_mask_pool=mask_pool,
    )
    for feature, core_column in zip(CORE_COLUMNS, CORE_PREDICTION_COLUMNS, strict=True):
        mask_column = MASK_PREDICTION_COLUMNS[FEATURE_COLUMNS.index(feature)]
        assert predictions[mask_column].eq(False).all()
        assert predictions[core_column].eq(CORE_MISSING_SENTINEL).all()
