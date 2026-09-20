from __future__ import annotations

import numpy as np
import torch

from heartshift.models.support_shrinkage import (
    SupportAwareRouter,
    fit_support_aware_router,
    fit_support_reference,
    predict_support_aware_router,
    support_router_objective,
    transform_support_features,
)


def test_support_features_use_frozen_source_masks() -> None:
    probabilities = np.asarray([[0.2, 0.3], [0.8, 0.7], [0.4, 0.6]])
    masks = np.asarray([[1, 1], [1, 0], [1, 1]], dtype=bool)
    reference, training = fit_support_reference(probabilities, masks)
    shifted = transform_support_features(
        reference,
        np.asarray([[0.5, 0.5]]),
        np.asarray([[0, 1]], dtype=bool),
    )
    assert training.shape[1] == shifted.shape[1]
    assert np.isfinite(training).all()
    assert np.isfinite(shifted).all()
    # The unseen mask has positive nearest-Hamming distance after standardization.
    assert not np.allclose(shifted, 0.0)


def test_support_router_objective_is_finite_and_differentiable() -> None:
    router = SupportAwareRouter(3, 2, hidden_dim=8, dropout=0.0)
    support = torch.randn(12, 3)
    expert_logits = torch.randn(12, 2)
    mixed, weights = router(support, expert_logits)
    objective, components = support_router_objective(
        mixed,
        expert_logits,
        torch.tensor([0, 1] * 6),
        torch.tensor([0, 1, 2, 3] * 3),
        torch.tensor([True, False] * 6),
        weights,
        dro_lambda=0.5,
        dro_tau=0.2,
        regret_beta=0.5,
        natural_margin=0.01,
        entropy_bonus=0.001,
    )
    objective.backward()  # type: ignore[no-untyped-call]
    assert torch.isfinite(objective)
    assert all(np.isfinite(value) for value in components.values())
    assert all(parameter.grad is not None for parameter in router.parameters())


def test_router_learns_mask_specific_shrinkage_from_source_only_data() -> None:
    rng = np.random.default_rng(9)
    n_rows = 240
    labels = np.tile([0, 1], n_rows // 2)
    masks = np.zeros((n_rows, 2), dtype=bool)
    masks[: n_rows // 2, 0] = True
    masks[n_rows // 2 :, 1] = True
    good = np.where(labels == 1, 0.9, 0.1)
    bad = np.where(labels == 1, 0.35, 0.65)
    probabilities = np.column_stack(
        [
            np.where(masks[:, 0], good, bad),
            np.where(masks[:, 1], good, bad),
        ]
    )
    probabilities = np.clip(probabilities + rng.normal(0, 0.01, probabilities.shape), 0.01, 0.99)
    training = np.arange(n_rows) % 3 != 0
    validation = ~training
    groups = labels + 2 * masks[:, 1].astype(int)
    result = fit_support_aware_router(
        probabilities[training],
        masks[training],
        labels[training],
        groups[training],
        np.ones(training.sum(), dtype=bool),
        probabilities[validation],
        masks[validation],
        labels[validation],
        groups[validation],
        expert_names=("left", "right"),
        parameters={
            "hidden_dim": 16,
            "dropout": 0.0,
            "learning_rate": 0.01,
            "max_epochs": 300,
            "patience": 50,
            "dro_lambda": 0.5,
            "regret_beta": 0.2,
            "entropy_bonus": 0.0,
        },
        seed=4,
        device="cpu",
    )
    predicted, weights = predict_support_aware_router(
        result, probabilities[validation], masks[validation]
    )
    assert np.mean(weights[masks[validation, 0], 0]) > 0.8
    assert np.mean(weights[masks[validation, 1], 1]) > 0.8
    assert np.mean((predicted >= 0.5) == labels[validation]) > 0.95
