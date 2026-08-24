from __future__ import annotations

import numpy as np

from heartshift.metrics import (
    area_under_risk_coverage,
    balanced_brier_score,
    balanced_log_loss,
    binary_metrics,
)


def test_balanced_scores_are_prevalence_invariant_for_class_constant_predictions() -> None:
    y_a = np.array([0, 1])
    p_a = np.array([0.2, 0.8])
    y_b = np.array([0, 0, 0, 1])
    p_b = np.array([0.2, 0.2, 0.2, 0.8])
    assert np.isclose(balanced_log_loss(y_a, p_a), balanced_log_loss(y_b, p_b))
    assert np.isclose(balanced_brier_score(y_a, p_a), balanced_brier_score(y_b, p_b))


def test_perfect_predictions_have_strong_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    metrics = binary_metrics(y, p)
    assert metrics["roc_auc"] == 1.0
    assert metrics["balanced_log_loss"] < 0.03
    assert area_under_risk_coverage(y, p) == 0.0
