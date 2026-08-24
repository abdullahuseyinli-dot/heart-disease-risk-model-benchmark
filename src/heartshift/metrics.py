"""Metrics that keep discrimination, calibration, and selective risk separate."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

EPSILON = 1e-7


def _arrays(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(y_score, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true and y_score must be aligned one-dimensional arrays")
    if not np.isfinite(p).all():
        raise ValueError("Non-finite prediction encountered")
    return y, p


def class_balanced_weights(y_true: Any) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.int64)
    weights = np.zeros(len(y), dtype=np.float64)
    for label in (0, 1):
        mask = y == label
        if not mask.any():
            raise ValueError("Balanced metrics require both outcome classes")
        weights[mask] = 0.5 / mask.sum()
    return weights


def balanced_log_loss(y_true: Any, y_score: Any) -> float:
    y, p = _arrays(y_true, y_score)
    point_loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.sum(class_balanced_weights(y) * point_loss))


def balanced_brier_score(y_true: Any, y_score: Any) -> float:
    y, p = _arrays(y_true, y_score)
    return float(np.sum(class_balanced_weights(y) * np.square(p - y)))


def calibration_intercept_slope(y_true: Any, y_score: Any) -> tuple[float, float]:
    y, p = _arrays(y_true, y_score)
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    logits = logit(p)

    def objective(parameters: np.ndarray) -> float:
        predicted = expit(parameters[0] + parameters[1] * logits)
        return float(log_loss(y, predicted, labels=[0, 1]))

    result = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    if not result.success:
        return float("nan"), float("nan")
    return float(result.x[0]), float(result.x[1])


def sensitivity_at_specificity(y_true: Any, y_score: Any, specificity: float = 0.80) -> float:
    y, p = _arrays(y_true, y_score)
    if np.unique(y).size < 2:
        return float("nan")
    false_positive_rate, true_positive_rate, _ = roc_curve(y, p)
    eligible = true_positive_rate[false_positive_rate <= 1.0 - specificity + 1e-12]
    return float(eligible.max()) if len(eligible) else 0.0


def area_under_risk_coverage(y_true: Any, y_score: Any) -> float:
    y, p = _arrays(y_true, y_score)
    predicted = p >= 0.5
    confidence = np.abs(p - 0.5)
    order = np.argsort(-confidence, kind="stable")
    errors = predicted[order] != y[order]
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    return float(np.trapezoid(cumulative_risk, coverage))


def binary_metrics(y_true: Any, y_score: Any, threshold: float = 0.5) -> dict[str, float]:
    y, p = _arrays(y_true, y_score)
    labels = (p >= threshold).astype(np.int64)
    intercept, slope = calibration_intercept_slope(y, p)
    has_both = np.unique(y).size == 2
    return {
        "n": float(len(y)),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if has_both else float("nan"),
        "average_precision": float(average_precision_score(y, p)) if has_both else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "balanced_log_loss": balanced_log_loss(y, p),
        "brier": float(brier_score_loss(y, p)),
        "balanced_brier": balanced_brier_score(y, p),
        "accuracy": float(accuracy_score(y, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y, labels)),
        "f1": float(f1_score(y, labels, zero_division=0)),
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "sensitivity_at_80_specificity": sensitivity_at_specificity(y, p, 0.80),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "aurc": area_under_risk_coverage(y, p),
    }
