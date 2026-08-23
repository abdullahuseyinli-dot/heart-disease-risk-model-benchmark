"""Assumption-aware prevalence adaptation for prior-separated evidence scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import energy_distance

EPSILON = 1e-6


@dataclass(frozen=True)
class EvidenceCalibrator:
    intercept: float
    slope: float

    def transform(self, evidence_logit: Any) -> np.ndarray:
        score = np.asarray(evidence_logit, dtype=np.float64)
        return self.intercept + self.slope * score


@dataclass(frozen=True)
class MixtureDiagnostic:
    accepted_priors: tuple[float, ...]
    best_prior: float
    best_statistic: float
    table: pd.DataFrame

    @property
    def accepted_interval(self) -> tuple[float, float] | None:
        if not self.accepted_priors:
            return None
        return min(self.accepted_priors), max(self.accepted_priors)


def _site_class_balanced_weights(labels: np.ndarray, sites: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=np.float64)
    unique_sites = np.unique(sites)
    for site in unique_sites:
        for label in (0, 1):
            selected = (sites == site) & (labels == label)
            if selected.any():
                weights[selected] = 1.0 / (2.0 * len(unique_sites) * selected.sum())
    if not np.isclose(weights.sum(), 1.0):
        weights /= weights.sum()
    return weights


def fit_evidence_calibrator(
    evidence_logit: Any,
    labels: Any,
    sites: Any,
) -> EvidenceCalibrator:
    """Fit monotone balanced Platt scaling to source-only out-of-fold scores."""
    score = np.asarray(evidence_logit, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(sites)
    if not (len(score) == len(y) == len(groups)):
        raise ValueError("Calibration arrays must align")
    weights = _site_class_balanced_weights(y, groups)

    def objective(parameters: np.ndarray) -> float:
        intercept = parameters[0]
        slope = np.exp(parameters[1])
        probability = np.clip(expit(intercept + slope * score), EPSILON, 1 - EPSILON)
        point_loss = -(y * np.log(probability) + (1 - y) * np.log(1 - probability))
        return float(np.sum(weights * point_loss))

    result = minimize(objective, np.array([0.0, 0.0]), method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Evidence calibration failed: {result.message}")
    return EvidenceCalibrator(float(result.x[0]), float(np.exp(result.x[1])))


def posterior_from_evidence(evidence_logit: Any, prevalence: float) -> np.ndarray:
    if not 0.0 < prevalence < 1.0:
        raise ValueError("Prevalence must lie strictly between zero and one")
    return np.asarray(
        expit(np.asarray(evidence_logit, dtype=np.float64) + logit(prevalence)),
        dtype=np.float64,
    )


def estimate_target_prevalence_mlls(
    evidence_logit: Any,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> float:
    """Maximum-likelihood target prevalence for calibrated log-likelihood ratios."""
    score = np.asarray(evidence_logit, dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError("Evidence scores must be finite")

    def negative_log_likelihood(prevalence: float) -> float:
        log_terms = np.logaddexp(np.log1p(-prevalence), np.log(prevalence) + score)
        return float(-np.sum(log_terms))

    result = minimize_scalar(
        negative_log_likelihood,
        method="bounded",
        bounds=(lower, upper),
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError("Target-prevalence optimization failed")
    return float(result.x)


def estimate_target_prevalence_soft_bbse(
    source_probability: Any,
    source_labels: Any,
    target_probability: Any,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> float:
    """Estimate binary target prevalence using soft black-box shift estimation."""
    source_score = np.asarray(source_probability, dtype=np.float64)
    labels = np.asarray(source_labels, dtype=np.int64)
    target_score = np.asarray(target_probability, dtype=np.float64)
    if len(source_score) != len(labels):
        raise ValueError("Source scores and labels must align")
    if np.unique(labels).size != 2:
        raise ValueError("Soft BBSE requires both source classes")
    if not (
        np.isfinite(source_score).all()
        and np.isfinite(target_score).all()
        and ((source_score >= 0.0) & (source_score <= 1.0)).all()
        and ((target_score >= 0.0) & (target_score <= 1.0)).all()
    ):
        raise ValueError("Soft BBSE probabilities must be finite and lie in [0, 1]")
    positive_mean_given_negative = float(source_score[labels == 0].mean())
    positive_mean_given_positive = float(source_score[labels == 1].mean())
    separation = positive_mean_given_positive - positive_mean_given_negative
    if abs(separation) < 1e-6:
        raise ValueError("Soft confusion matrix is singular")
    prevalence = (float(target_score.mean()) - positive_mean_given_negative) / separation
    return float(np.clip(prevalence, lower, upper))


def _draw_mixture(
    source_negative: np.ndarray,
    source_positive: np.ndarray,
    prevalence: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = rng.random(size) < prevalence
    sample = np.empty(size, dtype=np.float64)
    n_positive = int(labels.sum())
    sample[labels] = rng.choice(source_positive, size=n_positive, replace=True)
    sample[~labels] = rng.choice(source_negative, size=size - n_positive, replace=True)
    return sample


def mixture_fit_diagnostic(
    source_negative: Any,
    source_positive: Any,
    target_unlabelled: Any,
    *,
    prior_grid: Any = None,
    bootstrap_repetitions: int = 199,
    mixture_draws: int = 3,
    alpha: float = 0.05,
    seed: int = 5062,
) -> MixtureDiagnostic:
    """Test whether target scores resemble any source class-conditional mixture."""
    negative = np.asarray(source_negative, dtype=np.float64)
    positive = np.asarray(source_positive, dtype=np.float64)
    target = np.asarray(target_unlabelled, dtype=np.float64)
    if min(len(negative), len(positive), len(target)) < 5:
        raise ValueError("Mixture diagnostics require at least five observations per distribution")
    if prior_grid is None:
        prior_grid = np.linspace(0.05, 0.95, 19)
    priors = np.asarray(prior_grid, dtype=np.float64)
    rng = np.random.default_rng(seed)
    records = []
    for prevalence in priors:
        target_distances = [
            energy_distance(
                target,
                _draw_mixture(negative, positive, float(prevalence), len(target), rng),
            )
            for _ in range(mixture_draws)
        ]
        statistic = float(np.mean(target_distances))
        null_statistics = []
        for _ in range(bootstrap_repetitions):
            null_statistics.append(
                float(
                    np.mean(
                        [
                            energy_distance(
                                _draw_mixture(
                                    negative,
                                    positive,
                                    float(prevalence),
                                    len(target),
                                    rng,
                                ),
                                _draw_mixture(
                                    negative,
                                    positive,
                                    float(prevalence),
                                    len(target),
                                    rng,
                                ),
                            )
                            for _ in range(mixture_draws)
                        ]
                    )
                )
            )
        null_array = np.asarray(null_statistics)
        p_value = float((1 + np.sum(null_array >= statistic)) / (bootstrap_repetitions + 1))
        records.append(
            {
                "prevalence": float(prevalence),
                "energy_statistic": statistic,
                "p_value": p_value,
                "accepted": bool(p_value >= alpha),
            }
        )
    table = pd.DataFrame(records)
    accepted = tuple(float(value) for value in table.loc[table["accepted"], "prevalence"])
    best = table.sort_values(["energy_statistic", "prevalence"]).iloc[0]
    return MixtureDiagnostic(
        accepted_priors=accepted,
        best_prior=float(best["prevalence"]),
        best_statistic=float(best["energy_statistic"]),
        table=table,
    )
