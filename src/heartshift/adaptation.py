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


@dataclass(frozen=True)
class MaskSupportAudit:
    """Deterministic support check kept separate from shift hypothesis testing."""

    passed: bool
    all_feature_states_supported: bool
    exact_pattern_support_rate: float
    nearest_hamming_fraction_q95: float
    nearest_hamming_fraction_max: float
    maximum_allowed_nearest_hamming_fraction: float


@dataclass(frozen=True)
class CompositeMixtureDiagnostic:
    """Acquisition-aware multiview test of the observed-data label-shift null."""

    accepted: bool
    best_prior: float
    best_statistic: float
    p_value: float
    table: pd.DataFrame
    mask_support: MaskSupportAudit


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


def _finite_matrix(values: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty one- or two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _binary_mask_matrix(values: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isin(matrix, (0, 1, False, True)).all():
        raise ValueError(f"{name} must contain only binary observed-mask indicators")
    return matrix.astype(bool, copy=False)


def audit_mask_support(
    source_mask: Any,
    target_mask: Any,
    *,
    maximum_nearest_hamming_fraction: float = 0.25,
) -> MaskSupportAudit:
    """Check target masks have local support without treating support as a shift test."""
    source = _binary_mask_matrix(source_mask, name="source_mask")
    target = _binary_mask_matrix(target_mask, name="target_mask")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target masks must have the same feature count")
    if not 0.0 <= maximum_nearest_hamming_fraction <= 1.0:
        raise ValueError("The maximum nearest-Hamming fraction must lie in [0, 1]")

    all_states_supported = all(
        set(np.unique(target[:, column])).issubset(set(np.unique(source[:, column])))
        for column in range(source.shape[1])
    )
    source_patterns = {np.packbits(row).tobytes() for row in source}
    exact_support = float(
        np.mean([np.packbits(row).tobytes() in source_patterns for row in target])
    )

    # Chunking bounds memory for future cohorts while retaining an exact nearest
    # Hamming calculation for the small UCI hospitals used in this benchmark.
    nearest_parts: list[np.ndarray] = []
    for start in range(0, len(target), 256):
        chunk = target[start : start + 256]
        distances = np.count_nonzero(chunk[:, None, :] != source[None, :, :], axis=2)
        nearest_parts.append(distances.min(axis=1) / source.shape[1])
    nearest = np.concatenate(nearest_parts)
    q95 = float(np.quantile(nearest, 0.95, method="higher"))
    maximum = float(nearest.max())
    passed = bool(all_states_supported and q95 <= maximum_nearest_hamming_fraction)
    return MaskSupportAudit(
        passed=passed,
        all_feature_states_supported=bool(all_states_supported),
        exact_pattern_support_rate=exact_support,
        nearest_hamming_fraction_q95=q95,
        nearest_hamming_fraction_max=maximum,
        maximum_allowed_nearest_hamming_fraction=float(maximum_nearest_hamming_fraction),
    )


def _source_standardize(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    location = source.mean(axis=0)
    scale = source.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (source - location) / scale, (target - location) / scale


def _median_bandwidth(source: np.ndarray) -> float:
    if len(source) > 512:
        indices = np.linspace(0, len(source) - 1, 512, dtype=np.int64)
        reference = source[indices]
    else:
        reference = source
    differences = reference[:, None, :] - reference[None, :, :]
    squared = np.sum(differences * differences, axis=2)
    distances = np.sqrt(squared[np.triu_indices(len(reference), k=1)])
    positive = distances[distances > 1e-8]
    return float(np.median(positive)) if len(positive) else 1.0


def _rff_view(
    source: np.ndarray,
    target: np.ndarray,
    *,
    dimensions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    source_scaled, target_scaled = _source_standardize(source, target)
    bandwidth = max(_median_bandwidth(source_scaled), 1e-6)
    frequencies = rng.normal(
        loc=0.0,
        scale=1.0 / bandwidth,
        size=(source.shape[1], dimensions),
    )
    phases = rng.uniform(0.0, 2.0 * np.pi, size=dimensions)
    normalization = np.sqrt(2.0 / dimensions)
    source_features = normalization * np.cos(source_scaled @ frequencies + phases)
    target_features = normalization * np.cos(target_scaled @ frequencies + phases)
    return source_features, target_features, bandwidth


def _minimum_embedding_discrepancy(
    negative_mean: np.ndarray,
    positive_mean: np.ndarray,
    target_mean: np.ndarray,
    priors: np.ndarray,
) -> tuple[float, float]:
    mixture_means = (
        negative_mean[None, :] + priors[:, None] * (positive_mean - negative_mean)[None, :]
    )
    squared_distances = np.sum((mixture_means - target_mean[None, :]) ** 2, axis=1)
    best_index = int(np.argmin(squared_distances))
    return float(priors[best_index]), float(squared_distances[best_index])


def _composite_bootstrap_null(
    source_features: np.ndarray,
    labels: np.ndarray,
    *,
    target_size: int,
    fitted_prior: float,
    priors: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    negative = source_features[labels == 0]
    positive = source_features[labels == 1]
    null = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        negative_reference = negative[
            rng.integers(0, len(negative), size=len(negative), endpoint=False)
        ]
        positive_reference = positive[
            rng.integers(0, len(positive), size=len(positive), endpoint=False)
        ]
        pseudo_labels = rng.random(target_size) < fitted_prior
        pseudo_target = np.empty((target_size, source_features.shape[1]), dtype=np.float64)
        n_positive = int(pseudo_labels.sum())
        if n_positive:
            pseudo_target[pseudo_labels] = positive[
                rng.integers(0, len(positive), size=n_positive, endpoint=False)
            ]
        if n_positive < target_size:
            pseudo_target[~pseudo_labels] = negative[
                rng.integers(
                    0,
                    len(negative),
                    size=target_size - n_positive,
                    endpoint=False,
                )
            ]
        _, null[repetition] = _minimum_embedding_discrepancy(
            negative_reference.mean(axis=0),
            positive_reference.mean(axis=0),
            pseudo_target.mean(axis=0),
            priors,
        )
    return null


def acquisition_aware_label_shift_diagnostic(
    source_evidence: Any,
    source_labels: Any,
    source_core: Any,
    source_mask: Any,
    target_evidence: Any,
    target_core: Any,
    target_mask: Any,
    *,
    source_sample_ids: Any | None = None,
    target_sample_ids: Any | None = None,
    prior_grid: Any = None,
    bootstrap_repetitions: int = 199,
    rff_features_per_view: int = 128,
    alpha: float = 0.05,
    maximum_nearest_hamming_fraction: float = 0.25,
    seed: int = 5062,
) -> CompositeMixtureDiagnostic:
    """Test label-shift compatibility across evidence, core, and acquisition views.

    The target labels are neither accepted nor required.  Each source patient must
    occur once: counterfactual mask replicates are deliberately prohibited because
    they are not independent observations.  The composite-null bootstrap samples a
    target mixture at the fitted prior and re-estimates the prior in every replicate.
    """
    evidence_source = _finite_matrix(source_evidence, name="source_evidence")
    evidence_target = _finite_matrix(target_evidence, name="target_evidence")
    core_source = _finite_matrix(source_core, name="source_core")
    core_target = _finite_matrix(target_core, name="target_core")
    mask_source = _binary_mask_matrix(source_mask, name="source_mask")
    mask_target = _binary_mask_matrix(target_mask, name="target_mask")
    labels = np.asarray(source_labels, dtype=np.int64)

    source_sizes = {
        len(evidence_source),
        len(core_source),
        len(mask_source),
        len(labels),
    }
    target_sizes = {len(evidence_target), len(core_target), len(mask_target)}
    if len(source_sizes) != 1 or len(target_sizes) != 1:
        raise ValueError("Every diagnostic view must align within source and target")
    if evidence_source.shape[1] != evidence_target.shape[1]:
        raise ValueError("Source and target evidence dimensions differ")
    if core_source.shape[1] != core_target.shape[1]:
        raise ValueError("Source and target core dimensions differ")
    if mask_source.shape[1] != mask_target.shape[1]:
        raise ValueError("Source and target mask dimensions differ")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("The source diagnostic sample must contain both binary classes")
    if source_sample_ids is not None:
        source_ids = np.asarray(source_sample_ids).astype(str)
        if len(source_ids) != len(labels) or len(np.unique(source_ids)) != len(source_ids):
            raise ValueError("Source diagnostic predictions must contain unique patients")
    if target_sample_ids is not None:
        target_ids = np.asarray(target_sample_ids).astype(str)
        if len(target_ids) != len(evidence_target) or len(np.unique(target_ids)) != len(target_ids):
            raise ValueError("Target diagnostic predictions must contain unique patients")
    if prior_grid is None:
        prior_grid = np.linspace(0.05, 0.95, 19)
    priors = np.asarray(prior_grid, dtype=np.float64)
    if (
        priors.ndim != 1
        or len(priors) < 2
        or not np.isfinite(priors).all()
        or not ((priors > 0.0) & (priors < 1.0)).all()
        or np.any(np.diff(priors) <= 0.0)
    ):
        raise ValueError("The prior grid must be finite, increasing, and inside (0, 1)")
    if bootstrap_repetitions < 19:
        raise ValueError("At least 19 bootstrap repetitions are required")
    if rff_features_per_view < 16:
        raise ValueError("At least 16 random Fourier features per view are required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    support = audit_mask_support(
        mask_source,
        mask_target,
        maximum_nearest_hamming_fraction=maximum_nearest_hamming_fraction,
    )
    rng = np.random.default_rng(seed)
    raw_views = {
        "evidence": (evidence_source, evidence_target),
        "core": (core_source, core_target),
        "mask": (mask_source.astype(np.float64), mask_target.astype(np.float64)),
    }
    source_views: dict[str, np.ndarray] = {}
    target_views: dict[str, np.ndarray] = {}
    bandwidths: dict[str, float] = {}
    for name, (source_view, target_view) in raw_views.items():
        source_features, target_features, bandwidth = _rff_view(
            source_view,
            target_view,
            dimensions=rff_features_per_view,
            rng=rng,
        )
        source_views[name] = source_features
        target_views[name] = target_features
        bandwidths[name] = bandwidth
    view_count = len(source_views)
    source_views["composite"] = np.concatenate(
        [source_views[name] / np.sqrt(view_count) for name in raw_views], axis=1
    )
    target_views["composite"] = np.concatenate(
        [target_views[name] / np.sqrt(view_count) for name in raw_views], axis=1
    )

    records: list[dict[str, Any]] = []
    for view_name in (*raw_views, "composite"):
        source_features = source_views[view_name]
        target_features = target_views[view_name]
        best_prior, statistic = _minimum_embedding_discrepancy(
            source_features[labels == 0].mean(axis=0),
            source_features[labels == 1].mean(axis=0),
            target_features.mean(axis=0),
            priors,
        )
        null = _composite_bootstrap_null(
            source_features,
            labels,
            target_size=len(target_features),
            fitted_prior=best_prior,
            priors=priors,
            repetitions=bootstrap_repetitions,
            rng=rng,
        )
        p_value = float((1 + np.sum(null >= statistic)) / (bootstrap_repetitions + 1))
        records.append(
            {
                "view": view_name,
                "best_prior": best_prior,
                "mmd2_statistic": statistic,
                "bootstrap_critical_value": float(np.quantile(null, 1.0 - alpha, method="higher")),
                "p_value": p_value,
                "accepted": bool(p_value >= alpha),
                "alpha": float(alpha),
                "rff_dimensions": int(source_features.shape[1]),
                "source_bandwidth": bandwidths.get(view_name, np.nan),
            }
        )
    table = pd.DataFrame(records)
    composite = table.loc[table["view"].eq("composite")].iloc[0]
    all_views_accepted = bool(table["accepted"].all())
    return CompositeMixtureDiagnostic(
        accepted=bool(all_views_accepted and support.passed),
        best_prior=float(composite["best_prior"]),
        best_statistic=float(composite["mmd2_statistic"]),
        p_value=float(composite["p_value"]),
        table=table,
        mask_support=support,
    )
