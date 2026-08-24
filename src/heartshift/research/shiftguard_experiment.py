"""Source-only synthetic success and falsification experiment for ShiftGuard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from heartshift.config import config_hash
from heartshift.data.uci import sha256_file
from heartshift.evaluation.classical_benchmark import write_run_manifest
from heartshift.masks import observed_mask_hashes
from heartshift.models.shiftguard import (
    JointShiftGuardCalibration,
    PolynomialMomentDiagnostic,
    PrevalenceSet,
    QuantileCopulaDiagnostic,
    RandomFourierDiagnostic,
    ShiftGuardCalibration,
    ShiftGuardFitResult,
    compose_diagnostic_features,
    encode_shiftguard,
    fit_mixture_precision_matrices,
    fit_polynomial_moment_diagnostic,
    fit_quantile_copula_diagnostic,
    fit_random_fourier_diagnostic,
    fit_shiftguard,
    posterior_interval_from_prevalence_set,
    selective_decisions,
    shiftguard_prevalence_set,
)

VISIBLE_INVALID_MECHANISMS = (
    "conditional_translation",
    "conditional_scale",
    "support_translation",
    "outcome_dependent_dropout",
    "covariance_shear",
    "tail_contamination",
    "nonlinear_bimodal",
)
CONCEPT_FAILURE_CONTROL = "unidentifiable_concept_reversal"


def _balanced_log_loss(target: np.ndarray, score: np.ndarray) -> float:
    probability = np.clip(np.asarray(score, dtype=np.float64), 1e-7, 1 - 1e-7)
    labels = np.asarray(target, dtype=np.int8)
    class_losses = []
    for label in (0, 1):
        selected = labels == label
        if not selected.any():
            return float("nan")
        if label:
            class_losses.append(float(-np.log(probability[selected]).mean()))
        else:
            class_losses.append(float(-np.log(1.0 - probability[selected]).mean()))
    return float(np.mean(class_losses))


def _ordinary_log_loss(target: np.ndarray, score: np.ndarray) -> float:
    probability = np.clip(np.asarray(score, dtype=np.float64), 1e-7, 1 - 1e-7)
    labels = np.asarray(target, dtype=np.int8)
    return float(
        -np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))
    )


def _diagnostic_views(
    latent_features: np.ndarray,
    observed: np.ndarray,
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not observed.any(axis=1).all():
        raise ValueError("Every synthetic record must retain at least one feature")
    core = np.where(observed, latent_features, 0.0)
    evidence = np.sum(core * signal[None, :], axis=1)
    diagnostic = compose_diagnostic_features(evidence, core, observed)
    return diagnostic, evidence


def draw_synthetic_shift_batch(
    *,
    size: int,
    prevalence: float,
    signal: np.ndarray,
    mechanism: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw a labelled test batch, retaining the unidentifiable failure control."""
    if size < 16 or not 0.0 < prevalence < 1.0:
        raise ValueError("Synthetic batch size or prevalence is invalid")
    rng = np.random.default_rng(seed)
    latent_labels = (rng.random(size) < prevalence).astype(np.int8)
    signed = 2.0 * latent_labels - 1.0
    latent = rng.normal(size=(size, len(signal))) + 0.5 * signed[:, None] * signal
    observed = rng.random(latent.shape) >= 0.10

    if mechanism == "pure_label_shift":
        outcome = latent_labels
    elif mechanism == "conditional_translation":
        latent[latent_labels == 1, -1] += 2.5
        outcome = latent_labels
    elif mechanism == "conditional_scale":
        selected = latent_labels == 0
        class_center = -0.5 * signal[1]
        latent[selected, 1] = class_center + 2.5 * (latent[selected, 1] - class_center)
        outcome = latent_labels
    elif mechanism == "support_translation":
        latent[:, -1] += 3.0
        outcome = latent_labels
    elif mechanism == "outcome_dependent_dropout":
        affected = (latent_labels == 1) & (rng.random(size) < 0.90)
        observed[affected, 0] = False
        outcome = latent_labels
    elif mechanism == "covariance_shear":
        selected = latent_labels == 1
        centered = latent[selected, 1] - 0.5 * signal[1]
        latent[selected, 0] += 1.75 * centered
        outcome = latent_labels
    elif mechanism == "tail_contamination":
        contaminated = rng.random(size) < 0.15
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=int(contaminated.sum()))
        latent[contaminated, -1] += 6.0 * signs
        outcome = latent_labels
    elif mechanism == "nonlinear_bimodal":
        selected = latent_labels == 1
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=int(selected.sum()))
        latent[selected, -1] += 3.0 * signs
        outcome = latent_labels
    elif mechanism == CONCEPT_FAILURE_CONTROL:
        # The unlabelled feature distribution is exactly a valid label-shift
        # distribution, but endpoint semantics are reversed. No unlabelled
        # diagnostic can distinguish this from the corresponding valid batch.
        outcome = 1 - latent_labels
    else:
        raise KeyError(f"Unknown synthetic shift mechanism: {mechanism}")
    empty = ~observed.any(axis=1)
    observed[empty, 0] = True
    diagnostic, evidence = _diagnostic_views(latent, observed, signal)
    return diagnostic, evidence, observed, outcome.astype(np.int8), latent_labels


def _draw_balanced_source(
    *,
    n_rows: int,
    signal: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_rows < 400 or n_rows % 2:
        raise ValueError("Synthetic source size must be even and at least 400")
    rng = np.random.default_rng(seed)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), n_rows // 2)
    rng.shuffle(labels)
    signed = 2.0 * labels - 1.0
    latent = rng.normal(size=(n_rows, len(signal))) + 0.5 * signed[:, None] * signal
    observed = rng.random(latent.shape) >= 0.10
    observed[~observed.any(axis=1), 0] = True
    diagnostic, _ = _diagnostic_views(latent, observed, signal)
    return diagnostic, observed, labels


def _stratified_three_way_indices(
    labels: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[], [], []]
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        first = len(indices) // 2
        second = first + len(indices) // 4
        for target, values in zip(
            partitions,
            (indices[:first], indices[first:second], indices[second:]),
            strict=True,
        ):
            target.extend(values.tolist())
    arrays = tuple(np.asarray(sorted(values), dtype=np.int64) for values in partitions)
    return arrays[0], arrays[1], arrays[2]


def _multi_size_calibration(
    reference_embeddings: dict[str, np.ndarray],
    reference_labels: np.ndarray,
    pool_embeddings: dict[str, np.ndarray],
    pool_labels: np.ndarray,
    *,
    prior_grid: np.ndarray,
    batch_sizes: tuple[int, ...],
    repetitions_per_prior: int,
    alpha: float,
    seed: int,
    discrepancy_reducers: dict[str, str] | None = None,
    discrepancy_regularizations: dict[str, float] | None = None,
) -> dict[tuple[str, int], ShiftGuardCalibration]:
    """Calibrate all batch sizes from shared held-out resampling episodes."""
    if repetitions_per_prior < 20:
        raise ValueError("At least 20 calibration repetitions are required")
    if set(reference_embeddings) != set(pool_embeddings):
        raise ValueError("Reference and calibration representations differ")
    reducers = discrepancy_reducers or {
        name: "mean_square" for name in reference_embeddings
    }
    if set(reducers) != set(reference_embeddings):
        raise ValueError("Every calibrated representation requires one discrepancy reducer")
    regularizations = discrepancy_regularizations or {
        name: 0.05 for name in reference_embeddings
    }
    if set(regularizations) != set(reference_embeddings):
        raise ValueError("Every calibrated representation requires one regularization value")
    for name in reference_embeddings:
        if len(reference_embeddings[name]) != len(reference_labels):
            raise ValueError(f"Reference representation {name} does not align")
        if len(pool_embeddings[name]) != len(pool_labels):
            raise ValueError(f"Calibration representation {name} does not align")
    rng = np.random.default_rng(seed)
    maximum_size = max(batch_sizes)
    positive_pool = np.flatnonzero(pool_labels == 1)
    negative_pool = np.flatnonzero(pool_labels == 0)
    critical_samples = {
        (name, size): np.empty(
            (len(prior_grid), repetitions_per_prior), dtype=np.float64
        )
        for name in reference_embeddings
        for size in batch_sizes
    }
    reference_means = {
        name: (
            values[reference_labels == 0].mean(axis=0),
            values[reference_labels == 1].mean(axis=0),
        )
        for name, values in reference_embeddings.items()
    }
    precision_matrices = {
        name: (
            fit_mixture_precision_matrices(
                values,
                reference_labels,
                prior_grid,
                regularization=regularizations[name],
            )
            if reducers[name] == "spectral_ridge"
            else None
        )
        for name, values in reference_embeddings.items()
    }
    for prior_index, prior in enumerate(prior_grid):
        target_labels = rng.random((repetitions_per_prior, maximum_size)) < prior
        positive_draws = positive_pool[
            rng.integers(0, len(positive_pool), size=target_labels.shape)
        ]
        negative_draws = negative_pool[
            rng.integers(0, len(negative_pool), size=target_labels.shape)
        ]
        indices = np.where(target_labels, positive_draws, negative_draws)
        for name, pool in pool_embeddings.items():
            cumulative = np.cumsum(pool[indices], axis=1)
            negative_mean, positive_mean = reference_means[name]
            mixture = negative_mean + prior * (positive_mean - negative_mean)
            for size in batch_sizes:
                target_mean = cumulative[:, size - 1] / size
                squared = np.square(target_mean - mixture[None, :])
                if reducers[name] == "mean_square":
                    discrepancy = np.mean(squared, axis=1)
                elif reducers[name] == "max_square":
                    discrepancy = np.max(squared, axis=1)
                elif reducers[name] == "spectral_ridge":
                    precision = precision_matrices[name]
                    if precision is None:
                        raise RuntimeError("Spectral-ridge precision was not fitted")
                    residual = target_mean - mixture[None, :]
                    discrepancy = (
                        size
                        * np.einsum(
                            "ri,ij,rj->r",
                            residual,
                            precision[prior_index],
                            residual,
                            optimize=True,
                        )
                        / target_mean.shape[1]
                    )
                else:
                    raise KeyError(
                        f"Unknown ShiftGuard discrepancy reducer: {reducers[name]}"
                    )
                critical_samples[(name, size)][prior_index] = discrepancy
    calibrations = {}
    for (name, size), values in critical_samples.items():
        critical = np.quantile(values, 1.0 - alpha, axis=1, method="higher")
        calibrations[(name, size)] = ShiftGuardCalibration(
            prior_grid=prior_grid.copy(),
            critical_values=np.asarray(critical, dtype=np.float64),
            alpha=alpha,
            target_size=size,
            repetitions_per_prior=repetitions_per_prior,
            discrepancy_reducer=reducers[name],
            discrepancy_regularization=regularizations[name],
            precision_matrices=precision_matrices[name],
            null_discrepancies=values,
        )
    return calibrations


def _target_mean_discrepancy_matrix(
    reference_embedding: np.ndarray,
    reference_labels: np.ndarray,
    target_means: np.ndarray,
    calibration: ShiftGuardCalibration,
) -> np.ndarray:
    """Evaluate every resampled target mean against every candidate prior."""
    negative = reference_embedding[reference_labels == 0].mean(axis=0)
    positive = reference_embedding[reference_labels == 1].mean(axis=0)
    mixtures = negative[None, :] + calibration.prior_grid[:, None] * (
        positive - negative
    )[None, :]
    residual = target_means[:, None, :] - mixtures[None, :, :]
    squared = np.square(residual)
    if calibration.discrepancy_reducer == "mean_square":
        return np.asarray(np.mean(squared, axis=2), dtype=np.float64)
    if calibration.discrepancy_reducer == "max_square":
        return np.asarray(np.max(squared, axis=2), dtype=np.float64)
    if calibration.discrepancy_reducer == "spectral_ridge":
        precision = calibration.precision_matrices
        if precision is None:
            raise ValueError("Global guard lacks spectral-ridge precision matrices")
        quadratic = np.einsum(
            "rpi,pij,rpj->rp", residual, precision, residual, optimize=True
        )
        return np.asarray(
            calibration.target_size * quadratic / reference_embedding.shape[1],
            dtype=np.float64,
        )
    raise KeyError(
        f"Unknown ShiftGuard discrepancy reducer: {calibration.discrepancy_reducer}"
    )


def calibrate_global_compatibility_guard(
    reference_embeddings: dict[str, np.ndarray],
    reference_labels: np.ndarray,
    pool_embeddings: dict[str, np.ndarray],
    pool_labels: np.ndarray,
    calibrations: dict[tuple[str, int], ShiftGuardCalibration],
    *,
    view_names: tuple[str, ...],
    prior_grid: np.ndarray,
    batch_sizes: tuple[int, ...],
    repetitions_per_prior: int,
    minimum_valid_acceptance: float,
    seed: int,
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    """Calibrate the minimum-over-priors global goodness-of-fit guard.

    This operational guard is separate from the 95% prevalence confidence set.
    It uses new source-only resampling draws and the declared uniform prior grid.
    """
    if (
        set(view_names) - set(reference_embeddings)
        or set(reference_embeddings) != set(pool_embeddings)
        or repetitions_per_prior < 20
        or not 0.0 < minimum_valid_acceptance < 1.0
    ):
        raise ValueError("Global compatibility guard configuration is invalid")
    rng = np.random.default_rng(seed)
    maximum_size = max(batch_sizes)
    positive_pool = np.flatnonzero(pool_labels == 1)
    negative_pool = np.flatnonzero(pool_labels == 0)
    guard_samples: dict[int, list[np.ndarray]] = {size: [] for size in batch_sizes}
    for generating_prior in prior_grid:
        target_labels = rng.random((repetitions_per_prior, maximum_size)) < generating_prior
        positive_draws = positive_pool[
            rng.integers(0, len(positive_pool), size=target_labels.shape)
        ]
        negative_draws = negative_pool[
            rng.integers(0, len(negative_pool), size=target_labels.shape)
        ]
        indices = np.where(target_labels, positive_draws, negative_draws)
        cumulative = {
            name: np.cumsum(pool_embeddings[name][indices], axis=1)
            for name in view_names
        }
        for size in batch_sizes:
            normalized_views = []
            for name in view_names:
                calibration = calibrations[(name, size)]
                target_means = cumulative[name][:, size - 1] / size
                discrepancies = _target_mean_discrepancy_matrix(
                    reference_embeddings[name],
                    reference_labels,
                    target_means,
                    calibration,
                )
                normalized_views.append(
                    discrepancies
                    / np.maximum(calibration.critical_values[None, :], 1e-12)
                )
            combined = np.max(np.stack(normalized_views), axis=0)
            guard_samples[size].append(np.min(combined, axis=1))
    thresholds = {}
    records = []
    for size, chunks in guard_samples.items():
        values = np.concatenate(chunks)
        threshold = float(
            np.quantile(values, minimum_valid_acceptance, method="higher")
        )
        thresholds[size] = threshold
        records.append(
            {
                "batch_size": size,
                "null_episodes": len(values),
                "minimum_valid_acceptance": minimum_valid_acceptance,
                "global_guard_threshold": threshold,
                "null_pass_rate_at_threshold": float(np.mean(values <= threshold)),
                "null_statistic_median": float(np.median(values)),
                "null_statistic_q95": float(np.quantile(values, 0.95)),
            }
        )
    return thresholds, records


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total < 1:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z / denominator * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return float(center - half), float(center + half)


def _episode_metrics(
    target: np.ndarray,
    evidence: np.ndarray,
    prevalence: PrevalenceSet,
    *,
    true_population_prevalence: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    zero_score = expit(evidence)
    point_score = expit(evidence + logit(prevalence.best_prior))
    gated_score = point_score if prevalence.accepted else zero_score
    prior_oracle_score = expit(evidence + logit(true_population_prevalence))
    if prevalence.accepted:
        lower, upper = posterior_interval_from_prevalence_set(evidence, prevalence)
        decisions = selective_decisions(lower, upper, threshold=0.5)
    else:
        lower = np.full(len(target), np.nan)
        upper = np.full(len(target), np.nan)
        decisions = np.full(len(target), -1, dtype=np.int8)
    selected = decisions >= 0
    metrics = {
        "accepted": prevalence.accepted,
        "set_lower": prevalence.lower,
        "set_upper": prevalence.upper,
        "set_width": (
            prevalence.upper - prevalence.lower
            if prevalence.lower is not None and prevalence.upper is not None
            else np.nan
        ),
        "best_prior": prevalence.best_prior,
        "best_prior_absolute_error": abs(
            prevalence.best_prior - true_population_prevalence
        ),
        "prior_covered": any(
            np.isclose(value, true_population_prevalence)
            for value in prevalence.accepted_priors
        ),
        "zero_shot_balanced_log_loss": _balanced_log_loss(target, zero_score),
        "always_point_balanced_log_loss": _balanced_log_loss(target, point_score),
        "gated_balanced_log_loss": _balanced_log_loss(target, gated_score),
        "prior_oracle_balanced_log_loss": _balanced_log_loss(target, prior_oracle_score),
        "zero_shot_log_loss": _ordinary_log_loss(target, zero_score),
        "always_point_log_loss": _ordinary_log_loss(target, point_score),
        "gated_log_loss": _ordinary_log_loss(target, gated_score),
        "prior_oracle_log_loss": _ordinary_log_loss(target, prior_oracle_score),
        "selective_coverage": float(selected.mean()),
        "selective_error": (
            float(np.mean(decisions[selected] != target[selected])) if selected.any() else np.nan
        ),
        "mean_posterior_interval_width": (
            float(np.mean(upper - lower)) if prevalence.accepted else np.nan
        ),
    }
    scores = {
        "zero_shot_score": zero_score,
        "always_point_score": point_score,
        "gated_score": gated_score,
        "prior_oracle_score": prior_oracle_score,
        "posterior_lower": lower,
        "posterior_upper": upper,
        "selective_decision": decisions,
    }
    return metrics, scores


def _aggregate_episode_results(episodes: pd.DataFrame) -> pd.DataFrame:
    return (
        episodes.groupby(["representation", "mechanism", "batch_size"], as_index=False)
        .agg(
            episodes=("episode_id", "size"),
            seeds=("seed", "nunique"),
            acceptance_rate=("accepted", "mean"),
            prior_coverage=("prior_covered", "mean"),
            prior_mae=("best_prior_absolute_error", "mean"),
            mean_set_width=("set_width", "mean"),
            mean_selective_coverage=("selective_coverage", "mean"),
            mean_selective_error=("selective_error", "mean"),
            zero_shot_bll=("zero_shot_balanced_log_loss", "mean"),
            always_point_bll=("always_point_balanced_log_loss", "mean"),
            gated_bll=("gated_balanced_log_loss", "mean"),
            prior_oracle_bll=("prior_oracle_balanced_log_loss", "mean"),
            zero_shot_log_loss=("zero_shot_log_loss", "mean"),
            always_point_log_loss=("always_point_log_loss", "mean"),
            gated_log_loss=("gated_log_loss", "mean"),
            prior_oracle_log_loss=("prior_oracle_log_loss", "mean"),
        )
        .sort_values(["representation", "mechanism", "batch_size"])
    )


def _evaluate_gates(episodes: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    primary = str(config["primary_representation"])
    selected = episodes.loc[episodes["representation"].eq(primary)]
    valid = selected.loc[selected["mechanism"].eq("pure_label_shift")]
    invalid = selected.loc[selected["mechanism"].isin(VISIBLE_INVALID_MECHANISMS)]
    concept = selected.loc[selected["mechanism"].eq(CONCEPT_FAILURE_CONTROL)]
    valid_acceptance = float(valid["accepted"].mean())
    invalid_acceptance = float(invalid["accepted"].mean())
    valid_uninformative = float(1.0 - valid["selective_coverage"].mean())
    invalid_harm = float(
        (
            invalid["gated_log_loss"]
            - invalid["zero_shot_log_loss"]
        ).mean()
    )
    valid_ci = _wilson_interval(int(valid["accepted"].sum()), len(valid))
    invalid_ci = _wilson_interval(int(invalid["accepted"].sum()), len(invalid))
    thresholds = config["acceptance"]
    checks: dict[str, dict[str, Any]] = {
        "valid_shift_acceptance": {
            "observed": valid_acceptance,
            "wilson_95": valid_ci,
            "threshold": float(thresholds["minimum_valid_shift_acceptance"]),
            "passed": valid_acceptance
            >= float(thresholds["minimum_valid_shift_acceptance"]),
        },
        "observable_invalid_false_acceptance": {
            "observed": invalid_acceptance,
            "wilson_95": invalid_ci,
            "threshold": float(thresholds["maximum_invalid_shift_false_acceptance"]),
            "passed": invalid_acceptance
            <= float(thresholds["maximum_invalid_shift_false_acceptance"]),
        },
        "uninformative_interval_fraction": {
            "observed": valid_uninformative,
            "threshold": float(thresholds["maximum_uninformative_interval_fraction"]),
            "passed": valid_uninformative
            <= float(thresholds["maximum_uninformative_interval_fraction"]),
        },
        "harmful_invalid_adaptation_delta_log_loss": {
            "observed": invalid_harm,
            "threshold": float(thresholds.get("maximum_invalid_harm_delta", 0.0)),
            "passed": invalid_harm
            <= float(thresholds.get("maximum_invalid_harm_delta", 0.0)),
        },
    }
    return {
        "primary_representation": primary,
        "status": (
            "passed_all_synthetic_gates"
            if all(value["passed"] for value in checks.values())
            else "failed_one_or_more_synthetic_gates"
        ),
        "checks": checks,
        "unidentifiable_concept_control": {
            "acceptance_rate": float(concept["accepted"].mean()),
            "mean_gated_minus_zero_shot_bll": float(
                (
                    concept["gated_balanced_log_loss"]
                    - concept["zero_shot_balanced_log_loss"]
                ).mean()
            ),
            "mean_gated_minus_zero_shot_log_loss": float(
                (concept["gated_log_loss"] - concept["zero_shot_log_loss"]).mean()
            ),
            "expected_interpretation": (
                "Acceptance is not a defect claim: the unlabelled distribution is "
                "constructed to be indistinguishable from valid label shift."
            ),
        },
    }


def _representation_views(
    values: np.ndarray,
    *,
    raw_location: np.ndarray,
    raw_scale: np.ndarray,
    rff: RandomFourierDiagnostic,
    moments: PolynomialMomentDiagnostic,
    quantile_copula: QuantileCopulaDiagnostic,
    result: ShiftGuardFitResult,
    requested: tuple[str, ...],
) -> dict[str, np.ndarray]:
    raw = (values - raw_location) / raw_scale
    rff_values = rff.transform(values)
    moment_values = moments.transform(values)
    quantile_values = quantile_copula.transform(values)
    learned = encode_shiftguard(result, values)
    available = {
        "raw_linear": raw,
        "rff": rff_values,
        "moments": moment_values,
        "quantile_copula": quantile_values,
        "learned": learned,
        "hybrid": np.concatenate([learned, rff_values], axis=1),
    }
    unknown = set(requested) - set(available)
    if unknown:
        raise KeyError(f"Unknown diagnostic representations: {sorted(unknown)}")
    return {name: available[name] for name in requested}


def intersect_prevalence_sets(
    prevalence_sets: dict[str, PrevalenceSet],
    calibrations: dict[str, ShiftGuardCalibration],
) -> PrevalenceSet:
    """Intersect Bonferroni-calibrated views into one auditable omnibus set."""
    if not prevalence_sets or set(prevalence_sets) != set(calibrations):
        raise ValueError("Omnibus prevalence views and calibrations must align")
    names = tuple(prevalence_sets)
    prior_grid = calibrations[names[0]].prior_grid
    for name in names[1:]:
        if not np.array_equal(calibrations[name].prior_grid, prior_grid):
            raise ValueError("Omnibus prevalence grids differ")
    accepted = np.ones(len(prior_grid), dtype=bool)
    normalized_discrepancies = []
    for name in names:
        prevalence = prevalence_sets[name]
        calibration = calibrations[name]
        accepted &= prevalence.discrepancies <= calibration.critical_values
        normalized_discrepancies.append(
            prevalence.discrepancies / np.maximum(calibration.critical_values, 1e-12)
        )
    combined = np.max(np.stack(normalized_discrepancies), axis=0)
    accepted_priors = prior_grid[accepted]
    best_index = int(np.argmin(combined))
    return PrevalenceSet(
        accepted=bool(len(accepted_priors)),
        accepted_priors=tuple(float(value) for value in accepted_priors),
        lower=float(accepted_priors.min()) if len(accepted_priors) else None,
        upper=float(accepted_priors.max()) if len(accepted_priors) else None,
        best_prior=float(prior_grid[best_index]),
        best_discrepancy=float(combined[best_index]),
        discrepancies=combined,
    )


def fit_joint_omnibus_calibration(
    calibrations: dict[str, ShiftGuardCalibration],
    *,
    alpha: float,
) -> JointShiftGuardCalibration:
    """Calibrate a max-over-views statistic on shared null episodes.

    The view-specific null discrepancies are generated from the same held-out
    pure-label-shift draws.  Joint empirical calibration therefore retains their
    correlation rather than paying a Bonferroni penalty as if the views were
    independent.
    """
    if len(calibrations) < 2 or not 0.0 < alpha < 1.0:
        raise ValueError("Joint omnibus calibration requires multiple views and valid alpha")
    view_names = tuple(calibrations)
    reference = calibrations[view_names[0]]
    null_values = []
    for name in view_names:
        calibration = calibrations[name]
        if (
            not np.array_equal(calibration.prior_grid, reference.prior_grid)
            or calibration.target_size != reference.target_size
            or calibration.repetitions_per_prior != reference.repetitions_per_prior
        ):
            raise ValueError("Joint omnibus view calibrations differ")
        values = calibration.null_discrepancies
        expected = (len(reference.prior_grid), reference.repetitions_per_prior)
        if values is None or values.shape != expected or not np.isfinite(values).all():
            raise ValueError("Joint omnibus calibration lacks aligned null discrepancies")
        null_values.append(values)
    stacked = np.stack(null_values)
    scales = np.median(stacked, axis=2)
    scales = np.maximum(scales, 1e-12)
    combined_null = np.max(stacked / scales[:, :, None], axis=0)
    critical = np.quantile(combined_null, 1.0 - alpha, axis=1, method="higher")
    return JointShiftGuardCalibration(
        view_names=view_names,
        prior_grid=reference.prior_grid.copy(),
        view_scales=np.asarray(scales, dtype=np.float64),
        critical_values=np.asarray(critical, dtype=np.float64),
        alpha=alpha,
        target_size=reference.target_size,
        repetitions_per_prior=reference.repetitions_per_prior,
    )


def joint_omnibus_prevalence_set(
    prevalence_sets: dict[str, PrevalenceSet],
    calibration: JointShiftGuardCalibration,
) -> PrevalenceSet:
    """Invert the jointly calibrated correlated-view compatibility test."""
    if tuple(prevalence_sets) != calibration.view_names:
        raise ValueError("Joint omnibus prevalence views differ from calibration")
    discrepancies = np.stack(
        [prevalence_sets[name].discrepancies for name in calibration.view_names]
    )
    expected = (len(calibration.view_names), len(calibration.prior_grid))
    if discrepancies.shape != expected:
        raise ValueError("Joint omnibus discrepancy grids differ")
    combined = np.max(discrepancies / calibration.view_scales, axis=0)
    accepted_priors = calibration.prior_grid[
        combined <= calibration.critical_values
    ]
    best_index = int(np.argmin(combined / np.maximum(calibration.critical_values, 1e-12)))
    return PrevalenceSet(
        accepted=bool(len(accepted_priors)),
        accepted_priors=tuple(float(value) for value in accepted_priors),
        lower=float(accepted_priors.min()) if len(accepted_priors) else None,
        upper=float(accepted_priors.max()) if len(accepted_priors) else None,
        best_prior=float(calibration.prior_grid[best_index]),
        best_discrepancy=float(combined[best_index]),
        discrepancies=np.asarray(combined, dtype=np.float64),
    )


def run_shiftguard_synthetic_experiment(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    """Run all declared seeds and retain episode- and patient-level evidence."""
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "source_only_synthetic_falsification")
    signal = np.asarray(config["signal"], dtype=np.float64)
    seeds = tuple(int(value) for value in config["seeds"])
    batch_sizes = tuple(int(value) for value in config["batch_sizes"])
    prior_grid = np.asarray(config["prior_grid"], dtype=np.float64)
    mechanisms = tuple(str(value) for value in config["mechanisms"])
    representations_requested = tuple(str(value) for value in config["representations"])
    primary_representation = str(config["primary_representation"])
    intersection_name = str(config.get("intersection_name", ""))
    intersection_views = tuple(str(value) for value in config.get("intersection_views", []))
    intersection_calibration_mode = str(
        config.get("intersection_calibration", "bonferroni")
    )
    if primary_representation not in representations_requested and (
        not intersection_name or primary_representation != intersection_name
    ):
        raise ValueError("Primary representation is absent from representations")
    if bool(intersection_name) != bool(intersection_views):
        raise ValueError("Intersection name and views must be configured together")
    if set(intersection_views) - set(representations_requested):
        raise ValueError("An intersection view is absent from representations")
    if intersection_calibration_mode not in {"bonferroni", "joint_empirical"}:
        raise ValueError("Unknown omnibus calibration mode")
    episodes_per_cell = int(config["episodes_per_cell"])
    alpha = float(config["alpha"])
    calibration_repetitions = int(config["calibration_repetitions_per_prior"])
    discrepancy_reducers = {
        name: str(value)
        for name, value in config.get("discrepancy_reducers", {}).items()
    }
    if not discrepancy_reducers:
        discrepancy_reducers = {
            name: "mean_square" for name in representations_requested
        }
    discrepancy_regularizations = {
        name: float(value)
        for name, value in config.get("discrepancy_regularizations", {}).items()
    }
    if not discrepancy_regularizations:
        discrepancy_regularizations = {
            name: 0.05 for name in representations_requested
        }
    device = str(config["device"])
    episode_records: list[dict[str, Any]] = []
    sample_records: list[pd.DataFrame] = []
    history_records: list[dict[str, Any]] = []
    fit_records = []
    guard_records: list[dict[str, Any]] = []

    for seed in seeds:
        diagnostic, _source_masks, source_labels = _draw_balanced_source(
            n_rows=int(config["source_rows"]), signal=signal, seed=seed
        )
        training_index, validation_index, calibration_index = _stratified_three_way_indices(
            source_labels, seed=seed + 11
        )
        mask_start = 1 + len(signal)
        mask_indices = tuple(range(mask_start, mask_start + len(signal)))
        result = fit_shiftguard(
            diagnostic[training_index],
            source_labels[training_index],
            diagnostic[validation_index],
            source_labels[validation_index],
            parameters=dict(config["shiftguard"]),
            seed=seed,
            device=device,
            training_mask_feature_indices=mask_indices,
            validation_mask_feature_indices=mask_indices,
        )
        fit_records.append(
            {
                "seed": seed,
                "best_epoch": result.best_epoch,
                "validation_score": result.validation_score,
                "parameter_count": result.parameter_count,
                "device": result.device,
            }
        )
        for row in result.history:
            history_records.append({"seed": seed, **row})
        rff = fit_random_fourier_diagnostic(
            diagnostic[training_index],
            n_components=int(config["rff_components"]),
            seed=seed + 31,
            include_linear=True,
        )
        moments = fit_polynomial_moment_diagnostic(diagnostic[training_index])
        quantile_copula = fit_quantile_copula_diagnostic(
            diagnostic[training_index],
            continuous_feature_count=1 + len(signal),
        )
        raw_location = diagnostic[training_index].mean(axis=0)
        raw_scale = diagnostic[training_index].std(axis=0)
        raw_scale = np.where(raw_scale > 1e-8, raw_scale, 1.0)

        reference_index = np.concatenate([training_index, validation_index])
        representation_arguments = {
            "raw_location": raw_location,
            "raw_scale": raw_scale,
            "rff": rff,
            "moments": moments,
            "quantile_copula": quantile_copula,
            "result": result,
            "requested": representations_requested,
        }
        reference_embeddings = _representation_views(
            diagnostic[reference_index], **representation_arguments
        )
        pool_embeddings = _representation_views(
            diagnostic[calibration_index], **representation_arguments
        )
        reference_labels = source_labels[reference_index]
        pool_labels = source_labels[calibration_index]
        calibrations = _multi_size_calibration(
            reference_embeddings,
            reference_labels,
            pool_embeddings,
            pool_labels,
            prior_grid=prior_grid,
            batch_sizes=batch_sizes,
            repetitions_per_prior=calibration_repetitions,
            alpha=alpha,
            seed=seed + 41,
            discrepancy_reducers=discrepancy_reducers,
            discrepancy_regularizations=discrepancy_regularizations,
        )
        intersection_calibrations = (
            _multi_size_calibration(
                {name: reference_embeddings[name] for name in intersection_views},
                reference_labels,
                {name: pool_embeddings[name] for name in intersection_views},
                pool_labels,
                prior_grid=prior_grid,
                batch_sizes=batch_sizes,
                repetitions_per_prior=calibration_repetitions,
                alpha=(
                    alpha / len(intersection_views)
                    if intersection_calibration_mode == "bonferroni"
                    else alpha
                ),
                seed=seed + 43,
                discrepancy_reducers={
                    name: discrepancy_reducers[name] for name in intersection_views
                },
                discrepancy_regularizations={
                    name: discrepancy_regularizations[name] for name in intersection_views
                },
            )
            if intersection_views
            else {}
        )
        joint_calibrations = (
            {
                size: fit_joint_omnibus_calibration(
                    {
                        name: intersection_calibrations[(name, size)]
                        for name in intersection_views
                    },
                    alpha=alpha,
                )
                for size in batch_sizes
            }
            if intersection_views and intersection_calibration_mode == "joint_empirical"
            else {}
        )
        guard_config = config.get("global_compatibility_guard")
        guard_thresholds: dict[int, float] = {}
        if guard_config:
            if not intersection_views or intersection_calibration_mode != "bonferroni":
                raise ValueError(
                    "Global compatibility guard currently requires a Bonferroni omnibus"
                )
            guard_thresholds, seed_guard_records = calibrate_global_compatibility_guard(
                {name: reference_embeddings[name] for name in intersection_views},
                reference_labels,
                {name: pool_embeddings[name] for name in intersection_views},
                pool_labels,
                intersection_calibrations,
                view_names=intersection_views,
                prior_grid=prior_grid,
                batch_sizes=batch_sizes,
                repetitions_per_prior=int(guard_config["repetitions_per_prior"]),
                minimum_valid_acceptance=float(
                    guard_config["minimum_valid_acceptance"]
                ),
                seed=seed + 47,
            )
            guard_records.extend({"seed": seed, **row} for row in seed_guard_records)
        feature_names = tuple(f"synthetic_feature_{index}" for index in range(len(signal)))
        for batch_size in batch_sizes:
            for mechanism_index, mechanism in enumerate(mechanisms):
                for episode_index in range(episodes_per_cell):
                    nominal_prior = float(
                        prior_grid[(episode_index + mechanism_index) % len(prior_grid)]
                    )
                    episode_seed = (
                        seed * 10_000_000
                        + batch_size * 10_000
                        + mechanism_index * 1_000
                        + episode_index
                    )
                    target_diagnostic, evidence, target_masks, target, latent_labels = (
                        draw_synthetic_shift_batch(
                            size=batch_size,
                            prevalence=nominal_prior,
                            signal=signal,
                            mechanism=mechanism,
                            seed=episode_seed,
                        )
                    )
                    target_embeddings = _representation_views(
                        target_diagnostic, **representation_arguments
                    )
                    true_population_prior = (
                        1.0 - nominal_prior
                        if mechanism == CONCEPT_FAILURE_CONTROL
                        else nominal_prior
                    )
                    episode_id = (
                        f"s{seed}-n{batch_size}-{mechanism}-e{episode_index}"
                    )
                    prevalence_sets = {
                        representation: shiftguard_prevalence_set(
                            reference_embeddings[representation],
                            reference_labels,
                            target_embedding,
                            calibrations[(representation, batch_size)],
                        )
                        for representation, target_embedding in target_embeddings.items()
                    }
                    if intersection_views:
                        strict_sets = {
                            name: shiftguard_prevalence_set(
                                reference_embeddings[name],
                                reference_labels,
                                target_embeddings[name],
                                intersection_calibrations[(name, batch_size)],
                            )
                            for name in intersection_views
                        }
                        strict_calibrations = {
                            name: intersection_calibrations[(name, batch_size)]
                            for name in intersection_views
                        }
                        prevalence_sets[intersection_name] = (
                            joint_omnibus_prevalence_set(
                                strict_sets, joint_calibrations[batch_size]
                            )
                            if intersection_calibration_mode == "joint_empirical"
                            else intersect_prevalence_sets(strict_sets, strict_calibrations)
                        )
                        pre_guard_set = prevalence_sets[intersection_name]
                        global_guard_passed = True
                        if guard_thresholds:
                            global_guard_passed = bool(
                                pre_guard_set.best_discrepancy
                                <= guard_thresholds[batch_size]
                            )
                            if not global_guard_passed:
                                prevalence_sets[intersection_name] = PrevalenceSet(
                                    accepted=False,
                                    accepted_priors=(),
                                    lower=None,
                                    upper=None,
                                    best_prior=pre_guard_set.best_prior,
                                    best_discrepancy=pre_guard_set.best_discrepancy,
                                    discrepancies=pre_guard_set.discrepancies,
                                )
                    else:
                        pre_guard_set = None
                        global_guard_passed = True
                    for representation, prevalence_set in prevalence_sets.items():
                        metrics, scores = _episode_metrics(
                            target,
                            evidence,
                            prevalence_set,
                            true_population_prevalence=true_population_prior,
                        )
                        episode_records.append(
                            {
                                "seed": seed,
                                "episode_id": episode_id,
                                "episode_index": episode_index,
                                "batch_size": batch_size,
                                "mechanism": mechanism,
                                "representation": representation,
                                "nominal_latent_prevalence": nominal_prior,
                                "true_population_prevalence": true_population_prior,
                                "realized_outcome_prevalence": float(target.mean()),
                                "realized_latent_prevalence": float(latent_labels.mean()),
                                "best_discrepancy": prevalence_set.best_discrepancy,
                                "pre_guard_set_nonempty": (
                                    pre_guard_set.accepted
                                    if representation == intersection_name
                                    and pre_guard_set is not None
                                    else np.nan
                                ),
                                "global_guard_passed": (
                                    global_guard_passed
                                    if representation == intersection_name
                                    else np.nan
                                ),
                                "global_guard_threshold": (
                                    guard_thresholds.get(batch_size, np.nan)
                                    if representation == intersection_name
                                    else np.nan
                                ),
                                **metrics,
                            }
                        )
                        if representation == primary_representation:
                            samples = pd.DataFrame(
                                {
                                    "seed": seed,
                                    "episode_id": episode_id,
                                    "sample_index": np.arange(batch_size),
                                    "batch_size": batch_size,
                                    "mechanism": mechanism,
                                    "target": target,
                                    "latent_label": latent_labels,
                                    "evidence_logit": evidence,
                                    "accepted": prevalence_set.accepted,
                                    **scores,
                                }
                            )
                            samples["observed_mask_sha256"] = observed_mask_hashes(
                                target_masks, feature_names
                            )
                            sample_records.append(samples)

    episodes = pd.DataFrame(episode_records)
    samples = pd.concat(sample_records, ignore_index=True)
    aggregate = _aggregate_episode_results(episodes)
    gates = _evaluate_gates(episodes, config)
    paths = {
        "episodes": run_dir / "episode_results.parquet",
        "samples": run_dir / "sample_predictions.parquet",
        "aggregate": run_dir / "aggregate_results.csv",
        "fits": run_dir / "fit_summaries.csv",
        "history": run_dir / "training_history.csv",
        "gates": run_dir / "synthetic_gates.json",
        "audit": run_dir / "evidence_audit.json",
    }
    if guard_records:
        paths["guard_calibration"] = run_dir / "global_guard_calibration.csv"
    episodes.to_parquet(paths["episodes"], index=False)
    samples.to_parquet(paths["samples"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    pd.DataFrame(fit_records).to_csv(paths["fits"], index=False)
    pd.DataFrame(history_records).to_csv(paths["history"], index=False)
    if guard_records:
        pd.DataFrame(guard_records).to_csv(paths["guard_calibration"], index=False)
    paths["gates"].write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hashes = {
        path.name: sha256_file(path)
        for name, path in paths.items()
        if name != "audit" and path.is_file()
    }
    paths["audit"].write_text(
        json.dumps(
            {
                "status": "complete_source_only_synthetic_evidence",
                "config_sha256": config_hash(config),
                "episode_rows": len(episodes),
                "sample_rows": len(samples),
                "sha256": hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
