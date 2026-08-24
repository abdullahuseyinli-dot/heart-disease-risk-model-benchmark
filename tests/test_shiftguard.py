from __future__ import annotations

import numpy as np
import pytest

from heartshift.models.shiftguard import (
    calibrate_shiftguard_prevalence_set,
    compose_diagnostic_features,
    fit_polynomial_moment_diagnostic,
    fit_quantile_copula_diagnostic,
    fit_random_fourier_diagnostic,
    fit_shiftguard,
    generate_shift_episodes,
    posterior_interval_from_prevalence_set,
    selective_decisions,
    shiftguard_prevalence_set,
)


def _source(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], 150)
    features = rng.normal(0.0, 0.5, size=(len(labels), 4))
    features[:, 0] += 1.5 * labels
    features[:, 1] -= 0.8 * labels
    return features, labels


def test_diagnostic_feature_composition_requires_aligned_views() -> None:
    evidence = np.arange(5, dtype=float)
    core = np.ones((5, 2))
    mask = np.ones((5, 3), dtype=bool)
    features = compose_diagnostic_features(evidence, core, mask)
    assert features.shape == (5, 6)


def test_shift_episode_generator_preserves_visible_failure_labels() -> None:
    features, labels = _source()
    episodes = generate_shift_episodes(
        features,
        labels,
        episode_count=12,
        target_size=64,
        valid_fraction=0.5,
        seed=4,
    )
    assert sum(episode.valid_label_shift for episode in episodes) == 6
    assert {episode.mechanism for episode in episodes if not episode.valid_label_shift}


def test_prevalence_set_accepts_label_shift_and_rejects_large_support_shift() -> None:
    source, labels = _source()
    calibration = calibrate_shiftguard_prevalence_set(
        source,
        labels,
        prior_grid=np.linspace(0.1, 0.9, 17),
        target_size=128,
        repetitions_per_prior=40,
        alpha=0.05,
        seed=10,
    )
    rng = np.random.default_rng(11)
    positive = source[labels == 1]
    negative = source[labels == 0]
    target_labels = rng.random(128) < 0.7
    target = np.empty((128, source.shape[1]))
    target[target_labels] = positive[rng.integers(0, len(positive), target_labels.sum())]
    target[~target_labels] = negative[rng.integers(0, len(negative), (~target_labels).sum())]
    accepted = shiftguard_prevalence_set(source, labels, target, calibration)
    rejected = shiftguard_prevalence_set(source, labels, target + 8.0, calibration)
    assert accepted.accepted
    assert accepted.lower is not None and accepted.upper is not None
    assert not rejected.accepted


def test_posterior_set_produces_selective_abstention() -> None:
    source, labels = _source()
    calibration = calibrate_shiftguard_prevalence_set(
        source,
        labels,
        prior_grid=np.asarray([0.2, 0.5, 0.8]),
        target_size=64,
        repetitions_per_prior=20,
        alpha=0.05,
        seed=2,
    )
    # Force a transparent accepted interval to test the decision layer.
    calibration = type(calibration)(
        prior_grid=calibration.prior_grid,
        critical_values=np.full(3, 1e9),
        alpha=calibration.alpha,
        target_size=calibration.target_size,
        repetitions_per_prior=calibration.repetitions_per_prior,
    )
    prevalence = shiftguard_prevalence_set(source, labels, source[:64], calibration)
    lower, upper = posterior_interval_from_prevalence_set(np.asarray([-5.0, 0.0, 5.0]), prevalence)
    decisions = selective_decisions(lower, upper, threshold=0.5)
    assert decisions.tolist() == [0, -1, 1]


def test_shiftguard_training_smoke_is_finite() -> None:
    source, labels = _source()
    result = fit_shiftguard(
        source[:200],
        labels[:200],
        source[200:],
        labels[200:],
        parameters={
            "hidden_dim": 16,
            "embedding_dim": 4,
            "episode_count": 8,
            "target_size": 32,
            "max_epochs": 20,
            "patience": 5,
        },
        seed=3,
        device="cpu",
    )
    assert np.isfinite(result.validation_score)
    assert result.best_epoch >= 1


def test_random_fourier_diagnostic_is_source_fitted_and_deterministic() -> None:
    source, _ = _source()
    first = fit_random_fourier_diagnostic(source, n_components=32, seed=9)
    second = fit_random_fourier_diagnostic(source, n_components=32, seed=9)
    first_embedding = first.transform(source[:20])
    second_embedding = second.transform(source[:20])
    assert first_embedding.shape == (20, source.shape[1] + 32)
    assert np.isfinite(first_embedding).all()
    assert np.allclose(first_embedding, second_embedding)
    assert first.bandwidth > 0.0


def test_calibration_can_use_a_disjoint_episode_pool() -> None:
    source, labels = _source()
    calibration = calibrate_shiftguard_prevalence_set(
        source[:200],
        labels[:200],
        calibration_pool_embedding=source[200:],
        calibration_pool_labels=labels[200:],
        prior_grid=np.asarray([0.2, 0.5, 0.8]),
        target_size=32,
        repetitions_per_prior=20,
        alpha=0.05,
        seed=4,
    )
    assert calibration.critical_values.shape == (3,)
    assert np.isfinite(calibration.critical_values).all()


def test_polynomial_diagnostic_contains_standardized_second_moments() -> None:
    source, _ = _source()
    diagnostic = fit_polynomial_moment_diagnostic(source)
    transformed = diagnostic.transform(source)
    expected_moments = source.shape[1] * (source.shape[1] + 1) // 2
    assert transformed.shape == (len(source), source.shape[1] + expected_moments)
    assert np.isfinite(transformed).all()


def test_quantile_copula_diagnostic_contains_marginal_and_pairwise_indicators() -> None:
    source, _ = _source()
    diagnostic = fit_quantile_copula_diagnostic(
        source,
        continuous_feature_count=3,
    )
    transformed = diagnostic.transform(source[:20])
    expected = 3 * 9 + 3 * 9 + (source.shape[1] - 3)
    assert transformed.shape == (20, expected)
    assert np.isfinite(transformed).all()


def test_max_discrepancy_reducer_is_calibrated_and_used_at_inference() -> None:
    source, labels = _source()
    calibration = calibrate_shiftguard_prevalence_set(
        source,
        labels,
        prior_grid=np.asarray([0.2, 0.5, 0.8]),
        target_size=32,
        repetitions_per_prior=20,
        alpha=0.05,
        seed=12,
        discrepancy_reducer="max_square",
    )
    result = shiftguard_prevalence_set(source, labels, source[:32], calibration)
    assert calibration.discrepancy_reducer == "max_square"
    assert result.discrepancies.shape == (3,)


def test_spectral_ridge_reducer_is_source_fitted_and_used_at_inference() -> None:
    source, labels = _source()
    calibration = calibrate_shiftguard_prevalence_set(
        source,
        labels,
        prior_grid=np.asarray([0.2, 0.5, 0.8]),
        target_size=32,
        repetitions_per_prior=20,
        alpha=0.05,
        seed=21,
        discrepancy_reducer="spectral_ridge",
        discrepancy_regularization=0.1,
    )
    result = shiftguard_prevalence_set(source, labels, source[:32], calibration)
    assert calibration.discrepancy_reducer == "spectral_ridge"
    assert calibration.discrepancy_regularization == 0.1
    assert calibration.precision_matrices is not None
    assert calibration.precision_matrices.shape == (3, source.shape[1], source.shape[1])
    assert np.isfinite(result.discrepancies).all()


def test_shiftguard_rejects_target_size_mismatch() -> None:
    source, labels = _source()
    calibration = calibrate_shiftguard_prevalence_set(
        source,
        labels,
        prior_grid=np.asarray([0.2, 0.5, 0.8]),
        target_size=32,
        repetitions_per_prior=20,
        alpha=0.05,
        seed=4,
    )
    with pytest.raises(ValueError, match="batch size"):
        shiftguard_prevalence_set(source, labels, source[:31], calibration)
