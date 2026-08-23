from __future__ import annotations

import numpy as np

from heartshift.adaptation import (
    estimate_target_prevalence_mlls,
    estimate_target_prevalence_soft_bbse,
    mixture_fit_diagnostic,
    posterior_from_evidence,
)


def test_mlls_recovers_prevalence_from_known_likelihood_ratios() -> None:
    rng = np.random.default_rng(4)
    target_prevalence = 0.7
    labels = rng.random(5000) < target_prevalence
    # N(+/-1, 1) has log likelihood ratio 2x.
    values = rng.normal(np.where(labels, 1.0, -1.0), 1.0)
    estimate = estimate_target_prevalence_mlls(2.0 * values)
    assert abs(estimate - target_prevalence) < 0.04
    posterior = posterior_from_evidence(2.0 * values, estimate)
    assert ((posterior > 0) & (posterior < 1)).all()


def test_soft_bbse_recovers_prior_from_class_conditional_scores() -> None:
    rng = np.random.default_rng(18)
    source_negative = np.clip(rng.normal(0.2, 0.05, 2000), 0.01, 0.99)
    source_positive = np.clip(rng.normal(0.8, 0.05, 2000), 0.01, 0.99)
    source_probability = np.concatenate([source_negative, source_positive])
    source_labels = np.concatenate(
        [np.zeros(len(source_negative), dtype=int), np.ones(len(source_positive), dtype=int)]
    )
    target_prevalence = 0.7
    target_labels = rng.random(4000) < target_prevalence
    target_probability = np.where(
        target_labels,
        rng.choice(source_positive, size=len(target_labels)),
        rng.choice(source_negative, size=len(target_labels)),
    )
    estimated = estimate_target_prevalence_soft_bbse(
        source_probability, source_labels, target_probability
    )
    assert abs(estimated - target_prevalence) < 0.04


def test_mixture_diagnostic_accepts_compatible_and_rejects_large_shift() -> None:
    rng = np.random.default_rng(11)
    negative = rng.normal(-1.0, 1.0, 500)
    positive = rng.normal(1.0, 1.0, 500)
    labels = rng.random(300) < 0.7
    compatible = rng.normal(np.where(labels, 1.0, -1.0), 1.0)
    accepted = mixture_fit_diagnostic(
        negative,
        positive,
        compatible,
        prior_grid=[0.5, 0.6, 0.7, 0.8, 0.9],
        bootstrap_repetitions=49,
        mixture_draws=2,
        seed=5,
    )
    rejected = mixture_fit_diagnostic(
        negative,
        positive,
        compatible + 6.0,
        prior_grid=[0.5, 0.7, 0.9],
        bootstrap_repetitions=49,
        mixture_draws=2,
        seed=5,
    )
    assert accepted.accepted_interval is not None
    assert not rejected.accepted_priors
