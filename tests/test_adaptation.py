from __future__ import annotations

import numpy as np

from heartshift.adaptation import (
    acquisition_aware_label_shift_diagnostic,
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


def test_multiview_diagnostic_accepts_empirical_mixture_and_rejects_core_shift() -> None:
    rng = np.random.default_rng(41)
    source_labels = (rng.random(800) < 0.45).astype(int)
    source_evidence = rng.normal(2.0 * source_labels - 1.0, 0.7)
    source_core = np.column_stack(
        [
            rng.normal(2.0 * source_labels - 1.0, 0.8),
            rng.normal(source_labels, 1.0),
            rng.integers(0, 4, size=len(source_labels)),
        ]
    )
    observation_probability = np.where(
        source_labels[:, None].astype(bool),
        np.array([0.85, 0.70, 0.90, 0.65]),
        np.array([0.95, 0.80, 0.85, 0.75]),
    )
    source_mask = rng.random(observation_probability.shape) < observation_probability

    target_labels = rng.random(500) < 0.75
    target_indices = np.empty(len(target_labels), dtype=int)
    for label in (0, 1):
        candidates = np.flatnonzero(source_labels == label)
        selected = target_labels == label
        target_indices[selected] = rng.choice(candidates, size=int(selected.sum()), replace=True)
    target_evidence = source_evidence[target_indices]
    target_core = source_core[target_indices]
    target_mask = source_mask[target_indices]
    source_ids = [f"source-{index}" for index in range(len(source_labels))]
    target_ids = [f"target-{index}" for index in range(len(target_labels))]
    prior_grid = [0.55, 0.65, 0.75, 0.85, 0.95]
    accepted = acquisition_aware_label_shift_diagnostic(
        source_evidence,
        source_labels,
        source_core,
        source_mask,
        target_evidence,
        target_core,
        target_mask,
        source_sample_ids=source_ids,
        target_sample_ids=target_ids,
        prior_grid=prior_grid,
        bootstrap_repetitions=99,
        rff_features_per_view=64,
        seed=17,
    )
    shifted_core = target_core.copy()
    shifted_core[:, 0] += 6.0
    rejected = acquisition_aware_label_shift_diagnostic(
        source_evidence,
        source_labels,
        source_core,
        source_mask,
        target_evidence,
        shifted_core,
        target_mask,
        source_sample_ids=source_ids,
        target_sample_ids=target_ids,
        prior_grid=prior_grid,
        bootstrap_repetitions=99,
        rff_features_per_view=64,
        seed=17,
    )
    assert accepted.accepted
    assert accepted.mask_support.passed
    assert not rejected.accepted
    assert not bool(rejected.table.set_index("view").loc["core", "accepted"])
