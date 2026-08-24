from __future__ import annotations

import numpy as np

from heartshift.models.shiftguard import PrevalenceSet, ShiftGuardCalibration
from heartshift.research.shiftguard_experiment import (
    _multi_size_calibration,
    calibrate_global_compatibility_guard,
    draw_synthetic_shift_batch,
    fit_joint_omnibus_calibration,
    intersect_prevalence_sets,
    joint_omnibus_prevalence_set,
)


def test_multi_size_calibration_is_finite_and_nested_by_schema() -> None:
    rng = np.random.default_rng(4)
    labels = np.tile([0, 1], 100)
    values = rng.normal(size=(200, 3)) + labels[:, None]
    calibration = _multi_size_calibration(
        {"raw": values[:120]},
        labels[:120],
        {"raw": values[120:]},
        labels[120:],
        prior_grid=np.asarray([0.2, 0.5, 0.8]),
        batch_sizes=(16, 32),
        repetitions_per_prior=20,
        alpha=0.05,
        seed=8,
    )
    assert set(calibration) == {("raw", 16), ("raw", 32)}
    assert np.isfinite(calibration[("raw", 32)].critical_values).all()


def test_unidentifiable_control_has_same_features_but_reversed_endpoint() -> None:
    signal = np.asarray([1.0, -0.8, 0.4])
    valid = draw_synthetic_shift_batch(
        size=64,
        prevalence=0.7,
        signal=signal,
        mechanism="pure_label_shift",
        seed=11,
    )
    concept = draw_synthetic_shift_batch(
        size=64,
        prevalence=0.7,
        signal=signal,
        mechanism="unidentifiable_concept_reversal",
        seed=11,
    )
    assert np.array_equal(valid[0], concept[0])
    assert np.array_equal(valid[1], concept[1])
    assert np.array_equal(valid[2], concept[2])
    assert np.array_equal(valid[3], 1 - concept[3])


def test_omnibus_prevalence_set_intersects_accepted_priors() -> None:
    grid = np.asarray([0.2, 0.5, 0.8])
    calibration = ShiftGuardCalibration(grid, np.ones(3), 0.025, 32, 20)
    first = PrevalenceSet(True, (0.2, 0.5), 0.2, 0.5, 0.2, 0.1, np.asarray([0.1, 0.2, 2.0]))
    second = PrevalenceSet(True, (0.5, 0.8), 0.5, 0.8, 0.5, 0.1, np.asarray([2.0, 0.1, 0.2]))
    result = intersect_prevalence_sets(
        {"first": first, "second": second},
        {"first": calibration, "second": calibration},
    )
    assert result.accepted_priors == (0.5,)


def test_tail_and_bimodal_controls_preserve_binary_endpoint() -> None:
    signal = np.asarray([1.0, -0.8, 0.4])
    for mechanism in ("tail_contamination", "nonlinear_bimodal"):
        diagnostic, evidence, observed, target, latent = draw_synthetic_shift_batch(
            size=64,
            prevalence=0.5,
            signal=signal,
            mechanism=mechanism,
            seed=15,
        )
        assert diagnostic.shape[0] == len(evidence) == len(observed) == len(target)
        assert set(np.unique(target)) <= {0, 1}
        assert np.array_equal(target, latent)


def test_joint_omnibus_uses_aligned_empirical_null_draws() -> None:
    rng = np.random.default_rng(14)
    labels = np.tile([0, 1], 100)
    values = rng.normal(size=(200, 3)) + labels[:, None]
    calibrations = _multi_size_calibration(
        {"raw": values[:120], "scaled": 2.0 * values[:120]},
        labels[:120],
        {"raw": values[120:], "scaled": 2.0 * values[120:]},
        labels[120:],
        prior_grid=np.asarray([0.2, 0.5, 0.8]),
        batch_sizes=(32,),
        repetitions_per_prior=20,
        alpha=0.05,
        seed=18,
    )
    joint = fit_joint_omnibus_calibration(
        {name: calibrations[(name, 32)] for name in ("raw", "scaled")},
        alpha=0.05,
    )
    first = PrevalenceSet(
        True,
        (0.2, 0.5),
        0.2,
        0.5,
        0.2,
        0.1,
        np.asarray([0.1, 0.2, 20.0]),
    )
    second = PrevalenceSet(
        True,
        (0.2, 0.5),
        0.2,
        0.5,
        0.2,
        0.1,
        np.asarray([0.4, 0.8, 80.0]),
    )
    result = joint_omnibus_prevalence_set({"raw": first, "scaled": second}, joint)
    assert joint.view_scales.shape == (2, 3)
    assert joint.critical_values.shape == (3,)
    assert result.discrepancies.shape == (3,)
    assert np.isfinite(result.discrepancies).all()


def test_global_guard_calibrates_minimum_over_candidate_priors() -> None:
    rng = np.random.default_rng(24)
    labels = np.tile([0, 1], 100)
    values = rng.normal(size=(200, 3)) + labels[:, None]
    reference = {"raw": values[:120], "scaled": 2.0 * values[:120]}
    pool = {"raw": values[120:], "scaled": 2.0 * values[120:]}
    calibrations = _multi_size_calibration(
        reference,
        labels[:120],
        pool,
        labels[120:],
        prior_grid=np.asarray([0.2, 0.5, 0.8]),
        batch_sizes=(16,),
        repetitions_per_prior=20,
        alpha=0.025,
        seed=28,
    )
    thresholds, records = calibrate_global_compatibility_guard(
        reference,
        labels[:120],
        pool,
        labels[120:],
        calibrations,
        view_names=("raw", "scaled"),
        prior_grid=np.asarray([0.2, 0.5, 0.8]),
        batch_sizes=(16,),
        repetitions_per_prior=20,
        minimum_valid_acceptance=0.9,
        seed=29,
    )
    assert np.isfinite(thresholds[16])
    assert records[0]["null_episodes"] == 60
    assert records[0]["null_pass_rate_at_threshold"] >= 0.9
