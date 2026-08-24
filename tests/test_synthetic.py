from __future__ import annotations

import numpy as np
import pandas as pd

from heartshift.data.synthetic import SYNTHETIC_SCENARIOS, generate_synthetic_environments
from heartshift.data.uci import CORE_COLUMNS, FEATURE_COLUMNS
from heartshift.research.gates import evaluate_synthetic_gates, evaluate_synthetic_v3_gates


def test_synthetic_environments_are_reproducible_and_preserve_core_features() -> None:
    for scenario in SYNTHETIC_SCENARIOS:
        first = generate_synthetic_environments(scenario, n_per_environment=50, seed=9)
        second = generate_synthetic_environments(scenario, n_per_environment=50, seed=9)
        assert first.equals(second)
        assert len(first) == 200
        assert first["sample_id"].is_unique
        assert not first.loc[:, CORE_COLUMNS].isna().any().any()
        assert first.loc[:, FEATURE_COLUMNS].isna().any().any()


def test_prevalence_and_measurement_policy_shift_across_environments() -> None:
    data = generate_synthetic_environments("label_mar", n_per_environment=2000, seed=4)
    prevalence = data.groupby("site")["target"].mean().to_numpy()
    missingness = data.groupby("site")[list(FEATURE_COLUMNS)].apply(
        lambda frame: float(np.mean(frame.isna().to_numpy()))
    )
    assert np.all(np.diff(prevalence) > 0.1)
    assert np.all(np.diff(missingness.to_numpy()) > 0.02)


def test_v3_mechanisms_separate_label_acquisition_conditional_and_mnar_shift() -> None:
    label_only = generate_synthetic_environments("label_only", n_per_environment=3000, seed=8)
    mar_shift = generate_synthetic_environments("mar_policy_shift", n_per_environment=3000, seed=8)
    conditional = generate_synthetic_environments(
        "conditional_only", n_per_environment=3000, seed=8
    )
    mnar = generate_synthetic_environments("mnar_outcome_only", n_per_environment=3000, seed=8)

    measured = list(FEATURE_COLUMNS[3:])
    label_conditional_missingness = (
        label_only.groupby(["site", "target"])[measured]
        .apply(lambda frame: float(frame.isna().to_numpy().mean()))
        .unstack("target")
    )
    assert (
        label_conditional_missingness.max().max() - label_conditional_missingness.min().min() < 0.25
    )
    for target_value in (0, 1):
        rates = label_conditional_missingness[target_value].to_numpy()
        assert rates.max() - rates.min() < 0.04

    mar_site_missingness = mar_shift.groupby("site")[measured].apply(
        lambda frame: float(frame.isna().to_numpy().mean())
    )
    assert (
        mar_site_missingness.loc["synthetic_e3"] - mar_site_missingness.loc["synthetic_e2"] > 0.08
    )

    source_cp = conditional.loc[
        conditional["site"].eq("synthetic_e2") & conditional["target"].eq(1), "cp"
    ].mean()
    target_cp = conditional.loc[
        conditional["site"].eq("synthetic_e3") & conditional["target"].eq(1), "cp"
    ].mean()
    assert source_cp - target_cp > 1.0

    def outcome_missingness_gap(frame: pd.DataFrame) -> float:
        by_outcome = frame.groupby("target")[measured].apply(
            lambda group: float(group.isna().to_numpy().mean())
        )
        return float(by_outcome.loc[1] - by_outcome.loc[0])

    source_gap = outcome_missingness_gap(mnar.loc[mnar["site"].eq("synthetic_e2")])
    target_gap = outcome_missingness_gap(mnar.loc[mnar["site"].eq("synthetic_e3")])
    assert target_gap - source_gap > 0.20


def test_synthetic_gate_is_machine_checkable() -> None:
    summary = pd.DataFrame(
        {
            "scenario": ["label_mar", "conditional_shift", "mnar_outcome"],
            "experiment": ["candidate"] * 3,
            "seeds": [3, 3, 3],
            "diagnostic_acceptance_rate": [1.0, 0.0, 0.0],
            "mean_prevalence_absolute_error": [0.03, 0.4, 0.4],
            "mean_adapted_minus_equal_log_loss": [-0.2, 0.2, 0.2],
        }
    )
    config = {
        "adaptation_experiments": ["candidate"],
        "required_seeds_per_cell": 3,
        "label_mar_minimum_diagnostic_acceptance_rate": 0.5,
        "label_mar_maximum_mean_mlls_prevalence_absolute_error": 0.12,
        "label_mar_maximum_mean_adapted_minus_equal_log_loss": 0.0,
        "conditional_shift_maximum_diagnostic_acceptance_rate": 0.5,
        "mnar_outcome_maximum_diagnostic_acceptance_rate": 0.5,
    }
    assert evaluate_synthetic_gates(summary, config)["passed"]
    summary.loc[summary["scenario"].eq("conditional_shift"), "diagnostic_acceptance_rate"] = 1.0
    assert not evaluate_synthetic_gates(summary, config)["passed"]


def test_synthetic_v3_gate_is_machine_checkable() -> None:
    rows = []
    for scenario in (
        "label_only",
        "mar_policy_shift",
        "conditional_only",
        "mnar_outcome_only",
    ):
        for experiment in (
            "prior_separated",
            "structured_policy_bank",
            "mask_only_dro",
        ):
            rows.append(
                {
                    "scenario": scenario,
                    "experiment": experiment,
                    "variant": "test",
                    "seeds": 3,
                    "mean_balanced_log_loss": (0.59 if experiment == "mask_only_dro" else 0.60),
                    "mean_prevalence_absolute_error": 0.04,
                    "mean_adapted_minus_equal_log_loss": -0.10,
                    "diagnostic_acceptance_rate": 1.0 if scenario == "label_only" else 0.0,
                }
            )
    summary = pd.DataFrame(rows)
    config = {
        "required_experiments": [
            "prior_separated",
            "structured_policy_bank",
            "mask_only_dro",
        ],
        "adaptation_experiments": ["prior_separated", "mask_only_dro"],
        "required_seeds_per_cell": 3,
        "label_only_minimum_diagnostic_acceptance_rate": 2 / 3,
        "label_only_maximum_mean_mlls_prevalence_absolute_error": 0.12,
        "label_only_maximum_mean_adapted_minus_equal_log_loss": 0.0,
        "mar_robust_experiment": "mask_only_dro",
        "mar_reference_experiment": "structured_policy_bank",
        "mar_policy_maximum_robust_minus_reference_balanced_log_loss": 0.02,
        "mar_policy_shift_maximum_diagnostic_acceptance_rate": 0.5,
        "conditional_only_maximum_diagnostic_acceptance_rate": 0.5,
        "mnar_outcome_only_maximum_diagnostic_acceptance_rate": 0.5,
    }
    assert evaluate_synthetic_v3_gates(summary, config)["passed"]
    summary.loc[
        summary["scenario"].eq("conditional_only") & summary["experiment"].eq("mask_only_dro"),
        "diagnostic_acceptance_rate",
    ] = 1.0
    assert not evaluate_synthetic_v3_gates(summary, config)["passed"]
