from __future__ import annotations

import numpy as np
import pandas as pd

from heartshift.data.synthetic import SYNTHETIC_SCENARIOS, generate_synthetic_environments
from heartshift.data.uci import CORE_COLUMNS, FEATURE_COLUMNS
from heartshift.research.gates import evaluate_synthetic_gates


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
