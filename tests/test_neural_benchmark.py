from __future__ import annotations

from heartshift.evaluation.neural_benchmark import _parameter_grid
from heartshift.evaluation.neural_outer import evaluation_units


def test_parameter_grid_is_deterministic_and_preserves_base() -> None:
    combinations = _parameter_grid(
        {"fixed": 7, "learning_rate": 0.1},
        {"learning_rate": [0.01, 0.02], "dro_lambda": [0.25, 0.5]},
    )
    assert combinations == [
        {"fixed": 7, "learning_rate": 0.01, "dro_lambda": 0.25},
        {"fixed": 7, "learning_rate": 0.02, "dro_lambda": 0.25},
        {"fixed": 7, "learning_rate": 0.01, "dro_lambda": 0.5},
        {"fixed": 7, "learning_rate": 0.02, "dro_lambda": 0.5},
    ]


def test_outer_evaluation_units_match_policy_replication_rules() -> None:
    units = evaluation_units(3)
    assert len(units) == 19
    assert sum(policy.name == "natural" for policy, _ in units) == 1
    assert sum(policy.name == "mcar_30" for policy, _ in units) == 3
    assert sum(policy.name == "drop_advanced" for policy, _ in units) == 1
