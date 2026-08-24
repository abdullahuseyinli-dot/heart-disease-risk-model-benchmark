from __future__ import annotations

from pathlib import Path

from heartshift.data.uci import FEATURE_COLUMNS
from heartshift.evaluation.neural_benchmark import _parameter_grid
from heartshift.evaluation.neural_outer import _load_heart_outer_frames, evaluation_units

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_neural_outer_loader_excludes_locked_endpoint_until_scoring() -> None:
    source, target_unlabelled = _load_heart_outer_frames(
        REPO_ROOT / "data/processed/uci_heart_canonical_v1.parquet",
        outer_target="cleveland",
        feature_columns=FEATURE_COLUMNS,
    )
    assert len(source) == 920 - 303
    assert len(target_unlabelled) == 303
    assert "target" in source
    assert "target" not in target_unlabelled
    assert source["site"].ne("cleveland").all()
    assert target_unlabelled["site"].eq("cleveland").all()
