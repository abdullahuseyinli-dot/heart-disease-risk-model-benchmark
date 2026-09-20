from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from heartshift.config import load_yaml
from heartshift.research.study_contract import (
    assert_dataset_use_allowed,
    validate_historical_seed_extension_contract,
    validate_router_development_contract,
    validate_shiftguard_study_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shiftguard_repository_contract_is_complete() -> None:
    config = load_yaml(REPO_ROOT / "configs/research/shiftguard_v1.yaml")
    result = validate_shiftguard_study_config(REPO_ROOT, config)
    assert result["status"] == "passed_shiftguard_study_contract"
    assert result["seed_count"] == 10
    assert result["named_omission_count"] >= 3


def test_consumed_heart_cannot_be_relabelled_as_confirmation() -> None:
    config = load_yaml(REPO_ROOT / "configs/research/shiftguard_v1.yaml")
    assert_dataset_use_allowed(
        config,
        dataset_id="uci_heart_outer_v5",
        phase="mechanism_development",
    )
    with pytest.raises(PermissionError, match="consumed"):
        assert_dataset_use_allowed(
            config,
            dataset_id="uci_heart_outer_v5",
            phase="locked_confirmation",
        )


def _router_contract_configs() -> list[dict[str, Any]]:
    return [
        load_yaml(REPO_ROOT / "configs/research/heart_controls_v1.yaml"),
        load_yaml(REPO_ROOT / "configs/research/observed_backbones_v1.yaml"),
        load_yaml(REPO_ROOT / "configs/research/observed_backbones_outer_v1.yaml"),
        load_yaml(REPO_ROOT / "configs/research/support_router_outer_v3.yaml"),
        load_yaml(REPO_ROOT / "configs/reporting/heart_research_development_v1.yaml"),
    ]


def test_router_development_contract_aligns_seeds_masks_and_strong_controls() -> None:
    result = validate_router_development_contract(*_router_contract_configs())
    assert result["status"] == "passed_router_development_contract"
    assert result["seed_count"] == 10
    assert result["router_feature_mode_count"] == 3


def test_router_development_contract_rejects_seed_drift() -> None:
    configs = _router_contract_configs()
    configs[2]["seeds"] = [5062]
    with pytest.raises(ValueError, match="seeds must match"):
        validate_router_development_contract(*configs)


def _seed_extension_contract_configs() -> list[dict[str, Any]]:
    return [
        load_yaml(REPO_ROOT / "configs/research/heart_controls_v1.yaml"),
        load_yaml(REPO_ROOT / "configs/research/historical_psmask_seed_extension_v1.yaml"),
        load_yaml(REPO_ROOT / "configs/reporting/historical_psmask_seed_extension_v1.yaml"),
    ]


def test_historical_seed_extension_contract_is_explicit_and_aligned() -> None:
    result = validate_historical_seed_extension_contract(*_seed_extension_contract_configs())
    assert result["status"] == "passed_historical_seed_extension_contract"
    assert result["seed_count"] == 10
    assert result["historical_seed_count"] == 3
    assert result["experiment_count"] == 10


def test_historical_seed_extension_contract_rejects_uda_activation() -> None:
    configs = _seed_extension_contract_configs()
    configs[1]["adaptable_variants"] = ["target_entropy_calibration"]
    with pytest.raises(ValueError, match="adaptation is prohibited"):
        validate_historical_seed_extension_contract(*configs)


def test_historical_seed_extension_contract_rejects_confirmatory_relabelling() -> None:
    configs = _seed_extension_contract_configs()
    configs[2]["status"] = "locked_confirmation"
    with pytest.raises(ValueError, match="cannot be relabelled"):
        validate_historical_seed_extension_contract(*configs)
