from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from heartshift.config import load_yaml
from heartshift.data.uci import sha256_file
from heartshift.research.gates import (
    evaluate_psmask_confirmation_gate,
    evaluate_psmask_pivot_selection,
    validate_source_only_run,
    verify_frozen_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_configs_freeze_developmental_selections() -> None:
    expected = {
        "classical_inner_confirm_v1.yaml": "artifacts/runs/classical-inner-v1-rerun3",
        "modern_inner_confirm_v1.yaml": "artifacts/runs/modern-inner-v1",
        "tabpfn_v3_inner_confirm_v1.yaml": "artifacts/runs/tabpfn-v3-inner-v1",
        "mirrams_inner_confirm_v1.yaml": "artifacts/runs/mirrams-inner-v1",
        "psmask_inner_confirm_v1.yaml": "artifacts/runs/psmask-inner-v1",
    }
    for filename, selection_run in expected.items():
        config = load_yaml(REPO_ROOT / "configs" / "benchmark" / filename)
        assert config["fixed_selection_run"] == selection_run


def test_v2_freeze_prespecifies_pivot_outer_runs_and_reports() -> None:
    config = load_yaml(REPO_ROOT / "configs/release/freeze_v2.yaml")
    pivot = config["psmask_pivot_selection"]
    assert pivot["expected_selected_experiment"] == "v5_mask_only_dro"
    assert pivot["reference_experiment"] == "v4_structured_policy_bank"
    assert config["psmask_v1_gate_record"].endswith("acceptance_gate_v1.json")
    assert config["psmask_pivot_record"].endswith("pivot_selection_v2.json")
    frozen = set(config["frozen_files"])
    assert {
        "configs/release/freeze_v2.yaml",
        "configs/reporting/heart_outer_v2.yaml",
        "configs/reporting/readmission_outer_v2.yaml",
        "pyproject.toml",
        "uv.lock",
    } <= frozen
    for relative in config["outer_configs"]:
        outer = load_yaml(REPO_ROOT / relative)
        assert outer["freeze_lock"] == "artifacts/locks/heartshift_candidate_v2.json"
        assert str(outer["locked_run_name"]).endswith("-v2")


def test_source_gate_rejects_outer_target_predictions(tmp_path: Path) -> None:
    sites = ["cleveland", "hungary", "switzerland", "va_long_beach"]
    validation_sites = ["hungary", "cleveland", "cleveland", "cleveland"]
    predictions = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(4)],
            "site": validation_sites,
            "target": [0, 1, 0, 1],
            "outer_target": sites,
            "inner_validation": validation_sites,
            "record_sha256": [f"hash-{index}" for index in range(4)],
            "config_sha256": ["config-hash"] * 4,
            "y_score": [0.1, 0.8, 0.2, 0.9],
        }
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"phase": "inner_source_only", "config_sha256": "config-hash"}),
        encoding="utf-8",
    )
    predictions.to_parquet(tmp_path / "inner_predictions.parquet", index=False)
    pd.DataFrame({"outer_target": sites}).to_csv(
        tmp_path / "selected_hyperparameters.csv", index=False
    )
    assert validate_source_only_run(tmp_path)["prediction_rows"] == 4

    predictions.loc[0, "site"] = "cleveland"
    predictions.loc[0, "inner_validation"] = "cleveland"
    predictions.to_parquet(tmp_path / "inner_predictions.parquet", index=False)
    with pytest.raises(AssertionError, match="locked outer target"):
        validate_source_only_run(tmp_path)


def test_frozen_candidate_detects_method_changes(tmp_path: Path) -> None:
    method = tmp_path / "method.py"
    method.write_text("value = 1\n", encoding="utf-8")
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "status": "frozen_pre_outer",
                "git_commit": "unavailable",
                "frozen_files": [{"path": "method.py", "sha256": sha256_file(method)}],
                "source_evidence": {},
            }
        ),
        encoding="utf-8",
    )
    assert verify_frozen_candidate(tmp_path, lock_path)["status"] == "frozen_pre_outer"
    method.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="changed"):
        verify_frozen_candidate(tmp_path, lock_path)


def test_frozen_candidate_detects_pivot_record_changes(tmp_path: Path) -> None:
    pivot = tmp_path / "pivot.json"
    pivot.write_text('{"selected_experiment": "v5_mask_only_dro"}\n', encoding="utf-8")
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "status": "frozen_pre_outer",
                "git_commit": "unavailable",
                "frozen_files": [],
                "source_evidence": {},
                "psmask_pivot_record": "pivot.json",
                "psmask_pivot_record_sha256": sha256_file(pivot),
            }
        ),
        encoding="utf-8",
    )
    assert verify_frozen_candidate(tmp_path, lock_path)["status"] == "frozen_pre_outer"
    pivot.write_text('{"selected_experiment": "changed"}\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="pivot evidence changed"):
        verify_frozen_candidate(tmp_path, lock_path)


def test_psmask_confirmation_gate_checks_registered_tradeoffs(tmp_path: Path) -> None:
    experiments = {
        "v0_pooled_erm": (0.75, 0.78),
        "v2_prior_separated": (0.65, 0.77),
        "v4_structured_policy_bank": (0.60, 0.76),
        "v5_site_only_dro": (0.59, 0.757),
        "v5_mask_only_dro": (0.595, 0.756),
        "v5_site_mask_dro_brier": (0.58, 0.755),
        "v7_ane": (0.59, 0.75),
    }
    metrics = []
    fits = []
    selections = []
    for experiment, (loss, auc) in experiments.items():
        for outer in ("a", "b"):
            selections.append({"outer_target": outer, "experiment": experiment, "parameter_id": 0})
            for inner in ("x", "y"):
                for seed in (1, 2, 3):
                    fits.append(
                        {
                            "outer_target": outer,
                            "inner_validation": inner,
                            "experiment": experiment,
                            "seed": seed,
                        }
                    )
                    for policy in ("natural", "mcar_50"):
                        metrics.append(
                            {
                                "outer_target": outer,
                                "inner_validation": inner,
                                "experiment": experiment,
                                "parameter_id": 0,
                                "seed": seed,
                                "policy": policy,
                                "balanced_log_loss": loss,
                                "roc_auc": auc,
                            }
                        )
    pd.DataFrame(metrics).to_csv(tmp_path / "inner_metrics.csv", index=False)
    pd.DataFrame(fits).to_csv(tmp_path / "fit_summaries.csv", index=False)
    pd.DataFrame(selections).to_csv(tmp_path / "selected_configurations.csv", index=False)
    config = {
        "required_experiments": list(experiments),
        "minimum_confirmation_seeds": 3,
        "minimum_prior_separation_improvement": 0.01,
        "maximum_dro_macro_noninferiority_margin": 0.01,
        "minimum_dro_worst_cell_improvement": 0.0,
        "maximum_joint_axis_macro_noninferiority_margin": 0.01,
        "minimum_joint_axis_worst_cell_improvement": 0.0,
        "maximum_ane_natural_auroc_loss": 0.01,
    }
    assert evaluate_psmask_confirmation_gate(tmp_path, config)["passed"]
    pivot = evaluate_psmask_pivot_selection(
        tmp_path,
        {
            "protocol_version": "test-pivot-v2",
            "candidate_experiments": list(experiments),
            "reference_experiment": "v4_structured_policy_bank",
            "minimum_confirmation_seeds": 3,
            "maximum_natural_auroc_loss": 0.01,
            "maximum_worst_cell_regression": 0.0,
            "expected_selected_experiment": "v5_site_mask_dro_brier",
        },
    )
    assert pivot["passed"]
    assert pivot["selected_experiment"] == "v5_site_mask_dro_brier"
