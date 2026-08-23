"""Objective acceptance gates for source-only and synthetic research evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from heartshift.data.uci import sha256_file


def evaluate_synthetic_gates(
    summary: pd.DataFrame,
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate thresholds declared in configuration before the experiment ran."""
    selected = summary.loc[summary["experiment"].isin(gate_config["adaptation_experiments"])].copy()
    if selected.empty:
        raise ValueError("No configured adaptation experiments occur in the summary")
    required_scenarios = {"label_mar", "conditional_shift", "mnar_outcome"}
    if set(selected["scenario"]) != required_scenarios:
        raise ValueError("Synthetic summary does not contain every required scenario")

    label = selected.loc[selected["scenario"].eq("label_mar")]
    conditional = selected.loc[selected["scenario"].eq("conditional_shift")]
    mnar = selected.loc[selected["scenario"].eq("mnar_outcome")]
    checks: list[dict[str, Any]] = [
        {
            "name": "required_seeds_per_cell",
            "observed": int(selected["seeds"].min()),
            "operator": ">=",
            "threshold": int(gate_config["required_seeds_per_cell"]),
        },
        {
            "name": "label_mar_diagnostic_acceptance",
            "observed": float(label["diagnostic_acceptance_rate"].min()),
            "operator": ">=",
            "threshold": float(gate_config["label_mar_minimum_diagnostic_acceptance_rate"]),
        },
        {
            "name": "label_mar_mlls_prevalence_error",
            "observed": float(label["mean_prevalence_absolute_error"].max()),
            "operator": "<=",
            "threshold": float(
                gate_config["label_mar_maximum_mean_mlls_prevalence_absolute_error"]
            ),
        },
        {
            "name": "label_mar_adapted_log_loss_delta",
            "observed": float(label["mean_adapted_minus_equal_log_loss"].max()),
            "operator": "<=",
            "threshold": float(gate_config["label_mar_maximum_mean_adapted_minus_equal_log_loss"]),
        },
        {
            "name": "conditional_shift_diagnostic_acceptance",
            "observed": float(conditional["diagnostic_acceptance_rate"].max()),
            "operator": "<=",
            "threshold": float(gate_config["conditional_shift_maximum_diagnostic_acceptance_rate"]),
        },
        {
            "name": "mnar_outcome_diagnostic_acceptance",
            "observed": float(mnar["diagnostic_acceptance_rate"].max()),
            "operator": "<=",
            "threshold": float(gate_config["mnar_outcome_maximum_diagnostic_acceptance_rate"]),
        },
    ]
    for check in checks:
        observed = float(check["observed"])
        threshold = float(check["threshold"])
        check["finite"] = bool(np.isfinite(observed))
        check["passed"] = bool(
            check["finite"]
            and (observed >= threshold if check["operator"] == ">=" else observed <= threshold)
        )
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "adaptation_experiments": list(gate_config["adaptation_experiments"]),
    }


def evaluate_psmask_confirmation_gate(
    run_dir: Path,
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate source-only architecture gates declared before confirmation."""
    metrics = pd.read_csv(run_dir / "inner_metrics.csv")
    fits = pd.read_csv(run_dir / "fit_summaries.csv")
    selections = pd.read_csv(run_dir / "selected_configurations.csv")
    selected_metrics = metrics.merge(
        selections[["outer_target", "experiment", "parameter_id"]],
        on=["outer_target", "experiment", "parameter_id"],
        how="inner",
        validate="many_to_one",
    )
    policy_means = selected_metrics.groupby(
        ["experiment", "outer_target", "policy"], as_index=False
    ).agg(
        balanced_log_loss=("balanced_log_loss", "mean"),
        roc_auc=("roc_auc", "mean"),
    )
    site_worst = policy_means.groupby(["experiment", "outer_target"], as_index=False).agg(
        worst_policy_balanced_log_loss=("balanced_log_loss", "max")
    )
    primary = site_worst.groupby("experiment", as_index=False).agg(
        macro_site_worst_balanced_log_loss=(
            "worst_policy_balanced_log_loss",
            "mean",
        ),
        worst_site_policy_balanced_log_loss=(
            "worst_policy_balanced_log_loss",
            "max",
        ),
    )
    natural = (
        policy_means.loc[policy_means["policy"].eq("natural")]
        .groupby("experiment", as_index=False)
        .agg(macro_natural_roc_auc=("roc_auc", "mean"))
    )
    summary = primary.merge(natural, on="experiment", validate="one_to_one").set_index("experiment")
    required = set(str(value) for value in gate_config["required_experiments"])
    if not required <= set(summary.index):
        raise ValueError(
            f"PS-MaskDRO confirmation misses experiments: {sorted(required - set(summary.index))}"
        )
    minimum_seed_count = int(
        fits.groupby(["outer_target", "inner_validation", "experiment"])["seed"].nunique().min()
    )
    pooled = summary.loc["v0_pooled_erm"]
    prior = summary.loc["v2_prior_separated"]
    structured = summary.loc["v4_structured_policy_bank"]
    site_dro = summary.loc["v5_site_only_dro"]
    mask_dro = summary.loc["v5_mask_only_dro"]
    dro = summary.loc["v5_site_mask_dro_brier"]
    ane = summary.loc["v7_ane"]
    checks: list[dict[str, Any]] = [
        {
            "name": "minimum_confirmation_seeds",
            "observed": minimum_seed_count,
            "operator": ">=",
            "threshold": int(gate_config["minimum_confirmation_seeds"]),
        },
        {
            "name": "prior_separation_macro_improvement",
            "observed": float(
                pooled["macro_site_worst_balanced_log_loss"]
                - prior["macro_site_worst_balanced_log_loss"]
            ),
            "operator": ">=",
            "threshold": float(gate_config["minimum_prior_separation_improvement"]),
        },
        {
            "name": "dro_macro_noninferiority",
            "observed": float(
                dro["macro_site_worst_balanced_log_loss"]
                - structured["macro_site_worst_balanced_log_loss"]
            ),
            "operator": "<=",
            "threshold": float(gate_config["maximum_dro_macro_noninferiority_margin"]),
        },
        {
            "name": "dro_worst_cell_improvement",
            "observed": float(
                structured["worst_site_policy_balanced_log_loss"]
                - dro["worst_site_policy_balanced_log_loss"]
            ),
            "operator": ">=",
            "threshold": float(gate_config["minimum_dro_worst_cell_improvement"]),
        },
        {
            "name": "joint_dro_axis_macro_noninferiority",
            "observed": float(
                dro["macro_site_worst_balanced_log_loss"]
                - min(
                    site_dro["macro_site_worst_balanced_log_loss"],
                    mask_dro["macro_site_worst_balanced_log_loss"],
                )
            ),
            "operator": "<=",
            "threshold": float(gate_config["maximum_joint_axis_macro_noninferiority_margin"]),
        },
        {
            "name": "joint_dro_axis_worst_cell_improvement",
            "observed": float(
                min(
                    site_dro["worst_site_policy_balanced_log_loss"],
                    mask_dro["worst_site_policy_balanced_log_loss"],
                )
                - dro["worst_site_policy_balanced_log_loss"]
            ),
            "operator": ">=",
            "threshold": float(gate_config["minimum_joint_axis_worst_cell_improvement"]),
        },
        {
            "name": "ane_natural_auroc_loss",
            "observed": float(dro["macro_natural_roc_auc"] - ane["macro_natural_roc_auc"]),
            "operator": "<=",
            "threshold": float(gate_config["maximum_ane_natural_auroc_loss"]),
        },
    ]
    for check in checks:
        observed = float(check["observed"])
        threshold = float(check["threshold"])
        check["finite"] = bool(np.isfinite(observed))
        check["passed"] = bool(
            check["finite"]
            and (observed >= threshold if check["operator"] == ">=" else observed <= threshold)
        )
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "summary": summary.reset_index().to_dict(orient="records"),
    }


def validate_source_only_run(run_dir: Path) -> dict[str, Any]:
    """Prove that an inner run contains predictions only from source hospitals."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete source-only run: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase = str(manifest["phase"])
    if phase == "readmission_source_only":
        return _validate_readmission_source_only_run(run_dir, manifest)
    predictions_path = run_dir / "inner_predictions.parquet"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Incomplete source-only run: {run_dir}")
    if phase not in {"inner_source_only", "psmask_inner_source_only"}:
        raise AssertionError(f"Unexpected source-run phase in {manifest_path}")
    predictions = pd.read_parquet(predictions_path)
    required = {
        "sample_id",
        "site",
        "target",
        "outer_target",
        "inner_validation",
        "record_sha256",
        "config_sha256",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise AssertionError(f"Missing source prediction columns: {sorted(missing)}")
    if predictions.empty:
        raise AssertionError("Source prediction table is empty")
    if predictions["site"].eq(predictions["outer_target"]).any():
        raise AssertionError("A locked outer target appears in source-only predictions")
    if not predictions["site"].eq(predictions["inner_validation"]).all():
        raise AssertionError("Inner validation provenance is inconsistent")
    if not predictions["target"].isin([0, 1]).all():
        raise AssertionError("Source predictions contain an invalid target")
    hashes = predictions["config_sha256"].drop_duplicates().tolist()
    if hashes != [manifest["config_sha256"]]:
        raise AssertionError("Prediction and manifest configuration hashes disagree")
    score_column = "y_score" if "y_score" in predictions else "evidence_logit"
    if not np.isfinite(predictions[score_column].to_numpy(dtype=float)).all():
        raise AssertionError("Source predictions contain non-finite scores")
    selection_candidates = (
        run_dir / "selected_hyperparameters.csv",
        run_dir / "selected_configurations.csv",
    )
    selection_path = next((path for path in selection_candidates if path.is_file()), None)
    if selection_path is None:
        raise FileNotFoundError(f"No selected configuration table in {run_dir}")
    selections = pd.read_csv(selection_path)
    expected_sites = {"cleveland", "hungary", "switzerland", "va_long_beach"}
    if set(selections["outer_target"]) != expected_sites:
        raise AssertionError("Selected configurations do not cover all four outer targets")
    evidence = {
        "run_dir": str(run_dir),
        "phase": manifest["phase"],
        "config_sha256": manifest["config_sha256"],
        "prediction_rows": len(predictions),
        "unique_samples": int(predictions["sample_id"].nunique()),
        "outer_targets": sorted(expected_sites),
        "predictions_sha256": sha256_file(predictions_path),
        "selections_sha256": sha256_file(selection_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    provenance_path = run_dir / "selection_provenance.json"
    if provenance_path.is_file():
        evidence["selection_provenance_sha256"] = sha256_file(provenance_path)
    return evidence


def _validate_readmission_source_only_run(
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    predictions_path = run_dir / "validation_predictions.parquet"
    selection_path = run_dir / "selected_configurations.csv"
    config_path = run_dir / "config.resolved.json"
    for path in (predictions_path, selection_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete readmission source-only run: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(predictions_path)
    required = {
        "sample_id",
        "source_line_sha256",
        "target",
        "split",
        "environment",
        "experiment",
        "policy",
        "mask_replicate",
        "config_sha256",
        "y_score",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise AssertionError(f"Missing readmission source prediction columns: {sorted(missing)}")
    if predictions.empty or set(predictions["split"]) != {str(config["data"]["validation_split"])}:
        raise AssertionError("Readmission inner predictions are not validation-only")
    if not predictions["target"].isin([0, 1]).all():
        raise AssertionError("Readmission source predictions contain an invalid target")
    if not np.isfinite(predictions["y_score"].to_numpy(dtype=float)).all():
        raise AssertionError("Readmission source predictions contain non-finite scores")
    if predictions["config_sha256"].drop_duplicates().tolist() != [manifest["config_sha256"]]:
        raise AssertionError("Readmission prediction and manifest hashes disagree")

    repo_root = run_dir.parents[2]
    split = pd.read_parquet(repo_root / config["data"]["split_path"])
    test_ids = set(
        split.loc[
            split["split"].isin(
                ["id_test", "ood_validation", "ood_test", "quarantined_cross_domain"]
            ),
            "sample_id",
        ]
    )
    if set(predictions["sample_id"]) & test_ids:
        raise AssertionError("A held-out readmission sample appears in source predictions")
    selections = pd.read_csv(selection_path)
    expected_experiments = {str(item["name"]) for item in config["experiments"]}
    if set(selections["experiment"]) != expected_experiments:
        raise AssertionError("Readmission selections do not cover every experiment")
    evidence = {
        "run_dir": str(run_dir),
        "phase": manifest["phase"],
        "config_sha256": manifest["config_sha256"],
        "prediction_rows": len(predictions),
        "unique_samples": int(predictions["sample_id"].nunique()),
        "validation_split": str(config["data"]["validation_split"]),
        "predictions_sha256": sha256_file(predictions_path),
        "selections_sha256": sha256_file(selection_path),
        "manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
    }
    provenance_path = run_dir / "selection_provenance.json"
    if provenance_path.is_file():
        evidence["selection_provenance_sha256"] = sha256_file(provenance_path)
    return evidence


def _tree_hash(paths: list[Path], repo_root: Path) -> tuple[str, list[dict[str, str]]]:
    records = []
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(repo_root).as_posix()
        file_hash = sha256_file(path)
        records.append({"path": relative, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), records


def _git_value(repo_root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def freeze_candidate(
    repo_root: Path,
    freeze_config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Validate all pre-outer gates and write an immutable candidate lock."""
    if output_path.exists():
        raise FileExistsError(f"Candidate lock already exists: {output_path}")
    from heartshift.cli.validate import validate_repository

    validate_repository(repo_root)
    tag = str(freeze_config["legacy_tag"])
    if _git_value(repo_root, "rev-parse", tag) == "unavailable":
        raise AssertionError(f"Legacy preservation tag is unavailable: {tag}")
    source_evidence = {}
    for name, relative in freeze_config["source_runs"].items():
        source_evidence[str(name)] = validate_source_only_run(repo_root / str(relative))
    psmask_gate = evaluate_psmask_confirmation_gate(
        repo_root / str(freeze_config["source_runs"]["psmask"]),
        freeze_config["psmask_acceptance_gates"],
    )
    if not bool(psmask_gate["passed"]):
        raise AssertionError(
            "Source-only PS-MaskDRO confirmation gate failed; "
            "locked outer evaluation remains closed"
        )
    synthetic_run = repo_root / str(freeze_config["synthetic_run"])
    synthetic_gate_path = synthetic_run / "acceptance_gate.json"
    if not synthetic_gate_path.is_file():
        raise FileNotFoundError(f"Synthetic gate is missing: {synthetic_gate_path}")
    synthetic_gate = json.loads(synthetic_gate_path.read_text(encoding="utf-8"))
    if not bool(synthetic_gate.get("passed")):
        raise AssertionError(
            "Synthetic mechanism gate failed; locked outer evaluation remains closed"
        )

    git_status = _git_value(repo_root, "status", "--porcelain")
    if bool(freeze_config.get("require_clean_git", True)) and git_status:
        raise AssertionError("Candidate freeze requires a clean, committed worktree")

    freeze_files = []
    for relative in freeze_config["outer_configs"]:
        freeze_files.append(repo_root / str(relative))
    freeze_files.extend((repo_root / "src/heartshift").rglob("*.py"))
    method_hash, file_records = _tree_hash(freeze_files, repo_root)
    lock = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_pre_outer",
        "legacy_tag": tag,
        "legacy_tag_commit": _git_value(repo_root, "rev-parse", f"{tag}^{{commit}}"),
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_status_porcelain": git_status,
        "method_tree_sha256": method_hash,
        "frozen_files": file_records,
        "source_evidence": source_evidence,
        "psmask_confirmation_gate": psmask_gate,
        "synthetic_gate": synthetic_gate,
        "synthetic_gate_path": str(synthetic_gate_path.relative_to(repo_root)),
        "synthetic_gate_sha256": sha256_file(synthetic_gate_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def verify_frozen_candidate(repo_root: Path, lock_path: Path) -> dict[str, Any]:
    """Refuse a locked evaluation if code/configs changed after candidate freeze."""
    if not lock_path.is_file():
        raise FileNotFoundError(f"Candidate lock is missing: {lock_path}")
    lock = cast(dict[str, Any], json.loads(lock_path.read_text(encoding="utf-8")))
    if lock.get("status") != "frozen_pre_outer":
        raise AssertionError("Candidate lock does not have frozen_pre_outer status")
    current_commit = _git_value(repo_root, "rev-parse", "HEAD")
    if current_commit != lock.get("git_commit"):
        raise AssertionError("Git HEAD changed after the candidate freeze")
    for record in lock.get("frozen_files", []):
        path = repo_root / str(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise AssertionError(f"Frozen method/config changed: {record['path']}")
    for evidence in lock.get("source_evidence", {}).values():
        evidence_dir = Path(str(evidence["run_dir"]))
        if not evidence_dir.is_absolute():
            evidence_dir = repo_root / evidence_dir
        prediction_name = (
            "validation_predictions.parquet"
            if evidence["phase"] == "readmission_source_only"
            else "inner_predictions.parquet"
        )
        selection_name = (
            "selected_hyperparameters.csv"
            if (evidence_dir / "selected_hyperparameters.csv").is_file()
            else "selected_configurations.csv"
        )
        evidence_files = {
            "predictions_sha256": evidence_dir / prediction_name,
            "selections_sha256": evidence_dir / selection_name,
            "manifest_sha256": evidence_dir / "run_manifest.json",
        }
        if "selection_provenance_sha256" in evidence:
            evidence_files["selection_provenance_sha256"] = (
                evidence_dir / "selection_provenance.json"
            )
        for hash_name, path in evidence_files.items():
            if not path.is_file() or sha256_file(path) != evidence[hash_name]:
                raise AssertionError(f"Frozen source evidence changed: {path}")
    expected_synthetic_hash = lock.get("synthetic_gate_sha256")
    if expected_synthetic_hash:
        gate_path = repo_root / str(lock["synthetic_gate_path"])
        if not gate_path.is_file() or sha256_file(gate_path) != expected_synthetic_hash:
            raise AssertionError("Synthetic acceptance evidence changed after freeze")
    return lock
