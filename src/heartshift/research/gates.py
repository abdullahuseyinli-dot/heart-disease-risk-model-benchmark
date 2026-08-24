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


def evaluate_synthetic_v3_gates(
    summary: pd.DataFrame,
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the mechanism-separated protocol-v3 gate without post-hoc tuning."""
    required_scenarios = {
        "label_only",
        "mar_policy_shift",
        "conditional_only",
        "mnar_outcome_only",
    }
    required_experiments = set(str(value) for value in gate_config["required_experiments"])
    adaptation_experiments = set(str(value) for value in gate_config["adaptation_experiments"])
    if not adaptation_experiments <= required_experiments:
        raise ValueError("Every v3 adaptation experiment must also be required")
    selected = summary.loc[
        summary["scenario"].isin(required_scenarios)
        & summary["experiment"].isin(required_experiments)
    ].copy()
    expected_cells = pd.MultiIndex.from_product(
        [sorted(required_scenarios), sorted(required_experiments)],
        names=["scenario", "experiment"],
    )
    observed_cells = pd.MultiIndex.from_frame(selected[["scenario", "experiment"]])
    if len(selected) != len(expected_cells) or set(observed_cells) != set(expected_cells):
        raise ValueError("Synthetic v3 summary does not contain every required mechanism cell")

    adapted = selected.loc[selected["experiment"].isin(adaptation_experiments)]
    label = adapted.loc[adapted["scenario"].eq("label_only")]
    mar = adapted.loc[adapted["scenario"].eq("mar_policy_shift")]
    conditional = adapted.loc[adapted["scenario"].eq("conditional_only")]
    mnar = adapted.loc[adapted["scenario"].eq("mnar_outcome_only")]
    mar_all = selected.loc[selected["scenario"].eq("mar_policy_shift")].set_index("experiment")
    robust_name = str(gate_config["mar_robust_experiment"])
    reference_name = str(gate_config["mar_reference_experiment"])
    if robust_name not in mar_all.index or reference_name not in mar_all.index:
        raise ValueError("The configured MAR robustness comparison is unavailable")

    checks: list[dict[str, Any]] = [
        {
            "name": "required_seeds_per_cell",
            "observed": int(selected["seeds"].min()),
            "operator": ">=",
            "threshold": int(gate_config["required_seeds_per_cell"]),
        },
        {
            "name": "label_only_diagnostic_acceptance",
            "observed": float(label["diagnostic_acceptance_rate"].min()),
            "operator": ">=",
            "threshold": float(gate_config["label_only_minimum_diagnostic_acceptance_rate"]),
        },
        {
            "name": "label_only_mlls_prevalence_error",
            "observed": float(label["mean_prevalence_absolute_error"].max()),
            "operator": "<=",
            "threshold": float(
                gate_config["label_only_maximum_mean_mlls_prevalence_absolute_error"]
            ),
        },
        {
            "name": "label_only_adapted_log_loss_delta",
            "observed": float(label["mean_adapted_minus_equal_log_loss"].max()),
            "operator": "<=",
            "threshold": float(gate_config["label_only_maximum_mean_adapted_minus_equal_log_loss"]),
        },
        {
            "name": "mar_policy_maskdro_balanced_log_loss_delta",
            "observed": float(
                mar_all.loc[robust_name, "mean_balanced_log_loss"]
                - mar_all.loc[reference_name, "mean_balanced_log_loss"]
            ),
            "operator": "<=",
            "threshold": float(
                gate_config["mar_policy_maximum_robust_minus_reference_balanced_log_loss"]
            ),
        },
        {
            "name": "mar_policy_shift_diagnostic_acceptance",
            "observed": float(mar["diagnostic_acceptance_rate"].max()),
            "operator": "<=",
            "threshold": float(gate_config["mar_policy_shift_maximum_diagnostic_acceptance_rate"]),
        },
        {
            "name": "conditional_only_diagnostic_acceptance",
            "observed": float(conditional["diagnostic_acceptance_rate"].max()),
            "operator": "<=",
            "threshold": float(gate_config["conditional_only_maximum_diagnostic_acceptance_rate"]),
        },
        {
            "name": "mnar_outcome_only_diagnostic_acceptance",
            "observed": float(mnar["diagnostic_acceptance_rate"].max()),
            "operator": "<=",
            "threshold": float(gate_config["mnar_outcome_only_maximum_diagnostic_acceptance_rate"]),
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
        "protocol": "synthetic-mechanism-v3",
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "required_experiments": sorted(required_experiments),
        "adaptation_experiments": sorted(adaptation_experiments),
    }


def _psmask_confirmation_summary(run_dir: Path) -> tuple[pd.DataFrame, int]:
    """Reconstruct the registered PS-MaskDRO estimands from prediction metrics."""
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
    minimum_seed_count = int(
        fits.groupby(["outer_target", "inner_validation", "experiment"])["seed"].nunique().min()
    )
    return summary, minimum_seed_count


def evaluate_psmask_confirmation_gate(
    run_dir: Path,
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate source-only architecture gates declared before confirmation."""
    summary, minimum_seed_count = _psmask_confirmation_summary(run_dir)
    required = set(str(value) for value in gate_config["required_experiments"])
    if not required <= set(summary.index):
        raise ValueError(
            f"PS-MaskDRO confirmation misses experiments: {sorted(required - set(summary.index))}"
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


def evaluate_psmask_pivot_selection(
    run_dir: Path,
    selection_config: dict[str, Any],
) -> dict[str, Any]:
    """Select a post-gate pivot on source data only, before any outer label is opened."""
    summary, minimum_seed_count = _psmask_confirmation_summary(run_dir)
    candidates = [str(value) for value in selection_config["candidate_experiments"]]
    missing = set(candidates) - set(summary.index)
    if missing:
        raise ValueError(f"Pivot selection misses candidates: {sorted(missing)}")
    reference_name = str(selection_config["reference_experiment"])
    if reference_name not in summary.index:
        raise ValueError(f"Pivot reference is unavailable: {reference_name}")
    reference = summary.loc[reference_name]
    eligible = summary.loc[candidates].reset_index().copy()
    eligible["natural_auroc_loss_vs_reference"] = (
        float(reference["macro_natural_roc_auc"]) - eligible["macro_natural_roc_auc"]
    )
    eligible["worst_cell_regression_vs_reference"] = eligible[
        "worst_site_policy_balanced_log_loss"
    ] - float(reference["worst_site_policy_balanced_log_loss"])
    eligible["eligible"] = eligible["natural_auroc_loss_vs_reference"].le(
        float(selection_config["maximum_natural_auroc_loss"])
    ) & eligible["worst_cell_regression_vs_reference"].le(
        float(selection_config["maximum_worst_cell_regression"])
    )
    ranked = eligible.loc[eligible["eligible"]].sort_values(
        [
            "macro_site_worst_balanced_log_loss",
            "worst_site_policy_balanced_log_loss",
            "macro_natural_roc_auc",
            "experiment",
        ],
        ascending=[True, True, False, True],
    )
    if ranked.empty:
        raise AssertionError("No robust source candidate satisfies the pivot eligibility rule")
    selected = ranked.iloc[0]
    expected = str(selection_config["expected_selected_experiment"])
    checks: list[dict[str, Any]] = [
        {
            "name": "minimum_confirmation_seeds",
            "observed": minimum_seed_count,
            "operator": ">=",
            "threshold": int(selection_config["minimum_confirmation_seeds"]),
        },
        {
            "name": "selected_natural_auroc_loss",
            "observed": float(selected["natural_auroc_loss_vs_reference"]),
            "operator": "<=",
            "threshold": float(selection_config["maximum_natural_auroc_loss"]),
        },
        {
            "name": "selected_worst_cell_regression",
            "observed": float(selected["worst_cell_regression_vs_reference"]),
            "operator": "<=",
            "threshold": float(selection_config["maximum_worst_cell_regression"]),
        },
        {
            "name": "declared_selected_experiment",
            "observed": str(selected["experiment"]),
            "operator": "==",
            "threshold": expected,
        },
    ]
    for check in checks:
        if check["operator"] == "==":
            check["finite"] = True
            check["passed"] = check["observed"] == check["threshold"]
            continue
        observed = float(check["observed"])
        threshold = float(check["threshold"])
        check["finite"] = bool(np.isfinite(observed))
        check["passed"] = bool(
            check["finite"]
            and (observed >= threshold if check["operator"] == ">=" else observed <= threshold)
        )
    return {
        "protocol_version": str(selection_config["protocol_version"]),
        "selection_mode": "post_v1_gate_source_only_pre_outer",
        "selected_experiment": str(selected["experiment"]),
        "reference_experiment": reference_name,
        "eligible_experiments": ranked["experiment"].astype(str).tolist(),
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "candidate_summary": eligible.to_dict(orient="records"),
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
    additional_artifacts = {}
    for name in (
        "config.resolved.json",
        "inner_metrics.csv",
        "fit_summaries.csv",
        "evidence_audit.json",
        "independent_validation.json",
    ):
        path = run_dir / name
        if path.is_file():
            additional_artifacts[name] = sha256_file(path)
    if additional_artifacts:
        evidence["additional_artifacts_sha256"] = additional_artifacts
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
    additional_artifacts = {}
    for name in (
        "config.resolved.json",
        "validation_metrics.csv",
        "fit_summaries.csv",
        "evidence_audit.json",
        "independent_validation.json",
    ):
        path = run_dir / name
        if path.is_file():
            additional_artifacts[name] = sha256_file(path)
    if additional_artifacts:
        evidence["additional_artifacts_sha256"] = additional_artifacts
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
    psmask_run = repo_root / str(freeze_config["source_runs"]["psmask"])
    psmask_gate: dict[str, Any] | None = None
    psmask_v1_gate: dict[str, Any] | None = None
    psmask_pivot: dict[str, Any] | None = None
    v1_gate_record_path: Path | None = None
    pivot_record_path: Path | None = None
    if "psmask_pivot_selection" in freeze_config:
        psmask_v1_gate = evaluate_psmask_confirmation_gate(
            psmask_run,
            freeze_config["psmask_v1_acceptance_gates"],
        )
        v1_gate_record_path = repo_root / str(freeze_config["psmask_v1_gate_record"])
        stored_v1_gate = json.loads(v1_gate_record_path.read_text(encoding="utf-8"))
        if bool(psmask_v1_gate["passed"]):
            raise AssertionError("Pivot protocol requires the preserved v1 gate to have failed")
        if (
            stored_v1_gate.get("passed") != psmask_v1_gate["passed"]
            or stored_v1_gate.get("checks") != psmask_v1_gate["checks"]
            or stored_v1_gate.get("summary") != psmask_v1_gate["summary"]
        ):
            raise AssertionError(
                "Stored v1 gate failure does not match deterministic recomputation"
            )
        psmask_pivot = evaluate_psmask_pivot_selection(
            psmask_run,
            freeze_config["psmask_pivot_selection"],
        )
        if not bool(psmask_pivot["passed"]):
            raise AssertionError(
                "Source-only post-gate pivot selection failed; "
                "locked outer evaluation remains closed"
            )
        pivot_record_path = repo_root / str(freeze_config["psmask_pivot_record"])
        stored_pivot = json.loads(pivot_record_path.read_text(encoding="utf-8"))
        pivot_fields = (
            "protocol_version",
            "selection_mode",
            "selected_experiment",
            "reference_experiment",
            "eligible_experiments",
            "passed",
            "checks",
            "candidate_summary",
        )
        if any(stored_pivot.get(field) != psmask_pivot[field] for field in pivot_fields):
            raise AssertionError(
                "Stored source-only pivot selection does not match deterministic recomputation"
            )
    else:
        psmask_gate = evaluate_psmask_confirmation_gate(
            psmask_run,
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
    synthetic_config_path = synthetic_run / "config.resolved.json"
    synthetic_summary_path = synthetic_run / "synthetic_summary.csv"
    if not synthetic_config_path.is_file() or not synthetic_summary_path.is_file():
        raise FileNotFoundError("Synthetic configuration or round-trip summary is missing")
    synthetic_config = json.loads(synthetic_config_path.read_text(encoding="utf-8"))
    synthetic_summary = pd.read_csv(
        synthetic_summary_path,
        float_precision="round_trip",
    )
    if str(synthetic_config.get("acceptance_gate_version", "v2")) == "v3":
        recomputed_synthetic_gate = evaluate_synthetic_v3_gates(
            synthetic_summary,
            synthetic_config["acceptance_gates"],
        )
    else:
        recomputed_synthetic_gate = evaluate_synthetic_gates(
            synthetic_summary,
            synthetic_config["acceptance_gates"],
        )
    if recomputed_synthetic_gate != synthetic_gate:
        raise AssertionError("Stored synthetic gate differs from exact round-trip recomputation")
    if not bool(synthetic_gate.get("passed")):
        raise AssertionError(
            "Synthetic mechanism gate failed; locked outer evaluation remains closed"
        )
    synthetic_audit_path = synthetic_run / "evidence_audit.json"
    synthetic_audit: dict[str, Any] | None = None
    if synthetic_audit_path.is_file():
        synthetic_audit = json.loads(synthetic_audit_path.read_text(encoding="utf-8"))
        if not bool(synthetic_audit.get("gate_passed")):
            raise AssertionError("Synthetic evidence audit does not record a passed gate")
        for name, expected_hash in synthetic_audit.get("sha256", {}).items():
            artifact = synthetic_run / str(name)
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise AssertionError(f"Synthetic evidence audit hash failed: {artifact}")

    git_status = _git_value(repo_root, "status", "--porcelain")
    if bool(freeze_config.get("require_clean_git", True)) and git_status:
        raise AssertionError("Candidate freeze requires a clean, committed worktree")

    freeze_files = []
    configured_freeze_files = (
        freeze_config["frozen_files"]
        if "frozen_files" in freeze_config
        else freeze_config["outer_configs"]
    )
    for relative in configured_freeze_files:
        freeze_files.append(repo_root / str(relative))
    freeze_files.extend((repo_root / "src/heartshift").rglob("*.py"))
    method_hash, file_records = _tree_hash(freeze_files, repo_root)
    lock: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_pre_outer",
        "legacy_tag": tag,
        "legacy_tag_commit": _git_value(repo_root, "rev-parse", f"{tag}^{{commit}}"),
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_status_porcelain": git_status,
        "method_tree_sha256": method_hash,
        "frozen_files": file_records,
        "source_evidence": source_evidence,
        "synthetic_gate": synthetic_gate,
        "synthetic_gate_path": str(synthetic_gate_path.relative_to(repo_root)),
        "synthetic_gate_sha256": sha256_file(synthetic_gate_path),
        "synthetic_config_sha256": sha256_file(synthetic_config_path),
        "synthetic_summary_sha256": sha256_file(synthetic_summary_path),
    }
    if synthetic_audit is not None:
        lock["synthetic_evidence_audit_path"] = str(synthetic_audit_path.relative_to(repo_root))
        lock["synthetic_evidence_audit_sha256"] = sha256_file(synthetic_audit_path)
    if psmask_pivot is not None:
        if psmask_v1_gate is None or v1_gate_record_path is None or pivot_record_path is None:
            raise AssertionError("Pivot freeze state is incomplete")
        lock["psmask_v1_confirmation_gate"] = psmask_v1_gate
        lock["psmask_v1_gate_record"] = str(v1_gate_record_path.relative_to(repo_root))
        lock["psmask_v1_gate_record_sha256"] = sha256_file(v1_gate_record_path)
        lock["psmask_pivot_selection"] = psmask_pivot
        lock["psmask_pivot_record"] = str(pivot_record_path.relative_to(repo_root))
        lock["psmask_pivot_record_sha256"] = sha256_file(pivot_record_path)
    else:
        lock["psmask_confirmation_gate"] = psmask_gate
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
        for name, expected_hash in evidence.get("additional_artifacts_sha256", {}).items():
            path = evidence_dir / str(name)
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise AssertionError(f"Frozen source evidence changed: {path}")
        audit_path = evidence_dir / "evidence_audit.json"
        if "evidence_audit.json" in evidence.get("additional_artifacts_sha256", {}):
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            for name, expected_hash in audit.get("sha256", {}).items():
                path = evidence_dir / str(name)
                if not path.is_file() or sha256_file(path) != expected_hash:
                    raise AssertionError(f"Frozen source audit evidence changed: {path}")
    expected_synthetic_hash = lock.get("synthetic_gate_sha256")
    if expected_synthetic_hash:
        gate_path = repo_root / str(lock["synthetic_gate_path"])
        if not gate_path.is_file() or sha256_file(gate_path) != expected_synthetic_hash:
            raise AssertionError("Synthetic acceptance evidence changed after freeze")
    synthetic_gate_path = lock.get("synthetic_gate_path")
    synthetic_run_dir = (
        (repo_root / str(synthetic_gate_path)).parent if synthetic_gate_path is not None else None
    )
    expected_synthetic_config_hash = lock.get("synthetic_config_sha256")
    if expected_synthetic_config_hash:
        if synthetic_run_dir is None:
            raise AssertionError("Synthetic configuration hash has no gate path")
        config_path = synthetic_run_dir / "config.resolved.json"
        if not config_path.is_file() or sha256_file(config_path) != expected_synthetic_config_hash:
            raise AssertionError("Synthetic resolved configuration changed after freeze")
    expected_synthetic_summary_hash = lock.get("synthetic_summary_sha256")
    if expected_synthetic_summary_hash:
        if synthetic_run_dir is None:
            raise AssertionError("Synthetic summary hash has no gate path")
        summary_path = synthetic_run_dir / "synthetic_summary.csv"
        if (
            not summary_path.is_file()
            or sha256_file(summary_path) != expected_synthetic_summary_hash
        ):
            raise AssertionError("Synthetic round-trip summary changed after freeze")
    expected_synthetic_audit_hash = lock.get("synthetic_evidence_audit_sha256")
    if expected_synthetic_audit_hash:
        audit_path = repo_root / str(lock["synthetic_evidence_audit_path"])
        if not audit_path.is_file() or sha256_file(audit_path) != expected_synthetic_audit_hash:
            raise AssertionError("Synthetic evidence audit changed after freeze")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for name, expected_hash in audit.get("sha256", {}).items():
            artifact = audit_path.parent / str(name)
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise AssertionError(f"Synthetic evidence changed after freeze: {artifact}")
    expected_v1_gate_hash = lock.get("psmask_v1_gate_record_sha256")
    if expected_v1_gate_hash:
        v1_gate_path = repo_root / str(lock["psmask_v1_gate_record"])
        if not v1_gate_path.is_file() or sha256_file(v1_gate_path) != expected_v1_gate_hash:
            raise AssertionError("Preserved PS-MaskDRO v1 gate evidence changed after freeze")
    expected_pivot_hash = lock.get("psmask_pivot_record_sha256")
    if expected_pivot_hash:
        pivot_path = repo_root / str(lock["psmask_pivot_record"])
        if not pivot_path.is_file() or sha256_file(pivot_path) != expected_pivot_hash:
            raise AssertionError("Preserved PS-MaskDRO pivot evidence changed after freeze")
    return lock
