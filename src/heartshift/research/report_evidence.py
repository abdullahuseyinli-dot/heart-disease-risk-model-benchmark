"""Independent reconstruction and hashing of prediction-derived heart reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from heartshift.data.uci import sha256_file
from heartshift.reporting.outer_report import (
    adaptation_summary,
    cell_metrics,
    descriptive_sex_subgroup_metrics,
    load_heart_outer_predictions,
    paired_primary_bootstrap,
    primary_estimands,
)
from heartshift.research.outer_evidence import _assert_frames_match


def audit_heart_outer_report(
    repo_root: Path,
    config: dict[str, Any],
    report_dir: Path,
    *,
    replace: bool = False,
) -> dict[str, Path]:
    """Rebuild every report table, including all registered bootstrap replicates."""
    validation_path = report_dir / "independent_validation.json"
    audit_path = report_dir / "evidence_audit.json"
    if not replace and (validation_path.exists() or audit_path.exists()):
        raise FileExistsError(f"Report evidence already exists in {report_dir}")

    expected_predictions = load_heart_outer_predictions(repo_root, config["prediction_runs"])
    demographics = pd.read_parquet(
        repo_root / config["canonical_data_path"], columns=["sample_id", "sex"]
    )
    expected_predictions = expected_predictions.merge(
        demographics, on="sample_id", how="left", validate="many_to_one"
    )
    stored_predictions = pd.read_parquet(report_dir / "normalized_outer_predictions.parquet")
    prediction_key = [
        "method",
        "track",
        "outer_target",
        "policy",
        "mask_replicate",
        "sample_id",
    ]
    prediction_difference = _assert_frames_match(
        stored_predictions,
        expected_predictions,
        sort_by=prediction_key,
        check_dtype=False,
        tolerance=0.0,
    )

    expected_metrics = cell_metrics(expected_predictions)
    stored_metrics = pd.read_csv(
        report_dir / "site_policy_metrics.csv", float_precision="round_trip"
    )
    metric_key = ["method", "track", "outer_target", "policy", "mask_replicate"]
    metric_difference = _assert_frames_match(
        stored_metrics,
        expected_metrics,
        sort_by=metric_key,
    )
    expected_primary = primary_estimands(expected_metrics)
    stored_primary = pd.read_csv(
        report_dir / "primary_estimands.csv", float_precision="round_trip"
    )
    primary_difference = _assert_frames_match(
        stored_primary,
        expected_primary,
        sort_by=["method"],
    )
    expected_adaptation = adaptation_summary(expected_metrics)
    stored_adaptation = pd.read_csv(
        report_dir / "adaptation_summary.csv", float_precision="round_trip"
    )
    adaptation_difference = _assert_frames_match(
        stored_adaptation,
        expected_adaptation,
        sort_by=["method", "track"],
    )
    expected_subgroups = descriptive_sex_subgroup_metrics(expected_predictions)
    stored_subgroups = pd.read_csv(
        report_dir / "descriptive_sex_subgroup_metrics.csv",
        float_precision="round_trip",
    )
    subgroup_difference = _assert_frames_match(
        stored_subgroups,
        expected_subgroups,
        sort_by=["method", "outer_target", "sex"],
    )

    expected_replicates, expected_intervals = paired_primary_bootstrap(
        expected_predictions,
        reference_method=str(config["bootstrap"]["reference_method"]),
        comparison_methods=[
            str(value) for value in config["bootstrap"]["comparison_methods"]
        ],
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    stored_replicates = pd.read_parquet(report_dir / "paired_bootstrap_replicates.parquet")
    bootstrap_difference = _assert_frames_match(
        stored_replicates,
        expected_replicates,
        sort_by=["bootstrap_replicate", "method"],
        check_dtype=False,
        tolerance=0.0,
    )
    stored_intervals = pd.read_csv(
        report_dir / "paired_bootstrap_intervals.csv", float_precision="round_trip"
    )
    interval_difference = _assert_frames_match(
        stored_intervals,
        expected_intervals,
        sort_by=["method", "reference_method"],
    )

    manifest = json.loads((report_dir / "report_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "derived_from_locked_sample_level_predictions":
        raise AssertionError("Heart report manifest does not identify locked predictions")
    if int(manifest.get("prediction_rows", -1)) != len(stored_predictions):
        raise AssertionError("Heart report manifest prediction count does not reconstruct")
    if manifest.get("bootstrap_unit") != "patient_within_observed_outer_hospital":
        raise AssertionError("Heart report bootstrap unit is incorrect")
    if manifest.get("hospital_inference") != "conditional_on_four_observed_hospitals":
        raise AssertionError("Heart report overstates hospital-level inference")

    validation = {
        "status": "passed_independent_report_reconstruction",
        "prediction_rows": len(stored_predictions),
        "metric_rows": len(stored_metrics),
        "primary_estimand_rows": len(stored_primary),
        "adaptation_rows": len(stored_adaptation),
        "sex_subgroup_rows": len(stored_subgroups),
        "bootstrap_rows": len(stored_replicates),
        "bootstrap_interval_rows": len(stored_intervals),
        "bootstrap_repetitions": int(config["bootstrap"]["repetitions"]),
        "comparison_count": len(config["bootstrap"]["comparison_methods"]),
        "bootstrap_unit": manifest["bootstrap_unit"],
        "hospital_inference": manifest["hospital_inference"],
        "normalized_prediction_max_abs_difference": prediction_difference,
        "metric_max_abs_difference": metric_difference,
        "primary_estimand_max_abs_difference": primary_difference,
        "adaptation_max_abs_difference": adaptation_difference,
        "sex_subgroup_max_abs_difference": subgroup_difference,
        "bootstrap_replicate_max_abs_difference": bootstrap_difference,
        "bootstrap_interval_max_abs_difference": interval_difference,
    }
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hashes = {
        path.relative_to(report_dir).as_posix(): sha256_file(path)
        for path in sorted(report_dir.rglob("*"))
        if path.is_file() and path != audit_path
    }
    audit = {
        "audit_status": "complete_locked_report_evidence",
        "audit_kind": "heart_outer_report",
        "created_utc": datetime.now(UTC).isoformat(),
        "independent_validation_sha256": hashes["independent_validation.json"],
        "artifact_count": len(hashes),
        "sha256": hashes,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"validation": validation_path, "audit": audit_path}
