"""Independent reconstruction and hashing of locked outer-evaluation evidence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from heartshift.config import config_hash
from heartshift.data.uci import sha256_file
from heartshift.evaluation.neural_outer import _score_predictions
from heartshift.evaluation.readmission_benchmark import _readmission_outer_metrics
from heartshift.metrics import binary_metrics

HEART_METRIC_GROUPING = [
    "outer_target",
    "model",
    "weighting",
    "calibration",
    "policy",
    "mask_replicate",
]
READMISSION_METRIC_GROUPING = [
    "experiment",
    "backend",
    "variant",
    "model",
    "weighting",
    "split",
    "policy",
    "mask_replicate",
]


def _maximum_numeric_difference(left: pd.DataFrame, right: pd.DataFrame) -> float:
    maximum = 0.0
    for column in left.columns:
        if not pd.api.types.is_numeric_dtype(left[column]):
            continue
        left_values = left[column].to_numpy(dtype=np.float64)
        right_values = right[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        if finite.any():
            difference = np.abs(left_values[finite] - right_values[finite])
            maximum = max(maximum, float(np.max(difference)))
    return maximum


def _assert_frames_match(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_by: list[str],
    check_dtype: bool = False,
    tolerance: float = 1e-12,
) -> float:
    if set(observed.columns) != set(expected.columns):
        missing = sorted(set(expected.columns) - set(observed.columns))
        extra = sorted(set(observed.columns) - set(expected.columns))
        raise AssertionError(f"Frame columns differ; missing={missing}, extra={extra}")
    observed_ordered = (
        observed.loc[:, expected.columns].sort_values(sort_by, kind="stable").reset_index(drop=True)
    )
    expected_ordered = expected.sort_values(sort_by, kind="stable").reset_index(drop=True)
    if not check_dtype:
        for column in expected_ordered.columns:
            if observed_ordered[column].isna().all() and expected_ordered[column].isna().all():
                observed_ordered[column] = np.nan
                expected_ordered[column] = np.nan
    assert_frame_equal(
        observed_ordered,
        expected_ordered,
        check_dtype=check_dtype,
        check_exact=False,
        rtol=tolerance,
        atol=tolerance,
    )
    return _maximum_numeric_difference(observed_ordered, expected_ordered)


def _metric_record(
    item: tuple[tuple[Any, ...], pd.DataFrame],
    grouping: list[str],
    score_column: str,
) -> dict[str, Any]:
    keys, group = item
    return {
        **dict(zip(grouping, keys, strict=True)),
        **binary_metrics(group["target"], group[score_column]),
    }


def _recompute_binary_metric_table(
    predictions: pd.DataFrame,
    *,
    grouping: list[str],
    score_column: str,
    workers: int,
    dropna: bool,
) -> pd.DataFrame:
    grouped = [
        (keys if isinstance(keys, tuple) else (keys,), group[["target", score_column]].copy())
        for keys, group in predictions.groupby(grouping, sort=False, dropna=dropna)
    ]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        records = list(
            executor.map(
                lambda item: _metric_record(item, grouping, score_column),
                grouped,
            )
        )
    return pd.DataFrame.from_records(records)


def _resolved_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.resolved.json"
    if not path.is_file():
        raise FileNotFoundError(f"Resolved run configuration is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Resolved run configuration must be a mapping: {path}")
    return cast(dict[str, Any], payload)


def _validate_manifest(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_config_hash = config_hash(config)
    if manifest.get("config_sha256") != expected_config_hash:
        raise AssertionError("Run manifest configuration hash does not reconstruct")
    return cast(dict[str, Any], manifest)


def _write_evidence(
    run_dir: Path,
    validation: dict[str, Any],
    *,
    kind: str,
    replace: bool,
) -> dict[str, Path]:
    validation_path = run_dir / "independent_validation.json"
    audit_path = run_dir / "evidence_audit.json"
    if not replace and (validation_path.exists() or audit_path.exists()):
        raise FileExistsError(f"Outer evidence already exists in {run_dir}")
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path != audit_path
    }
    audit = {
        "audit_status": "complete_locked_outer_evidence",
        "audit_kind": kind,
        "created_utc": datetime.now(UTC).isoformat(),
        "independent_validation_sha256": hashes["independent_validation.json"],
        "artifact_count": len(hashes),
        "sha256": hashes,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"validation": validation_path, "audit": audit_path}


def audit_heart_classical_outer(
    repo_root: Path,
    run_dir: Path,
    *,
    workers: int = 8,
    replace: bool = False,
) -> dict[str, Path]:
    """Reconstruct a classical/modern heart outer run from prediction-level evidence."""
    config = _resolved_config(run_dir)
    manifest = _validate_manifest(run_dir, config)
    predictions = pd.read_parquet(run_dir / "outer_predictions.parquet")
    seed_predictions = pd.read_parquet(run_dir / "outer_seed_predictions.parquet")
    unlabelled = pd.read_parquet(run_dir / "unlabelled_outer_predictions.parquet")
    unlabelled_seed = pd.read_parquet(run_dir / "unlabelled_outer_seed_predictions.parquet")
    stored_metrics = pd.read_csv(run_dir / "outer_metrics.csv", float_precision="round_trip")

    if "target" in unlabelled.columns or "target" in unlabelled_seed.columns:
        raise AssertionError("An endpoint leaked into an unlabelled heart prediction artifact")
    unlabelled_difference = _assert_frames_match(
        predictions.drop(columns="target"),
        unlabelled,
        sort_by=[
            "outer_target",
            "model",
            "weighting",
            "calibration",
            "policy",
            "mask_replicate",
            "sample_id",
        ],
        check_dtype=True,
        tolerance=0.0,
    )
    unlabelled_seed_difference = _assert_frames_match(
        seed_predictions.drop(columns="target"),
        unlabelled_seed,
        sort_by=[
            "outer_target",
            "model",
            "weighting",
            "training_seed",
            "calibration",
            "policy",
            "mask_replicate",
            "sample_id",
        ],
        check_dtype=True,
        tolerance=0.0,
    )

    canonical = pd.read_parquet(
        repo_root / config["data"]["canonical_path"],
        columns=["sample_id", "site", "record_sha256", "target"],
    )
    observed_labels = predictions[
        ["sample_id", "site", "record_sha256", "target"]
    ].drop_duplicates()
    expected_labels = canonical.loc[canonical["sample_id"].isin(observed_labels["sample_id"])]
    label_difference = _assert_frames_match(
        observed_labels,
        expected_labels,
        sort_by=["sample_id"],
        check_dtype=False,
        tolerance=0.0,
    )
    if not predictions["site"].eq(predictions["outer_target"]).all():
        raise AssertionError("A heart outer prediction is not from its declared held-out site")
    if (
        not np.isfinite(predictions["y_score"]).all()
        or not predictions["y_score"].between(0.0, 1.0).all()
    ):
        raise AssertionError("A heart outer score is non-finite or outside [0, 1]")

    ensemble_key = [
        column for column in unlabelled.columns if column not in {"y_score", "observed_fraction"}
    ]
    if unlabelled.duplicated(ensemble_key).any():
        raise AssertionError("Duplicate heart ensemble prediction keys detected")
    reconstructed_ensemble = (
        unlabelled_seed.groupby(ensemble_key, as_index=False, sort=False, dropna=False)
        .agg(y_score=("y_score", "mean"), observed_fraction=("observed_fraction", "mean"))
        .loc[:, unlabelled.columns]
    )
    ensemble_difference = _assert_frames_match(
        reconstructed_ensemble,
        unlabelled,
        sort_by=ensemble_key,
        tolerance=1e-14,
    )
    expected_seeds = {int(value) for value in config["seeds"]}
    if set(unlabelled_seed["training_seed"].unique()) != expected_seeds:
        raise AssertionError("Heart outer training seeds differ from the resolved configuration")

    recomputed_metrics = _recompute_binary_metric_table(
        predictions,
        grouping=HEART_METRIC_GROUPING,
        score_column="y_score",
        workers=workers,
        dropna=True,
    )
    metric_difference = _assert_frames_match(
        recomputed_metrics,
        stored_metrics,
        sort_by=HEART_METRIC_GROUPING,
    )
    expected_config_hash = config_hash(config)
    if set(predictions["config_sha256"].unique()) != {expected_config_hash}:
        raise AssertionError("Heart prediction configuration hash does not reconstruct")

    validation = {
        "status": "passed_independent_reconstruction",
        "audit_kind": "heart_classical_outer",
        "run_dir": run_dir.relative_to(repo_root).as_posix(),
        "run_git_commit": manifest.get("git_commit"),
        "endpoint_loaded_only_for_post_run_validation": True,
        "prediction_rows": len(predictions),
        "seed_prediction_rows": len(seed_predictions),
        "metric_rows": len(stored_metrics),
        "sample_count": int(predictions["sample_id"].nunique()),
        "outer_targets": sorted(predictions["outer_target"].unique().tolist()),
        "models": sorted(predictions["model"].unique().tolist()),
        "training_seeds": sorted(expected_seeds),
        "target_absent_from_unlabelled_artifacts": True,
        "canonical_label_max_abs_difference": label_difference,
        "unlabelled_labelled_max_abs_difference": unlabelled_difference,
        "unlabelled_seed_labelled_max_abs_difference": unlabelled_seed_difference,
        "ensemble_max_abs_difference": ensemble_difference,
        "metric_max_abs_difference": metric_difference,
    }
    return _write_evidence(
        run_dir,
        validation,
        kind="heart_classical_outer",
        replace=replace,
    )


def audit_readmission_outer(
    repo_root: Path,
    run_dir: Path,
    *,
    workers: int = 8,
    replace: bool = False,
) -> dict[str, Path]:
    """Reconstruct the patient-disjoint readmission outer run and every seed ensemble."""
    config = _resolved_config(run_dir)
    manifest = _validate_manifest(run_dir, config)
    predictions = pd.read_parquet(run_dir / "test_predictions.parquet")
    stored_metrics = pd.read_csv(run_dir / "test_metrics.csv", float_precision="round_trip")
    shard_index = json.loads((run_dir / "shard_index.json").read_text(encoding="utf-8"))
    ensemble_paths = [
        repo_root / Path(value) for value in shard_index["ensemble_prediction_shards"]
    ]
    seed_paths = [repo_root / Path(value) for value in shard_index["seed_prediction_shards"]]
    if len(ensemble_paths) != len(seed_paths) or not ensemble_paths:
        raise AssertionError("Readmission shard index is empty or unpaired")

    maximum_shard_difference = 0.0
    maximum_ensemble_difference = 0.0
    expected_seeds = {int(value) for value in config["seeds"]}
    for ensemble_path, seed_path in zip(ensemble_paths, seed_paths, strict=True):
        ensemble = pd.read_parquet(ensemble_path)
        seeds = pd.read_parquet(seed_path)
        if "target" in ensemble.columns or "target" in seeds.columns:
            raise AssertionError("An endpoint leaked into an unlabelled readmission shard")
        experiments = ensemble["experiment"].unique()
        if len(experiments) != 1:
            raise AssertionError(f"Readmission shard has multiple experiments: {ensemble_path}")
        experiment = str(experiments[0])
        labelled_group = predictions.loc[predictions["experiment"].eq(experiment)].drop(
            columns="target"
        )
        maximum_shard_difference = max(
            maximum_shard_difference,
            _assert_frames_match(
                labelled_group,
                ensemble,
                sort_by=["sample_id", "policy", "mask_replicate"],
                check_dtype=False,
                tolerance=0.0,
            ),
        )
        if set(seeds["seed"].unique()) != expected_seeds:
            raise AssertionError(f"Readmission seeds differ from configuration: {seed_path}")
        ensemble_key = [
            column
            for column in ensemble.columns
            if column not in {"evidence_logit", "y_score", "observed_fraction"}
        ]
        reconstructed = (
            seeds.groupby(ensemble_key, as_index=False, sort=False, dropna=False)
            .agg(
                evidence_logit=("evidence_logit", "mean"),
                y_score=("y_score", "mean"),
                observed_fraction=("observed_fraction", "mean"),
            )
            .loc[:, ensemble.columns]
        )
        maximum_ensemble_difference = max(
            maximum_ensemble_difference,
            _assert_frames_match(
                reconstructed,
                ensemble,
                sort_by=ensemble_key,
                check_dtype=False,
                tolerance=1e-14,
            ),
        )

    target_column = str(config["data"]["target"])
    canonical = pd.read_parquet(
        repo_root / config["data"]["canonical_path"],
        columns=["sample_id", "patient_nbr", "source_line_sha256", target_column],
    ).rename(columns={target_column: "target"})
    split = pd.read_parquet(repo_root / config["data"]["split_path"])
    expected = canonical.merge(
        split[["sample_id", "split"]], on="sample_id", how="inner", validate="one_to_one"
    )
    expected = expected.loc[expected["split"].isin(config["data"]["test_splits"])]
    observed = predictions[
        ["sample_id", "patient_nbr", "source_line_sha256", "target", "split"]
    ].drop_duplicates()
    label_difference = _assert_frames_match(
        observed,
        expected.loc[:, observed.columns],
        sort_by=["sample_id"],
        check_dtype=False,
        tolerance=0.0,
    )
    patient_split = canonical[["sample_id", "patient_nbr"]].merge(
        split[["sample_id", "split"]], on="sample_id", how="inner", validate="one_to_one"
    )
    refit_patients = set(
        patient_split.loc[
            patient_split["split"].isin(config["data"]["refit_splits"]), "patient_nbr"
        ]
    )
    test_patients = set(
        patient_split.loc[patient_split["split"].isin(config["data"]["test_splits"]), "patient_nbr"]
    )
    patient_overlap = refit_patients & test_patients
    if patient_overlap:
        raise AssertionError("Readmission refit and test patients overlap")
    if (
        not np.isfinite(predictions["y_score"]).all()
        or not predictions["y_score"].between(0.0, 1.0).all()
    ):
        raise AssertionError("A readmission score is non-finite or outside [0, 1]")
    prediction_key = ["experiment", "sample_id", "policy", "mask_replicate"]
    if predictions.duplicated(prediction_key).any():
        raise AssertionError("Duplicate readmission prediction keys detected")

    recomputed_metrics = _readmission_outer_metrics(predictions)
    metric_difference = _assert_frames_match(
        recomputed_metrics,
        stored_metrics,
        sort_by=READMISSION_METRIC_GROUPING,
    )
    threaded_metrics = _recompute_binary_metric_table(
        predictions,
        grouping=READMISSION_METRIC_GROUPING,
        score_column="y_score",
        workers=workers,
        dropna=False,
    )
    threaded_metric_difference = _assert_frames_match(
        threaded_metrics,
        stored_metrics,
        sort_by=READMISSION_METRIC_GROUPING,
    )
    expected_config_hash = config_hash(config)
    if set(predictions["config_sha256"].unique()) != {expected_config_hash}:
        raise AssertionError("Readmission prediction configuration hash does not reconstruct")

    validation = {
        "status": "passed_independent_reconstruction",
        "audit_kind": "readmission_outer",
        "run_dir": run_dir.relative_to(repo_root).as_posix(),
        "run_git_commit": manifest.get("git_commit"),
        "endpoint_loaded_only_for_post_run_validation": True,
        "prediction_rows": len(predictions),
        "metric_rows": len(stored_metrics),
        "sample_count": int(predictions["sample_id"].nunique()),
        "patient_count": int(predictions["patient_nbr"].nunique()),
        "experiments": sorted(predictions["experiment"].unique().tolist()),
        "training_seeds": sorted(expected_seeds),
        "target_absent_from_unlabelled_artifacts": True,
        "refit_test_patient_overlap": len(patient_overlap),
        "canonical_label_max_abs_difference": label_difference,
        "shard_top_level_max_abs_difference": maximum_shard_difference,
        "ensemble_max_abs_difference": maximum_ensemble_difference,
        "metric_max_abs_difference": metric_difference,
        "threaded_metric_max_abs_difference": threaded_metric_difference,
        "known_nonsemantic_dtype_coercion": "epochs null/object shards to float64 concatenation",
    }
    return _write_evidence(run_dir, validation, kind="readmission_outer", replace=replace)


def audit_heart_neural_outer(
    repo_root: Path,
    run_dir: Path,
    *,
    replace: bool = False,
) -> dict[str, Path]:
    """Validate neural outer shards, gating semantics, labels, ensembles, and metrics."""
    config = _resolved_config(run_dir)
    manifest = _validate_manifest(run_dir, config)
    predictions = pd.read_parquet(run_dir / "outer_predictions.parquet")
    seed_predictions = pd.read_parquet(run_dir / "outer_seed_predictions.parquet")
    unlabelled = pd.concat(
        (
            pd.read_parquet(path)
            for path in sorted((run_dir / "shards").glob("*/unlabelled_outer_predictions.parquet"))
        ),
        ignore_index=True,
    )
    unlabelled_seed = pd.concat(
        (
            pd.read_parquet(path)
            for path in sorted(
                (run_dir / "shards").glob("*/unlabelled_outer_seed_predictions.parquet")
            )
        ),
        ignore_index=True,
    )
    if "target" in unlabelled.columns or "target" in unlabelled_seed.columns:
        raise AssertionError("An endpoint leaked into an unlabelled neural prediction artifact")
    prediction_sort = [
        "outer_target",
        "experiment",
        "policy",
        "mask_replicate",
        "sample_id",
    ]
    unlabelled_difference = _assert_frames_match(
        predictions.drop(columns="target"),
        unlabelled,
        sort_by=prediction_sort,
        check_dtype=False,
        tolerance=0.0,
    )
    unlabelled_seed_difference = _assert_frames_match(
        seed_predictions.drop(columns="target"),
        unlabelled_seed,
        sort_by=[*prediction_sort[:-1], "training_seed", "sample_id"],
        check_dtype=False,
        tolerance=0.0,
    )

    canonical = pd.read_parquet(
        repo_root / config["data"]["canonical_path"],
        columns=["sample_id", "site", "record_sha256", "target"],
    )
    observed_labels = predictions[
        ["sample_id", "site", "record_sha256", "target"]
    ].drop_duplicates()
    expected_labels = canonical.loc[canonical["sample_id"].isin(observed_labels["sample_id"])]
    label_difference = _assert_frames_match(
        observed_labels,
        expected_labels,
        sort_by=["sample_id"],
        check_dtype=False,
        tolerance=0.0,
    )
    if not predictions["site"].eq(predictions["outer_target"]).all():
        raise AssertionError("A neural prediction is not from its declared held-out site")
    if (
        not np.isfinite(predictions["y_score_zero_shot"]).all()
        or not predictions["y_score_zero_shot"].between(0.0, 1.0).all()
    ):
        raise AssertionError("A neural zero-shot score is non-finite or outside [0, 1]")

    adaptable_variants = {str(value) for value in config["adaptable_variants"]}
    adaptable = predictions["variant"].isin(adaptable_variants)
    research_columns = ["y_score_uda_mlls_research", "y_score_uda_soft_bbse_research"]
    deployment_columns = ["y_score_uda_mlls", "y_score_uda_soft_bbse"]
    if predictions.loc[adaptable, research_columns].isna().any().any():
        raise AssertionError("An adaptable neural prediction lacks a research adaptation score")
    if predictions.loc[~adaptable, [*research_columns, *deployment_columns]].notna().any().any():
        raise AssertionError("A non-adaptable neural control contains adaptation scores")
    if not predictions.loc[~adaptable, "adaptation_status"].eq("not_applicable").all():
        raise AssertionError("A non-adaptable neural control has an invalid adaptation status")
    accepted = predictions["adaptation_allowed"].astype(bool)
    if predictions.loc[accepted, deployment_columns].isna().any().any():
        raise AssertionError("An accepted neural cell lacks deployment adaptation scores")
    if predictions.loc[~accepted, deployment_columns].notna().any().any():
        raise AssertionError("A rejected/non-applicable neural cell exposes deployment scores")
    if (
        not predictions.loc[adaptable, "adaptation_status"]
        .isin(["accepted", "diagnostic_rejected"])
        .all()
    ):
        raise AssertionError("An adaptable neural cell has an invalid adaptation status")

    ensemble_grouping = [
        "sample_id",
        "site",
        "record_sha256",
        "policy",
        "mask_replicate",
        "outer_target",
        "experiment",
        "variant",
        "epochs",
        "config_sha256",
    ]
    reconstructed = seed_predictions.groupby(
        ensemble_grouping, as_index=False, sort=False, dropna=False
    ).agg(
        evidence_logit=("evidence_logit", "mean"),
        y_score_zero_shot=("y_score", "mean"),
        observed_fraction=("observed_fraction", "mean"),
    )
    ensemble_expected = predictions.loc[
        :,
        [*ensemble_grouping, "evidence_logit", "y_score_zero_shot", "observed_fraction"],
    ]
    ensemble_difference = _assert_frames_match(
        reconstructed,
        ensemble_expected,
        sort_by=ensemble_grouping,
        check_dtype=False,
        tolerance=1e-14,
    )
    expected_seeds = {int(value) for value in config["seeds"]}
    if set(seed_predictions["training_seed"].unique()) != expected_seeds:
        raise AssertionError("Neural outer training seeds differ from configuration")

    calibration = pd.read_parquet(run_dir / "source_calibration_predictions.parquet")
    if adaptable_variants:
        adaptable_calibration = calibration.loc[calibration["variant"].isin(adaptable_variants)]
        calibration_key = [
            "outer_target",
            "experiment",
            "evaluation_policy",
            "policy_replicate",
            "sample_id",
        ]
        if adaptable_calibration.empty or adaptable_calibration.duplicated(calibration_key).any():
            raise AssertionError("Adaptable source calibration is empty or not patient-unique")
        if adaptable_calibration["site"].eq(adaptable_calibration["outer_target"]).any():
            raise AssertionError("A held-out target patient appears in source calibration")
    diagnostics_path = run_dir / "adaptation_diagnostics.csv"
    summary_path = run_dir / "adaptation_summary.csv"
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.is_file() else pd.DataFrame()
    adaptation_summary = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    if adaptable.any() and (diagnostics.empty or adaptation_summary.empty):
        raise AssertionError("Adaptable neural variants lack diagnostic evidence")

    recomputed_metrics = _score_predictions(predictions)
    stored_metrics = pd.read_csv(run_dir / "outer_metrics.csv", float_precision="round_trip")
    neural_metric_grouping = [
        "outer_target",
        "experiment",
        "variant",
        "policy",
        "mask_replicate",
        "track",
    ]
    metric_difference = _assert_frames_match(
        recomputed_metrics,
        stored_metrics,
        sort_by=neural_metric_grouping,
    )
    if "source_failed_run" in config:
        source_config = _resolved_config(repo_root / str(config["source_failed_run"]))
        expected_prediction_config_hash = config_hash(source_config)
    else:
        expected_prediction_config_hash = config_hash(config)
    if set(predictions["config_sha256"].unique()) != {expected_prediction_config_hash}:
        raise AssertionError("Neural prediction configuration hash does not reconstruct")

    validation = {
        "status": "passed_independent_reconstruction",
        "audit_kind": "heart_neural_outer",
        "run_dir": run_dir.relative_to(repo_root).as_posix(),
        "run_git_commit": manifest.get("git_commit"),
        "endpoint_loaded_only_for_post_run_validation": True,
        "prediction_rows": len(predictions),
        "seed_prediction_rows": len(seed_predictions),
        "calibration_prediction_rows": len(calibration),
        "metric_rows": len(stored_metrics),
        "sample_count": int(predictions["sample_id"].nunique()),
        "outer_targets": sorted(predictions["outer_target"].unique().tolist()),
        "experiments": sorted(predictions["experiment"].unique().tolist()),
        "training_seeds": sorted(expected_seeds),
        "adaptable_variants": sorted(adaptable_variants),
        "no_refit_finalization": "source_failed_run" in config,
        "prediction_config_sha256": expected_prediction_config_hash,
        "accepted_adaptation_cells": int(
            predictions.loc[accepted, ["outer_target", "experiment", "policy", "mask_replicate"]]
            .drop_duplicates()
            .shape[0]
        ),
        "target_absent_from_unlabelled_artifacts": True,
        "canonical_label_max_abs_difference": label_difference,
        "unlabelled_labelled_max_abs_difference": unlabelled_difference,
        "unlabelled_seed_labelled_max_abs_difference": unlabelled_seed_difference,
        "ensemble_max_abs_difference": ensemble_difference,
        "metric_max_abs_difference": metric_difference,
    }
    return _write_evidence(run_dir, validation, kind="heart_neural_outer", replace=replace)
