"""Controlled success/failure experiments for PS-MaskDRO and prior adaptation."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from heartshift.adaptation import (
    acquisition_aware_label_shift_diagnostic,
    estimate_target_prevalence_mlls,
    estimate_target_prevalence_soft_bbse,
    fit_evidence_calibrator,
    mixture_fit_diagnostic,
    posterior_from_evidence,
)
from heartshift.config import config_hash
from heartshift.data.synthetic import generate_synthetic_environments
from heartshift.data.uci import FEATURE_COLUMNS, sha256_file
from heartshift.evaluation.classical_benchmark import write_run_manifest
from heartshift.masks import MaskPolicy, policy_seed
from heartshift.metrics import binary_metrics
from heartshift.models.ps_maskdro import (
    CORE_PREDICTION_COLUMNS,
    MASK_PREDICTION_COLUMNS,
    fit_ps_maskdro,
    predict_policy_bank,
)
from heartshift.research.gates import evaluate_synthetic_gates, evaluate_synthetic_v3_gates


def _write_synthetic_evidence_audit(
    *,
    run_dir: Path,
    artifact_paths: dict[str, Path],
    config: dict[str, Any],
    predictions: pd.DataFrame,
    source_predictions: pd.DataFrame | None,
    results: pd.DataFrame,
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    history: pd.DataFrame,
    gate: dict[str, Any],
) -> Path:
    """Validate and hash a completed synthetic run before returning it."""
    cell_columns = ["scenario", "experiment", "seed"]
    sample_key = [*cell_columns, "sample_id"]
    expected_cells = len(config["scenarios"]) * len(config["experiments"]) * len(config["seeds"])
    if len(results) != expected_cells or results.duplicated(cell_columns).any():
        raise AssertionError("Synthetic result cells are incomplete or duplicated")
    if predictions.duplicated(sample_key).any():
        raise AssertionError("Synthetic target predictions duplicate a patient within a cell")
    if predictions.groupby(cell_columns)["sample_id"].nunique().nunique() != 1:
        raise AssertionError("Synthetic target cells have inconsistent patient coverage")
    if int(predictions.groupby(cell_columns)["sample_id"].nunique().iloc[0]) != int(
        config["n_per_environment"]
    ):
        raise AssertionError("Synthetic target cells do not cover the configured target size")
    probability_columns = [
        "equal_prior_probability",
        "adapted_probability",
        "soft_bbse_probability",
        "oracle_probability",
    ]
    if not np.isfinite(predictions.loc[:, probability_columns].to_numpy()).all():
        raise AssertionError("Synthetic research probabilities are not finite")
    probabilities = predictions.loc[:, probability_columns].to_numpy()
    if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
        raise AssertionError("Synthetic research probabilities fall outside [0, 1]")
    allowed = predictions["automatic_adaptation_allowed"].astype(bool).to_numpy()
    deployed = predictions["deployment_probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(deployed[allowed]).all() or not np.isnan(deployed[~allowed]).all():
        raise AssertionError("Automatic deployment probabilities violate abstention decisions")
    if set(predictions["target"].unique()) - {0, 1}:
        raise AssertionError("Synthetic target outcomes are not binary")
    if not np.isfinite(diagnostics["p_value"].to_numpy(dtype=np.float64)).all():
        raise AssertionError("Synthetic diagnostic p-values are not finite")
    if not predictions["config_sha256"].eq(config_hash(config)).all():
        raise AssertionError("Synthetic target prediction configuration hashes differ")

    source_rows = 0
    if source_predictions is not None:
        source_rows = len(source_predictions)
        if source_predictions.duplicated(sample_key).any():
            raise AssertionError("Synthetic source predictions duplicate a patient within a cell")
        if not source_predictions["config_sha256"].eq(config_hash(config)).all():
            raise AssertionError("Synthetic source prediction configuration hashes differ")
        if source_predictions.groupby(cell_columns)["sample_id"].nunique().nunique() != 1:
            raise AssertionError("Synthetic source cells have inconsistent patient coverage")

    stored_gate = json.loads(artifact_paths["gate"].read_text(encoding="utf-8"))
    if stored_gate != gate:
        raise AssertionError("Persisted synthetic gate differs from exact recomputation")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    hash_paths = {
        path.name: path
        for path in run_dir.iterdir()
        if path.is_file() and path.name != "evidence_audit.json"
    }
    failed_checks = [
        {
            "name": check["name"],
            "observed": check["observed"],
            "operator": check["operator"],
            "threshold": check["threshold"],
        }
        for check in gate["checks"]
        if not check["passed"]
    ]
    protocol_version = str(config.get("protocol_version", ""))
    if "dryrun" in protocol_version:
        audit_status = "code_dryrun_complete_noninferential"
    else:
        audit_status = (
            "complete_passed_gate" if gate["passed"] else "complete_failed_gate_preserved"
        )
    audit = {
        "audit_status": audit_status,
        "config_sha256": config_hash(config),
        "gate_recomputed_exactly_from_csv_with_float_precision_round_trip": True,
        "git_commit": manifest["git_commit"],
        "prediction_rows": len(predictions),
        "source_prediction_rows": source_rows,
        "result_rows": len(results),
        "diagnostic_rows": len(diagnostics),
        "summary_rows": len(summary),
        "training_history_rows": len(history),
        "unique_experiment_scenario_seed_cells": int(
            results[cell_columns].drop_duplicates().shape[0]
        ),
        "seeds": sorted(int(value) for value in results["seed"].unique()),
        "gate_passed": bool(gate["passed"]),
        "failed_checks": failed_checks,
        "sha256": {name: sha256_file(path) for name, path in sorted(hash_paths.items())},
    }
    audit_path = run_dir / "evidence_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit_path


def run_synthetic_experiment(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "synthetic_mechanism_validation")
    device = torch.device(str(config.get("device", "cuda")))
    predictions_records: list[pd.DataFrame] = []
    source_prediction_records: list[pd.DataFrame] = []
    result_records: list[dict[str, Any]] = []
    diagnostic_records: list[pd.DataFrame] = []
    history_records: list[dict[str, Any]] = []
    diagnostic_mode = str(config.get("diagnostic", {}).get("mode", "energy_v2"))
    if diagnostic_mode not in {"energy_v2", "composite_multiview_v3"}:
        raise ValueError(f"Unknown synthetic diagnostic mode: {diagnostic_mode}")

    for scenario in config["scenarios"]:
        for seed in config["seeds"]:
            prevalence_values = tuple(float(value) for value in config["prevalences"])
            if len(prevalence_values) != 4:
                raise ValueError("Synthetic experiment requires four prevalences")
            data = generate_synthetic_environments(
                str(scenario),
                n_per_environment=int(config["n_per_environment"]),
                seed=int(seed),
                prevalences=cast(tuple[float, float, float, float], prevalence_values),
            )
            training = data.loc[data["site"].isin(["synthetic_e0", "synthetic_e1"])].copy()
            validation = data.loc[data["site"].eq("synthetic_e2")].copy()
            target = data.loc[data["site"].eq("synthetic_e3")].copy()
            training_mask_pool = training.loc[:, FEATURE_COLUMNS].notna().to_numpy()
            target_mask_pool = target.loc[:, FEATURE_COLUMNS].notna().to_numpy()

            for experiment in config["experiments"]:
                experiment_name = str(experiment["name"])
                variant = str(experiment["variant"])
                parameters = dict(config["common_parameters"]) | dict(
                    experiment.get("parameters", {})
                )
                fit_seed = policy_seed(int(seed), f"{scenario}|{experiment_name}", 0)
                result = fit_ps_maskdro(
                    training,
                    validation,
                    variant=variant,
                    parameters=parameters,
                    seed=fit_seed,
                    device=str(device),
                )
                natural_policy = (MaskPolicy("natural", "natural"),)
                target_predictions = predict_policy_bank(
                    result.model,
                    result.preprocessor,
                    target,
                    natural_policy,
                    device=device,
                    base_seed=fit_seed,
                    replicates=1,
                    empirical_mask_pool=training_mask_pool,
                )

                if diagnostic_mode == "composite_multiview_v3":
                    # One natural-policy prediction per held-out source patient.
                    # The v2 target-mask intersection path remains available only
                    # for reproducing its immutable failed experiment.
                    source_predictions = predict_policy_bank(
                        result.model,
                        result.preprocessor,
                        validation,
                        natural_policy,
                        device=device,
                        base_seed=fit_seed + 1,
                        replicates=1,
                        empirical_mask_pool=training_mask_pool,
                    )
                    if not source_predictions["sample_id"].is_unique:
                        raise AssertionError(
                            "Protocol-v3 diagnostics require unique source patients"
                        )
                else:
                    # Historical v2 UDA policy matching uses target masks but never
                    # target labels.  It is intentionally preserved for reproduction.
                    source_predictions = predict_policy_bank(
                        result.model,
                        result.preprocessor,
                        validation,
                        (MaskPolicy("target_policy_matched", "empirical"),),
                        device=device,
                        base_seed=fit_seed + 1,
                        replicates=int(config["target_policy_match_replicates"]),
                        empirical_mask_pool=target_mask_pool,
                    )
                calibrator = fit_evidence_calibrator(
                    source_predictions["evidence_logit"],
                    source_predictions["target"],
                    source_predictions["site"],
                )
                source_evidence = calibrator.transform(source_predictions["evidence_logit"])
                target_evidence = calibrator.transform(target_predictions["evidence_logit"])
                prevalence_estimate = estimate_target_prevalence_mlls(target_evidence)
                bbse_prevalence_estimate = estimate_target_prevalence_soft_bbse(
                    posterior_from_evidence(source_evidence, 0.5),
                    source_predictions["target"],
                    posterior_from_evidence(target_evidence, 0.5),
                )
                equal_prior_probability = posterior_from_evidence(target_evidence, 0.5)
                adapted_probability = posterior_from_evidence(target_evidence, prevalence_estimate)
                bbse_probability = posterior_from_evidence(
                    target_evidence, bbse_prevalence_estimate
                )
                if diagnostic_mode == "composite_multiview_v3":
                    diagnostic = acquisition_aware_label_shift_diagnostic(
                        source_evidence,
                        source_predictions["target"],
                        source_predictions.loc[:, CORE_PREDICTION_COLUMNS],
                        source_predictions.loc[:, MASK_PREDICTION_COLUMNS],
                        target_evidence,
                        target_predictions.loc[:, CORE_PREDICTION_COLUMNS],
                        target_predictions.loc[:, MASK_PREDICTION_COLUMNS],
                        source_sample_ids=source_predictions["sample_id"],
                        target_sample_ids=target_predictions["sample_id"],
                        prior_grid=config["diagnostic"]["prior_grid"],
                        bootstrap_repetitions=int(config["diagnostic"]["bootstrap_repetitions"]),
                        rff_features_per_view=int(config["diagnostic"]["rff_features_per_view"]),
                        alpha=float(config["diagnostic"]["alpha"]),
                        maximum_nearest_hamming_fraction=float(
                            config["diagnostic"]["maximum_nearest_hamming_fraction"]
                        ),
                        seed=fit_seed + 2,
                    )
                    automatic_allowed = diagnostic.accepted
                    accepted_prior_lower = np.nan
                    accepted_prior_upper = np.nan
                    diagnostic_best_prior = diagnostic.best_prior
                    diagnostic_statistic = diagnostic.best_statistic
                    diagnostic_p_value = diagnostic.p_value
                    mask_support_passed = diagnostic.mask_support.passed
                    mask_all_feature_states_supported = (
                        diagnostic.mask_support.all_feature_states_supported
                    )
                    mask_exact_support_rate = diagnostic.mask_support.exact_pattern_support_rate
                    mask_nearest_hamming_q95 = diagnostic.mask_support.nearest_hamming_fraction_q95
                    mask_nearest_hamming_max = diagnostic.mask_support.nearest_hamming_fraction_max
                    diagnostic_table = diagnostic.table.copy()

                    source_frame = source_predictions.copy()
                    source_frame["scenario"] = scenario
                    source_frame["experiment"] = experiment_name
                    source_frame["variant"] = variant
                    source_frame["seed"] = int(seed)
                    source_frame["fit_seed"] = fit_seed
                    source_frame["evidence_logit_calibrated"] = source_evidence
                    source_frame["config_sha256"] = config_hash(config)
                    source_prediction_records.append(source_frame)
                else:
                    legacy_diagnostic = mixture_fit_diagnostic(
                        source_evidence[source_predictions["target"].to_numpy() == 0],
                        source_evidence[source_predictions["target"].to_numpy() == 1],
                        target_evidence,
                        prior_grid=config["diagnostic"]["prior_grid"],
                        bootstrap_repetitions=int(config["diagnostic"]["bootstrap_repetitions"]),
                        mixture_draws=int(config["diagnostic"]["mixture_draws"]),
                        alpha=float(config["diagnostic"]["alpha"]),
                        seed=fit_seed + 2,
                    )
                    automatic_allowed = legacy_diagnostic.accepted_interval is not None
                    accepted_prior_lower = (
                        legacy_diagnostic.accepted_interval[0]
                        if legacy_diagnostic.accepted_interval is not None
                        else np.nan
                    )
                    accepted_prior_upper = (
                        legacy_diagnostic.accepted_interval[1]
                        if legacy_diagnostic.accepted_interval is not None
                        else np.nan
                    )
                    diagnostic_best_prior = legacy_diagnostic.best_prior
                    diagnostic_statistic = legacy_diagnostic.best_statistic
                    best_row = legacy_diagnostic.table.sort_values(
                        ["energy_statistic", "prevalence"]
                    ).iloc[0]
                    diagnostic_p_value = float(best_row["p_value"])
                    mask_support_passed = True
                    mask_all_feature_states_supported = True
                    mask_exact_support_rate = np.nan
                    mask_nearest_hamming_q95 = np.nan
                    mask_nearest_hamming_max = np.nan
                    diagnostic_table = legacy_diagnostic.table.copy()

                # Synthetic target labels are touched only after every unlabelled-
                # target adaptation and automatic-deployment decision is finalized.
                true_prevalence = float(target["target"].mean())
                oracle_probability = posterior_from_evidence(target_evidence, true_prevalence)

                prediction_frame = target_predictions.loc[
                    :, ["sample_id", "site", "target", "record_sha256"]
                ].copy()
                if diagnostic_mode == "composite_multiview_v3":
                    diagnostic_input_columns = [
                        "evidence_logit",
                        "observed_fraction",
                        "observed_mask_code",
                        *CORE_PREDICTION_COLUMNS,
                        *MASK_PREDICTION_COLUMNS,
                    ]
                    prediction_frame.loc[:, diagnostic_input_columns] = target_predictions.loc[
                        :, diagnostic_input_columns
                    ]
                prediction_frame["scenario"] = scenario
                prediction_frame["experiment"] = experiment_name
                prediction_frame["variant"] = variant
                prediction_frame["seed"] = int(seed)
                prediction_frame["evidence_logit_calibrated"] = target_evidence
                prediction_frame["equal_prior_probability"] = equal_prior_probability
                prediction_frame["adapted_probability"] = adapted_probability
                prediction_frame["soft_bbse_probability"] = bbse_probability
                prediction_frame["oracle_probability"] = oracle_probability
                prediction_frame["automatic_adaptation_allowed"] = automatic_allowed
                prediction_frame["deployment_probability"] = (
                    adapted_probability if automatic_allowed else np.nan
                )
                prediction_frame["config_sha256"] = config_hash(config)
                predictions_records.append(prediction_frame)

                equal_metrics = binary_metrics(target["target"], equal_prior_probability)
                adapted_metrics = binary_metrics(target["target"], adapted_probability)
                bbse_metrics = binary_metrics(target["target"], bbse_probability)
                oracle_metrics = binary_metrics(target["target"], oracle_probability)
                result_records.append(
                    {
                        "scenario": scenario,
                        "experiment": experiment_name,
                        "variant": variant,
                        "seed": int(seed),
                        "fit_seed": fit_seed,
                        "best_epoch": result.best_epoch,
                        "parameter_count": result.parameter_count,
                        "true_prevalence": true_prevalence,
                        "estimated_prevalence": prevalence_estimate,
                        "prevalence_absolute_error": abs(prevalence_estimate - true_prevalence),
                        "soft_bbse_estimated_prevalence": bbse_prevalence_estimate,
                        "soft_bbse_prevalence_absolute_error": abs(
                            bbse_prevalence_estimate - true_prevalence
                        ),
                        "calibration_intercept": calibrator.intercept,
                        "calibration_slope": calibrator.slope,
                        "diagnostic_mode": diagnostic_mode,
                        "diagnostic_accepted": automatic_allowed,
                        "diagnostic_best_prior": diagnostic_best_prior,
                        "diagnostic_statistic": diagnostic_statistic,
                        "diagnostic_p_value": diagnostic_p_value,
                        "accepted_prior_lower": accepted_prior_lower,
                        "accepted_prior_upper": accepted_prior_upper,
                        "mask_support_passed": mask_support_passed,
                        "mask_all_feature_states_supported": (mask_all_feature_states_supported),
                        "mask_exact_pattern_support_rate": mask_exact_support_rate,
                        "mask_nearest_hamming_fraction_q95": mask_nearest_hamming_q95,
                        "mask_nearest_hamming_fraction_max": mask_nearest_hamming_max,
                        **{f"equal_{key}": value for key, value in equal_metrics.items()},
                        **{f"adapted_{key}": value for key, value in adapted_metrics.items()},
                        **{f"soft_bbse_{key}": value for key, value in bbse_metrics.items()},
                        **{f"oracle_{key}": value for key, value in oracle_metrics.items()},
                    }
                )
                diagnostic_table["scenario"] = scenario
                diagnostic_table["experiment"] = experiment_name
                diagnostic_table["variant"] = variant
                diagnostic_table["seed"] = int(seed)
                diagnostic_table["fit_seed"] = fit_seed
                diagnostic_table["diagnostic_mode"] = diagnostic_mode
                diagnostic_table["automatic_adaptation_allowed"] = automatic_allowed
                diagnostic_table["mask_support_passed"] = mask_support_passed
                diagnostic_table["mask_all_feature_states_supported"] = (
                    mask_all_feature_states_supported
                )
                diagnostic_table["mask_exact_pattern_support_rate"] = mask_exact_support_rate
                diagnostic_table["mask_nearest_hamming_fraction_q95"] = mask_nearest_hamming_q95
                diagnostic_table["mask_nearest_hamming_fraction_max"] = mask_nearest_hamming_max
                diagnostic_records.append(diagnostic_table)
                for row in result.history:
                    history_records.append(
                        {
                            "scenario": scenario,
                            "experiment": experiment_name,
                            "variant": variant,
                            "seed": int(seed),
                            **row,
                        }
                    )
                del result
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    paths = {
        "predictions": run_dir / "synthetic_target_predictions.parquet",
        "results": run_dir / "synthetic_results.csv",
        "diagnostics": run_dir / "mixture_diagnostics.csv",
        "history": run_dir / "training_history.csv",
        "summary": run_dir / "synthetic_summary.csv",
        "gate": run_dir / "acceptance_gate.json",
    }
    if diagnostic_mode == "composite_multiview_v3":
        paths["source_predictions"] = run_dir / "synthetic_source_predictions.parquet"
    predictions = pd.concat(predictions_records, ignore_index=True)
    source_predictions_output: pd.DataFrame | None = None
    results = pd.DataFrame(result_records)
    diagnostics = pd.concat(diagnostic_records, ignore_index=True)
    history = pd.DataFrame(history_records)
    summary = (
        results.groupby(["scenario", "experiment", "variant"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_roc_auc=("equal_roc_auc", "mean"),
            mean_balanced_log_loss=("equal_balanced_log_loss", "mean"),
            mean_equal_log_loss=("equal_log_loss", "mean"),
            mean_adapted_log_loss=("adapted_log_loss", "mean"),
            mean_soft_bbse_log_loss=("soft_bbse_log_loss", "mean"),
            mean_prevalence_absolute_error=("prevalence_absolute_error", "mean"),
            mean_soft_bbse_prevalence_absolute_error=(
                "soft_bbse_prevalence_absolute_error",
                "mean",
            ),
            diagnostic_acceptance_rate=("diagnostic_accepted", "mean"),
        )
        .sort_values(["scenario", "mean_balanced_log_loss"])
    )
    summary["mean_adapted_minus_equal_log_loss"] = (
        summary["mean_adapted_log_loss"] - summary["mean_equal_log_loss"]
    )
    predictions.to_parquet(paths["predictions"], index=False)
    if diagnostic_mode == "composite_multiview_v3":
        if not source_prediction_records:
            raise AssertionError("Protocol-v3 source diagnostic evidence is missing")
        source_predictions_output = pd.concat(source_prediction_records, ignore_index=True)
        source_predictions_output.to_parquet(paths["source_predictions"], index=False)
    results.to_csv(paths["results"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    history.to_csv(paths["history"], index=False)
    summary.to_csv(paths["summary"], index=False)
    persisted_summary = pd.read_csv(paths["summary"], float_precision="round_trip")
    if str(config.get("acceptance_gate_version", "v2")) == "v3":
        gate = evaluate_synthetic_v3_gates(persisted_summary, config["acceptance_gates"])
    else:
        gate = evaluate_synthetic_gates(persisted_summary, config["acceptance_gates"])
    paths["gate"].write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["audit"] = _write_synthetic_evidence_audit(
        run_dir=run_dir,
        artifact_paths=paths,
        config=config,
        predictions=predictions,
        source_predictions=source_predictions_output,
        results=results,
        diagnostics=diagnostics,
        summary=persisted_summary,
        history=history,
        gate=gate,
    )
    return paths
