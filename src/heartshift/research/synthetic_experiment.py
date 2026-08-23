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
    estimate_target_prevalence_mlls,
    estimate_target_prevalence_soft_bbse,
    fit_evidence_calibrator,
    mixture_fit_diagnostic,
    posterior_from_evidence,
)
from heartshift.config import config_hash
from heartshift.data.synthetic import generate_synthetic_environments
from heartshift.data.uci import FEATURE_COLUMNS
from heartshift.evaluation.classical_benchmark import write_run_manifest
from heartshift.masks import MaskPolicy, policy_seed
from heartshift.metrics import binary_metrics
from heartshift.models.ps_maskdro import fit_ps_maskdro, predict_policy_bank
from heartshift.research.gates import evaluate_synthetic_gates


def run_synthetic_experiment(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "synthetic_mechanism_validation")
    device = torch.device(str(config.get("device", "cuda")))
    predictions_records: list[pd.DataFrame] = []
    result_records: list[dict[str, Any]] = []
    diagnostic_records: list[pd.DataFrame] = []
    history_records: list[dict[str, Any]] = []

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

                # UDA policy matching uses target masks but never target labels.
                matched_source = predict_policy_bank(
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
                    matched_source["evidence_logit"],
                    matched_source["target"],
                    matched_source["site"],
                )
                source_evidence = calibrator.transform(matched_source["evidence_logit"])
                target_evidence = calibrator.transform(target_predictions["evidence_logit"])
                prevalence_estimate = estimate_target_prevalence_mlls(target_evidence)
                bbse_prevalence_estimate = estimate_target_prevalence_soft_bbse(
                    posterior_from_evidence(source_evidence, 0.5),
                    matched_source["target"],
                    posterior_from_evidence(target_evidence, 0.5),
                )
                diagnostic = mixture_fit_diagnostic(
                    source_evidence[matched_source["target"].to_numpy() == 0],
                    source_evidence[matched_source["target"].to_numpy() == 1],
                    target_evidence,
                    prior_grid=config["diagnostic"]["prior_grid"],
                    bootstrap_repetitions=int(config["diagnostic"]["bootstrap_repetitions"]),
                    mixture_draws=int(config["diagnostic"]["mixture_draws"]),
                    alpha=float(config["diagnostic"]["alpha"]),
                    seed=fit_seed + 2,
                )
                true_prevalence = float(target["target"].mean())
                equal_prior_probability = posterior_from_evidence(target_evidence, 0.5)
                adapted_probability = posterior_from_evidence(target_evidence, prevalence_estimate)
                bbse_probability = posterior_from_evidence(
                    target_evidence, bbse_prevalence_estimate
                )
                oracle_probability = posterior_from_evidence(target_evidence, true_prevalence)
                automatic_allowed = diagnostic.accepted_interval is not None

                prediction_frame = target_predictions.loc[
                    :, ["sample_id", "site", "target", "record_sha256"]
                ].copy()
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
                        "diagnostic_accepted": automatic_allowed,
                        "accepted_prior_lower": (
                            diagnostic.accepted_interval[0]
                            if diagnostic.accepted_interval is not None
                            else np.nan
                        ),
                        "accepted_prior_upper": (
                            diagnostic.accepted_interval[1]
                            if diagnostic.accepted_interval is not None
                            else np.nan
                        ),
                        **{f"equal_{key}": value for key, value in equal_metrics.items()},
                        **{f"adapted_{key}": value for key, value in adapted_metrics.items()},
                        **{f"soft_bbse_{key}": value for key, value in bbse_metrics.items()},
                        **{f"oracle_{key}": value for key, value in oracle_metrics.items()},
                    }
                )
                diagnostic_table = diagnostic.table.copy()
                diagnostic_table["scenario"] = scenario
                diagnostic_table["experiment"] = experiment_name
                diagnostic_table["variant"] = variant
                diagnostic_table["seed"] = int(seed)
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
    predictions = pd.concat(predictions_records, ignore_index=True)
    results = pd.DataFrame(result_records)
    diagnostics = pd.concat(diagnostic_records, ignore_index=True)
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
    results.to_csv(paths["results"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    pd.DataFrame(history_records).to_csv(paths["history"], index=False)
    summary.to_csv(paths["summary"], index=False)
    gate = evaluate_synthetic_gates(summary, config["acceptance_gates"])
    paths["gate"].write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths
