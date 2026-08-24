"""Locked outer evaluation for PS-MaskDRO, including assumption-gated UDA."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

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
from heartshift.evaluation.classical_benchmark import write_run_manifest
from heartshift.masks import MaskPolicy, apply_mask_policy, default_policy_bank, policy_seed
from heartshift.metrics import binary_metrics
from heartshift.models.ps_maskdro import (
    CORE_PREDICTION_COLUMNS,
    MASK_PREDICTION_COLUMNS,
    fit_ps_maskdro_fixed_epochs,
    predict_policy_bank,
)

DIAGNOSTIC_VIEW_COLUMNS = (
    *CORE_PREDICTION_COLUMNS,
    *MASK_PREDICTION_COLUMNS,
    "observed_mask_code",
)


def evaluation_units(
    outer_mask_replicates: int,
) -> tuple[tuple[MaskPolicy, int], ...]:
    """Return the exact policy/replicate pairs emitted by ``predict_policy_bank``."""
    if outer_mask_replicates < 1:
        raise ValueError("outer_mask_replicates must be positive")
    units: list[tuple[MaskPolicy, int]] = []
    for policy in default_policy_bank():
        repetitions = 1 if policy.kind in {"natural", "panel"} else outer_mask_replicates
        units.extend((policy, replicate) for replicate in range(repetitions))
    return tuple(units)


def _fit_seed(seed: int, *parts: str) -> int:
    return policy_seed(seed, "|".join(parts), 0)


def _free_model(result: Any) -> None:
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _selected_parameters(row: pd.Series) -> tuple[dict[str, Any], int]:
    parameters = json.loads(str(row["parameters_json"]))
    epochs = max(1, round(float(row["median_best_epoch"])))
    parameters["max_epochs"] = epochs
    return parameters, epochs


def _load_heart_outer_frames(
    data_path: Path,
    *,
    outer_target: str,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load source endpoints without bringing the locked target endpoint into memory."""
    source = pd.read_parquet(data_path, filters=[("site", "!=", outer_target)])
    target_columns = [
        "sample_id",
        "site",
        "record_sha256",
        *feature_columns,
    ]
    target_unlabelled = pd.read_parquet(
        data_path,
        columns=list(dict.fromkeys(target_columns)),
        filters=[("site", "==", outer_target)],
    )
    if source.empty or target_unlabelled.empty:
        raise AssertionError(f"Outer split is empty for {outer_target}")
    if source["site"].eq(outer_target).any():
        raise AssertionError("Outer-target rows leaked into the labelled source frame")
    if not target_unlabelled["site"].eq(outer_target).all():
        raise AssertionError("Unlabelled outer frame contains a non-target hospital")
    if "target" in target_unlabelled:
        raise AssertionError("Locked outer endpoint was loaded before prediction")
    if target_unlabelled["sample_id"].duplicated().any():
        raise AssertionError("Unlabelled outer frame contains duplicate sample IDs")
    return source, target_unlabelled


def _target_mask_pools(
    target_unlabelled: pd.DataFrame,
    source_mask_pool: np.ndarray,
    units: tuple[tuple[MaskPolicy, int], ...],
    *,
    base_seed: int,
) -> dict[tuple[str, int], np.ndarray]:
    pools = {}
    for policy, replicate in units:
        kwargs = {"empirical_mask_pool": source_mask_pool} if policy.kind == "empirical" else {}
        pools[(policy.name, replicate)] = apply_mask_policy(
            target_unlabelled,
            policy,
            base_seed=base_seed,
            replicate=replicate,
            **kwargs,
        )
    return pools


def _crossfit_target_policy_scores(
    source: pd.DataFrame,
    target_mask_pools: dict[tuple[str, int], np.ndarray],
    *,
    outer_target: str,
    experiment: str,
    variant: str,
    parameters: dict[str, Any],
    epochs: int,
    seeds: tuple[int, ...],
    match_replicates: int,
    device: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records = []
    histories: list[dict[str, Any]] = []
    for inner_validation in tuple(source["site"].drop_duplicates()):
        training = source.loc[source["site"].ne(inner_validation)].copy()
        validation = source.loc[source["site"].eq(inner_validation)].copy()
        for seed in seeds:
            fit_seed = _fit_seed(
                seed,
                outer_target,
                experiment,
                "outer_crossfit",
                str(inner_validation),
            )
            result = fit_ps_maskdro_fixed_epochs(
                training,
                variant=variant,
                parameters=parameters,
                seed=fit_seed,
                device=device,
                epochs=epochs,
            )
            for (policy_name, policy_replicate), target_mask_pool in target_mask_pools.items():
                matched = predict_policy_bank(
                    result.model,
                    result.preprocessor,
                    validation,
                    (MaskPolicy("target_policy_matched", "empirical"),),
                    device=torch.device(device),
                    base_seed=policy_seed(fit_seed, policy_name, policy_replicate),
                    replicates=match_replicates,
                    empirical_mask_pool=target_mask_pool,
                ).rename(columns={"mask_replicate": "match_replicate"})
                matched = matched.drop(columns=["policy", "y_score"])
                matched["evaluation_policy"] = policy_name
                matched["policy_replicate"] = policy_replicate
                matched["training_seed"] = seed
                matched["fit_seed"] = fit_seed
                records.append(matched)
            for history in result.history:
                histories.append(
                    {
                        "stage": "outer_crossfit",
                        "outer_target": outer_target,
                        "inner_validation": inner_validation,
                        "experiment": experiment,
                        "variant": variant,
                        "training_seed": seed,
                        "fit_seed": fit_seed,
                        **history,
                    }
                )
            _free_model(result)

    predictions = pd.concat(records, ignore_index=True)
    grouping = [
        "sample_id",
        "site",
        "target",
        "record_sha256",
        "evaluation_policy",
        "policy_replicate",
        "match_replicate",
    ]
    predictions = (
        predictions.groupby(grouping, as_index=False)
        .agg(
            evidence_logit=("evidence_logit", "mean"),
            observed_fraction=("observed_fraction", "mean"),
        )
        .sort_values(grouping)
        .reset_index(drop=True)
    )
    return predictions, histories


def _crossfit_same_policy_scores(
    source: pd.DataFrame,
    source_mask_pool: np.ndarray,
    *,
    outer_target: str,
    experiment: str,
    variant: str,
    parameters: dict[str, Any],
    epochs: int,
    seeds: tuple[int, ...],
    outer_mask_replicates: int,
    evaluation_seed: int,
    device: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Cross-fit one unique source prediction under each target evaluation policy.

    Unlike the failed protocol-v2 mask-intersection route, this applies the same
    named intervention separately to source and target natural observations. It
    never attempts to reveal a naturally missing source value and never treats
    repeated masks of one patient as independent diagnostic observations.
    """
    records = []
    histories: list[dict[str, Any]] = []
    for inner_validation in tuple(source["site"].drop_duplicates()):
        training = source.loc[source["site"].ne(inner_validation)].copy()
        validation = source.loc[source["site"].eq(inner_validation)].copy()
        for seed in seeds:
            fit_seed = _fit_seed(
                seed,
                outer_target,
                experiment,
                "outer_crossfit_v3_same_policy",
                str(inner_validation),
            )
            result = fit_ps_maskdro_fixed_epochs(
                training,
                variant=variant,
                parameters=parameters,
                seed=fit_seed,
                device=device,
                epochs=epochs,
            )
            predictions = predict_policy_bank(
                result.model,
                result.preprocessor,
                validation,
                default_policy_bank(),
                device=torch.device(device),
                base_seed=evaluation_seed,
                replicates=outer_mask_replicates,
                empirical_mask_pool=source_mask_pool,
            ).rename(
                columns={
                    "policy": "evaluation_policy",
                    "mask_replicate": "policy_replicate",
                }
            )
            predictions = predictions.drop(columns=["y_score"])
            predictions["training_seed"] = seed
            predictions["fit_seed"] = fit_seed
            records.append(predictions)
            for history in result.history:
                histories.append(
                    {
                        "stage": "outer_crossfit_v3_same_policy",
                        "outer_target": outer_target,
                        "inner_validation": inner_validation,
                        "experiment": experiment,
                        "variant": variant,
                        "training_seed": seed,
                        "fit_seed": fit_seed,
                        **history,
                    }
                )
            _free_model(result)

    predictions = pd.concat(records, ignore_index=True)
    grouping = [
        "sample_id",
        "site",
        "target",
        "record_sha256",
        "evaluation_policy",
        "policy_replicate",
    ]
    for column in DIAGNOSTIC_VIEW_COLUMNS:
        if predictions.groupby(grouping, sort=False)[column].nunique(dropna=False).max() != 1:
            raise AssertionError(f"Source diagnostic view changed across seeds: {column}")
    aggregation: dict[str, tuple[str, str]] = {
        "evidence_logit": ("evidence_logit", "mean"),
        "observed_fraction": ("observed_fraction", "mean"),
    }
    aggregation.update({column: (column, "first") for column in DIAGNOSTIC_VIEW_COLUMNS})
    predictions = (
        predictions.groupby(grouping, as_index=False)
        .agg(**aggregation)
        .sort_values(grouping)
        .reset_index(drop=True)
    )
    if predictions.duplicated([*grouping]).any():
        raise AssertionError("Protocol-v3 cross-fit output contains repeated source patients")
    return predictions, histories


def _final_target_predictions(
    source: pd.DataFrame,
    target_unlabelled: pd.DataFrame,
    *,
    outer_target: str,
    experiment: str,
    variant: str,
    parameters: dict[str, Any],
    epochs: int,
    seeds: tuple[int, ...],
    outer_mask_replicates: int,
    evaluation_seed: int,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    source_mask_pool = source.loc[:, parameters["feature_columns"]].notna().to_numpy()
    seed_records = []
    histories: list[dict[str, Any]] = []
    for seed in seeds:
        fit_seed = _fit_seed(seed, outer_target, experiment, "outer_final")
        result = fit_ps_maskdro_fixed_epochs(
            source,
            variant=variant,
            parameters=parameters,
            seed=fit_seed,
            device=device,
            epochs=epochs,
        )
        predictions = predict_policy_bank(
            result.model,
            result.preprocessor,
            target_unlabelled,
            default_policy_bank(),
            device=torch.device(device),
            base_seed=evaluation_seed,
            replicates=outer_mask_replicates,
            empirical_mask_pool=source_mask_pool,
        ).drop(columns=["target"])
        predictions["training_seed"] = seed
        predictions["fit_seed"] = fit_seed
        seed_records.append(predictions)
        for history in result.history:
            histories.append(
                {
                    "stage": "outer_final",
                    "outer_target": outer_target,
                    "inner_validation": None,
                    "experiment": experiment,
                    "variant": variant,
                    "training_seed": seed,
                    "fit_seed": fit_seed,
                    **history,
                }
            )
        _free_model(result)

    seed_predictions = pd.concat(seed_records, ignore_index=True)
    grouping = ["sample_id", "site", "record_sha256", "policy", "mask_replicate"]
    for column in DIAGNOSTIC_VIEW_COLUMNS:
        if seed_predictions.groupby(grouping, sort=False)[column].nunique(dropna=False).max() != 1:
            raise AssertionError(f"Target diagnostic view changed across seeds: {column}")
    aggregation: dict[str, tuple[str, str]] = {
        "evidence_logit": ("evidence_logit", "mean"),
        "y_score_zero_shot": ("y_score", "mean"),
        "observed_fraction": ("observed_fraction", "mean"),
    }
    aggregation.update({column: (column, "first") for column in DIAGNOSTIC_VIEW_COLUMNS})
    ensemble = (
        seed_predictions.groupby(grouping, as_index=False)
        .agg(**aggregation)
        .sort_values(grouping)
        .reset_index(drop=True)
    )
    return ensemble, seed_predictions, histories


def _adapt_predictions(
    target_predictions: pd.DataFrame,
    source_predictions: pd.DataFrame,
    *,
    adaptable: bool,
    diagnostic_config: dict[str, Any],
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adapted_records: list[pd.DataFrame] = []
    diagnostic_records: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    diagnostic_mode = str(diagnostic_config.get("mode", "energy_v2"))
    if diagnostic_mode not in {"energy_v2", "composite_multiview_v3"}:
        raise ValueError(f"Unknown outer diagnostic mode: {diagnostic_mode}")
    for (policy, replicate), target_group in target_predictions.groupby(
        ["policy", "mask_replicate"], sort=False
    ):
        target_group = target_group.copy()
        if adaptable:
            required_source_columns = {"evaluation_policy", "policy_replicate"}
            if missing := required_source_columns - set(source_predictions.columns):
                raise AssertionError(
                    "Adaptable outer predictions require source-calibration columns: "
                    f"{sorted(missing)}"
                )
            source_group = source_predictions.loc[
                source_predictions["evaluation_policy"].eq(policy)
                & source_predictions["policy_replicate"].eq(replicate)
            ].copy()
            if source_group.empty:
                raise AssertionError(
                    "Adaptable outer predictions require a matching source-calibration group "
                    f"for policy={policy!r}, mask_replicate={replicate!r}"
                )
        else:
            source_group = pd.DataFrame()
        target_group["calibrated_evidence_logit"] = np.nan
        target_group["y_score_calibrated_equal_prior"] = np.nan
        target_group["estimated_target_prevalence_mlls"] = np.nan
        target_group["estimated_target_prevalence_soft_bbse"] = np.nan
        target_group["y_score_uda_mlls_research"] = np.nan
        target_group["y_score_uda_soft_bbse_research"] = np.nan
        target_group["y_score_uda_mlls"] = np.nan
        target_group["y_score_uda_soft_bbse"] = np.nan
        target_group["adaptation_allowed"] = False
        target_group["adaptation_status"] = "not_applicable"
        if adaptable:
            calibrator = fit_evidence_calibrator(
                source_group["evidence_logit"],
                source_group["target"],
                source_group["site"],
            )
            source_evidence = calibrator.transform(source_group["evidence_logit"])
            target_evidence = calibrator.transform(target_group["evidence_logit"])
            prevalence_mlls = estimate_target_prevalence_mlls(target_evidence)
            prevalence_soft_bbse = estimate_target_prevalence_soft_bbse(
                posterior_from_evidence(source_evidence, 0.5),
                source_group["target"],
                posterior_from_evidence(target_evidence, 0.5),
            )
            if diagnostic_mode == "composite_multiview_v3":
                if not source_group["sample_id"].is_unique:
                    raise AssertionError(
                        "Protocol-v3 outer diagnostics require unique source patients"
                    )
                if not target_group["sample_id"].is_unique:
                    raise AssertionError(
                        "Protocol-v3 outer diagnostics require unique target patients"
                    )
                diagnostic = acquisition_aware_label_shift_diagnostic(
                    source_evidence,
                    source_group["target"],
                    source_group.loc[:, CORE_PREDICTION_COLUMNS],
                    source_group.loc[:, MASK_PREDICTION_COLUMNS],
                    target_evidence,
                    target_group.loc[:, CORE_PREDICTION_COLUMNS],
                    target_group.loc[:, MASK_PREDICTION_COLUMNS],
                    source_sample_ids=source_group["sample_id"],
                    target_sample_ids=target_group["sample_id"],
                    prior_grid=diagnostic_config["prior_grid"],
                    bootstrap_repetitions=int(diagnostic_config["bootstrap_repetitions"]),
                    rff_features_per_view=int(diagnostic_config["rff_features_per_view"]),
                    alpha=float(diagnostic_config["alpha"]),
                    maximum_nearest_hamming_fraction=float(
                        diagnostic_config["maximum_nearest_hamming_fraction"]
                    ),
                    seed=policy_seed(base_seed, str(policy), int(replicate)),
                )
                allowed = diagnostic.accepted
                diagnostic_table = diagnostic.table.copy()
                diagnostic_best_prior = diagnostic.best_prior
                diagnostic_best_statistic = diagnostic.best_statistic
                diagnostic_p_value = diagnostic.p_value
                accepted_interval_low = np.nan
                accepted_interval_high = np.nan
                mask_support_passed = diagnostic.mask_support.passed
                mask_all_feature_states_supported = (
                    diagnostic.mask_support.all_feature_states_supported
                )
                mask_exact_pattern_support_rate = diagnostic.mask_support.exact_pattern_support_rate
                mask_nearest_hamming_fraction_q95 = (
                    diagnostic.mask_support.nearest_hamming_fraction_q95
                )
                mask_nearest_hamming_fraction_max = (
                    diagnostic.mask_support.nearest_hamming_fraction_max
                )
            else:
                legacy_diagnostic = mixture_fit_diagnostic(
                    source_evidence[source_group["target"].to_numpy() == 0],
                    source_evidence[source_group["target"].to_numpy() == 1],
                    target_evidence,
                    prior_grid=diagnostic_config["prior_grid"],
                    bootstrap_repetitions=int(diagnostic_config["bootstrap_repetitions"]),
                    mixture_draws=int(diagnostic_config["mixture_draws"]),
                    alpha=float(diagnostic_config["alpha"]),
                    seed=policy_seed(base_seed, str(policy), int(replicate)),
                )
                allowed = legacy_diagnostic.accepted_interval is not None
                diagnostic_table = legacy_diagnostic.table.copy()
                diagnostic_best_prior = legacy_diagnostic.best_prior
                diagnostic_best_statistic = legacy_diagnostic.best_statistic
                best_row = diagnostic_table.sort_values(["energy_statistic", "prevalence"]).iloc[0]
                diagnostic_p_value = float(best_row["p_value"])
                accepted_interval_low = (
                    legacy_diagnostic.accepted_interval[0]
                    if legacy_diagnostic.accepted_interval is not None
                    else np.nan
                )
                accepted_interval_high = (
                    legacy_diagnostic.accepted_interval[1]
                    if legacy_diagnostic.accepted_interval is not None
                    else np.nan
                )
                mask_support_passed = True
                mask_all_feature_states_supported = True
                mask_exact_pattern_support_rate = np.nan
                mask_nearest_hamming_fraction_q95 = np.nan
                mask_nearest_hamming_fraction_max = np.nan

            equal_prior_calibrated = posterior_from_evidence(target_evidence, 0.5)
            research_mlls = posterior_from_evidence(target_evidence, prevalence_mlls)
            research_soft_bbse = posterior_from_evidence(target_evidence, prevalence_soft_bbse)
            target_group["calibrated_evidence_logit"] = target_evidence
            target_group["y_score_calibrated_equal_prior"] = equal_prior_calibrated
            target_group["estimated_target_prevalence_mlls"] = prevalence_mlls
            target_group["estimated_target_prevalence_soft_bbse"] = prevalence_soft_bbse
            target_group["y_score_uda_mlls_research"] = research_mlls
            target_group["y_score_uda_soft_bbse_research"] = research_soft_bbse
            target_group["adaptation_allowed"] = allowed
            target_group["adaptation_status"] = "accepted" if allowed else "diagnostic_rejected"
            if allowed:
                target_group["y_score_uda_mlls"] = research_mlls
                target_group["y_score_uda_soft_bbse"] = research_soft_bbse
            diagnostic_table["policy"] = policy
            diagnostic_table["mask_replicate"] = replicate
            diagnostic_table["diagnostic_mode"] = diagnostic_mode
            diagnostic_table["automatic_adaptation_allowed"] = allowed
            diagnostic_table["mask_support_passed"] = mask_support_passed
            diagnostic_table["mask_all_feature_states_supported"] = (
                mask_all_feature_states_supported
            )
            diagnostic_table["mask_exact_pattern_support_rate"] = mask_exact_pattern_support_rate
            diagnostic_table["mask_nearest_hamming_fraction_q95"] = (
                mask_nearest_hamming_fraction_q95
            )
            diagnostic_table["mask_nearest_hamming_fraction_max"] = (
                mask_nearest_hamming_fraction_max
            )
            diagnostic_records.append(diagnostic_table)
            summary_records.append(
                {
                    "policy": policy,
                    "mask_replicate": replicate,
                    "diagnostic_mode": diagnostic_mode,
                    "calibration_intercept": calibrator.intercept,
                    "calibration_slope": calibrator.slope,
                    "estimated_target_prevalence_mlls": prevalence_mlls,
                    "estimated_target_prevalence_soft_bbse": prevalence_soft_bbse,
                    "diagnostic_best_prior": diagnostic_best_prior,
                    "diagnostic_best_statistic": diagnostic_best_statistic,
                    "diagnostic_p_value": diagnostic_p_value,
                    "diagnostic_accepted_interval_low": accepted_interval_low,
                    "diagnostic_accepted_interval_high": accepted_interval_high,
                    "mask_support_passed": mask_support_passed,
                    "mask_all_feature_states_supported": (mask_all_feature_states_supported),
                    "mask_exact_pattern_support_rate": (mask_exact_pattern_support_rate),
                    "mask_nearest_hamming_fraction_q95": (mask_nearest_hamming_fraction_q95),
                    "mask_nearest_hamming_fraction_max": (mask_nearest_hamming_fraction_max),
                    "adaptation_allowed": allowed,
                }
            )
        adapted_records.append(target_group)
    diagnostics = (
        pd.concat(diagnostic_records, ignore_index=True) if diagnostic_records else pd.DataFrame()
    )
    return (
        pd.concat(adapted_records, ignore_index=True),
        diagnostics,
        pd.DataFrame(summary_records),
    )


def _score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouping = ["outer_target", "experiment", "variant", "policy", "mask_replicate"]
    for keys, group in predictions.groupby(grouping, sort=False):
        base = dict(zip(grouping, keys, strict=True))
        records.append(
            {
                **base,
                "track": "dg_zero_shot",
                **binary_metrics(group["target"], group["y_score_zero_shot"]),
            }
        )
        if bool(group["adaptation_allowed"].iloc[0]):
            records.append(
                {
                    **base,
                    "track": "uda_mlls",
                    **binary_metrics(group["target"], group["y_score_uda_mlls"]),
                }
            )
            records.append(
                {
                    **base,
                    "track": "uda_soft_bbse",
                    **binary_metrics(group["target"], group["y_score_uda_soft_bbse"]),
                }
            )
    return pd.DataFrame(records)


def _exact_shard_files(shard_root: Path, filename: str) -> list[Path]:
    """Return only an exact shard basename, never a prefixed endpoint-free file."""
    if Path(filename).name != filename:
        raise ValueError(f"Shard filename must be a basename: {filename}")
    return sorted(shard_root.glob(f"*/{filename}"))


def aggregate_neural_outer_shards(shard_root: Path, run_dir: Path) -> dict[str, Path]:
    """Aggregate exact shard files after every probability and endpoint join is fixed."""
    prediction_files = _exact_shard_files(shard_root, "outer_predictions.parquet")
    seed_files = _exact_shard_files(shard_root, "outer_seed_predictions.parquet")
    crossfit_files = _exact_shard_files(shard_root, "source_calibration_predictions.parquet")
    diagnostic_files = _exact_shard_files(shard_root, "adaptation_diagnostics.csv")
    summary_files = _exact_shard_files(shard_root, "adaptation_summary.csv")
    history_files = _exact_shard_files(shard_root, "fit_history.csv")
    if not prediction_files or len(prediction_files) != len(seed_files):
        raise AssertionError("Neural labelled prediction shards are missing or unpaired")
    predictions = pd.concat((pd.read_parquet(path) for path in prediction_files), ignore_index=True)
    if predictions["target"].isna().any():
        raise AssertionError("An exact labelled neural shard contains a missing endpoint")
    paths = {
        "predictions": run_dir / "outer_predictions.parquet",
        "seed_predictions": run_dir / "outer_seed_predictions.parquet",
        "calibration_predictions": run_dir / "source_calibration_predictions.parquet",
        "diagnostics": run_dir / "adaptation_diagnostics.csv",
        "adaptation_summary": run_dir / "adaptation_summary.csv",
        "history": run_dir / "fit_history.csv",
        "metrics": run_dir / "outer_metrics.csv",
    }
    predictions.to_parquet(paths["predictions"], index=False)
    pd.concat((pd.read_parquet(path) for path in seed_files), ignore_index=True).to_parquet(
        paths["seed_predictions"], index=False
    )
    pd.concat((pd.read_parquet(path) for path in crossfit_files), ignore_index=True).to_parquet(
        paths["calibration_predictions"], index=False
    )
    if diagnostic_files:
        pd.concat((pd.read_csv(path) for path in diagnostic_files), ignore_index=True).to_csv(
            paths["diagnostics"], index=False
        )
    if summary_files:
        pd.concat((pd.read_csv(path) for path in summary_files), ignore_index=True).to_csv(
            paths["adaptation_summary"], index=False
        )
    pd.concat((pd.read_csv(path) for path in history_files), ignore_index=True).to_csv(
        paths["history"], index=False
    )
    _score_predictions(predictions).to_csv(paths["metrics"], index=False)
    return paths


def run_neural_outer(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    """Run a source-frozen outer study, writing a recoverable shard per method/site."""
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(
        repo_root,
        run_dir,
        config,
        str(config.get("manifest_stage", "locked_outer_psmask")),
    )
    data_path = repo_root / config["data"]["canonical_path"]
    selections = pd.read_csv(repo_root / config["inner_run"] / "selected_configurations.csv")
    seeds = tuple(int(seed) for seed in config["seeds"])
    units = evaluation_units(int(config["outer_mask_replicates"]))
    device = str(config["device"])
    shard_root = run_dir / "shards"
    shard_root.mkdir()

    for _, selected in selections.iterrows():
        outer_target = str(selected["outer_target"])
        experiment = str(selected["experiment"])
        variant = str(selected["variant"])
        shard = shard_root / f"{outer_target}__{experiment}"
        shard.mkdir()
        parameters, epochs = _selected_parameters(selected)
        parameters["feature_columns"] = list(config["features"])
        feature_columns = tuple(str(value) for value in config["features"])
        source, target_unlabelled = _load_heart_outer_frames(
            data_path,
            outer_target=outer_target,
            feature_columns=feature_columns,
        )
        # Prediction helpers retain a target column in their generic output contract.
        # This sentinel is created only after the locked endpoint-free frame is loaded.
        target_unlabelled["target"] = 0
        source_mask_pool = source.loc[:, config["features"]].notna().to_numpy()
        evaluation_seed = policy_seed(seeds[0], f"{outer_target}|outer_evaluation", 0)
        adaptable = variant in set(config["adaptable_variants"])
        if adaptable:
            diagnostic_mode = str(config["diagnostic"].get("mode", "energy_v2"))
            if diagnostic_mode == "composite_multiview_v3":
                crossfit, crossfit_history = _crossfit_same_policy_scores(
                    source,
                    source_mask_pool,
                    outer_target=outer_target,
                    experiment=experiment,
                    variant=variant,
                    parameters=parameters,
                    epochs=epochs,
                    seeds=seeds,
                    outer_mask_replicates=int(config["outer_mask_replicates"]),
                    evaluation_seed=evaluation_seed,
                    device=device,
                )
            else:
                mask_pools = _target_mask_pools(
                    target_unlabelled,
                    source_mask_pool,
                    units,
                    base_seed=evaluation_seed,
                )
                crossfit, crossfit_history = _crossfit_target_policy_scores(
                    source,
                    mask_pools,
                    outer_target=outer_target,
                    experiment=experiment,
                    variant=variant,
                    parameters=parameters,
                    epochs=epochs,
                    seeds=seeds,
                    match_replicates=int(config["target_policy_match_replicates"]),
                    device=device,
                )
        else:
            crossfit = pd.DataFrame()
            crossfit_history = []
        ensemble, seed_predictions, final_history = _final_target_predictions(
            source,
            target_unlabelled,
            outer_target=outer_target,
            experiment=experiment,
            variant=variant,
            parameters=parameters,
            epochs=epochs,
            seeds=seeds,
            outer_mask_replicates=int(config["outer_mask_replicates"]),
            evaluation_seed=evaluation_seed,
            device=device,
        )
        adapted, diagnostics, adaptation_summary = _adapt_predictions(
            ensemble,
            crossfit,
            adaptable=adaptable,
            diagnostic_config=config["diagnostic"],
            base_seed=evaluation_seed,
        )
        for frame in (adapted, seed_predictions, crossfit, diagnostics, adaptation_summary):
            frame["outer_target"] = outer_target
            frame["experiment"] = experiment
            frame["variant"] = variant
            frame["epochs"] = epochs
            frame["config_sha256"] = config_hash(config)
        adapted.to_parquet(shard / "unlabelled_outer_predictions.parquet", index=False)
        seed_predictions.to_parquet(
            shard / "unlabelled_outer_seed_predictions.parquet", index=False
        )
        crossfit.to_parquet(shard / "source_calibration_predictions.parquet", index=False)
        diagnostics.to_csv(shard / "adaptation_diagnostics.csv", index=False)
        adaptation_summary.to_csv(shard / "adaptation_summary.csv", index=False)
        pd.DataFrame(crossfit_history + final_history).to_csv(
            shard / "fit_history.csv", index=False
        )
        (shard / "selection.json").write_text(
            json.dumps(
                {
                    "outer_target": outer_target,
                    "experiment": experiment,
                    "variant": variant,
                    "parameters": parameters,
                    "epochs": epochs,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        gc.collect()

    # No target endpoint is loaded until every model, UDA decision, and probability is on disk.
    labels = pd.read_parquet(data_path, columns=["sample_id", "site", "target"])
    for shard in sorted(shard_root.iterdir()):
        outer_target = shard.name.split("__", maxsplit=1)[0]
        target_labels = labels.loc[labels["site"].eq(outer_target), ["sample_id", "target"]]
        for source_name, destination_name in (
            ("unlabelled_outer_predictions.parquet", "outer_predictions.parquet"),
            ("unlabelled_outer_seed_predictions.parquet", "outer_seed_predictions.parquet"),
        ):
            unlabelled = pd.read_parquet(shard / source_name)
            labelled = unlabelled.merge(
                target_labels,
                on="sample_id",
                how="left",
                validate="many_to_one",
            )
            if labelled["target"].isna().any():
                raise AssertionError("An outer prediction is missing its endpoint")
            labelled.to_parquet(shard / destination_name, index=False)

    return aggregate_neural_outer_shards(shard_root, run_dir)
