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
from heartshift.models.ps_maskdro import fit_ps_maskdro_fixed_epochs, predict_policy_bank


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
    ensemble = (
        seed_predictions.groupby(grouping, as_index=False)
        .agg(
            evidence_logit=("evidence_logit", "mean"),
            y_score_zero_shot=("y_score", "mean"),
            observed_fraction=("observed_fraction", "mean"),
        )
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
    adapted_records = []
    diagnostic_records = []
    summary_records = []
    for (policy, replicate), target_group in target_predictions.groupby(
        ["policy", "mask_replicate"], sort=False
    ):
        target_group = target_group.copy()
        source_group = source_predictions.loc[
            source_predictions["evaluation_policy"].eq(policy)
            & source_predictions["policy_replicate"].eq(replicate)
        ].copy()
        target_group["calibrated_evidence_logit"] = np.nan
        target_group["estimated_target_prevalence_mlls"] = np.nan
        target_group["estimated_target_prevalence_soft_bbse"] = np.nan
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
            diagnostic = mixture_fit_diagnostic(
                source_evidence[source_group["target"].to_numpy() == 0],
                source_evidence[source_group["target"].to_numpy() == 1],
                target_evidence,
                prior_grid=diagnostic_config["prior_grid"],
                bootstrap_repetitions=int(diagnostic_config["bootstrap_repetitions"]),
                mixture_draws=int(diagnostic_config["mixture_draws"]),
                alpha=float(diagnostic_config["alpha"]),
                seed=policy_seed(base_seed, str(policy), int(replicate)),
            )
            allowed = diagnostic.accepted_interval is not None
            target_group["calibrated_evidence_logit"] = target_evidence
            target_group["estimated_target_prevalence_mlls"] = prevalence_mlls
            target_group["estimated_target_prevalence_soft_bbse"] = prevalence_soft_bbse
            target_group["adaptation_allowed"] = allowed
            target_group["adaptation_status"] = "accepted" if allowed else "diagnostic_rejected"
            if allowed:
                target_group["y_score_uda_mlls"] = posterior_from_evidence(
                    target_evidence, prevalence_mlls
                )
                target_group["y_score_uda_soft_bbse"] = posterior_from_evidence(
                    target_evidence, prevalence_soft_bbse
                )
            diagnostic_table = diagnostic.table.copy()
            diagnostic_table["policy"] = policy
            diagnostic_table["mask_replicate"] = replicate
            diagnostic_records.append(diagnostic_table)
            summary_records.append(
                {
                    "policy": policy,
                    "mask_replicate": replicate,
                    "calibration_intercept": calibrator.intercept,
                    "calibration_slope": calibrator.slope,
                    "estimated_target_prevalence_mlls": prevalence_mlls,
                    "estimated_target_prevalence_soft_bbse": prevalence_soft_bbse,
                    "diagnostic_best_prior": diagnostic.best_prior,
                    "diagnostic_best_statistic": diagnostic.best_statistic,
                    "diagnostic_accepted_interval_low": (
                        diagnostic.accepted_interval[0]
                        if diagnostic.accepted_interval is not None
                        else np.nan
                    ),
                    "diagnostic_accepted_interval_high": (
                        diagnostic.accepted_interval[1]
                        if diagnostic.accepted_interval is not None
                        else np.nan
                    ),
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


def run_neural_outer(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    """Run a source-frozen outer study, writing a recoverable shard per method/site."""
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "locked_outer_psmask")
    data_path = repo_root / config["data"]["canonical_path"]
    data = pd.read_parquet(data_path)
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
        source = data.loc[data["site"].ne(outer_target)].copy()
        target_unlabelled = data.loc[data["site"].eq(outer_target)].copy()
        target_unlabelled["target"] = 0
        source_mask_pool = source.loc[:, config["features"]].notna().to_numpy()
        evaluation_seed = policy_seed(seeds[0], f"{outer_target}|outer_evaluation", 0)
        adaptable = variant in set(config["adaptable_variants"])
        if adaptable:
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

    prediction_files = sorted(shard_root.glob("*/*outer_predictions.parquet"))
    seed_files = sorted(shard_root.glob("*/*outer_seed_predictions.parquet"))
    crossfit_files = sorted(shard_root.glob("*/*source_calibration_predictions.parquet"))
    diagnostic_files = sorted(shard_root.glob("*/*adaptation_diagnostics.csv"))
    summary_files = sorted(shard_root.glob("*/*adaptation_summary.csv"))
    history_files = sorted(shard_root.glob("*/*fit_history.csv"))
    predictions = pd.concat((pd.read_parquet(path) for path in prediction_files), ignore_index=True)
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
