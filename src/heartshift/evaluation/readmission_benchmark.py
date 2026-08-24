"""Source-only model selection for the independent readmission shift task."""

from __future__ import annotations

import gc
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from heartshift.config import config_hash
from heartshift.evaluation.classical_benchmark import write_run_manifest
from heartshift.masks import observed_mask_hashes
from heartshift.metrics import binary_metrics
from heartshift.models.classical import (
    build_generic_classical_pipeline,
    group_sample_weights,
)
from heartshift.models.generic_observed import (
    FeatureMaskPolicy,
    apply_feature_mask_policy,
    fit_generic_observed_fixed_epochs,
    fit_generic_observed_model,
    predict_generic_policy_bank,
)


def _parameter_grid(base: dict[str, Any], grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [dict(base)]
    keys = tuple(sorted(grid))
    return [
        dict(base) | dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def _policies(config: dict[str, Any]) -> tuple[FeatureMaskPolicy, ...]:
    return tuple(
        FeatureMaskPolicy(
            name=str(policy["name"]),
            kind=str(policy["kind"]),
            rate=float(policy.get("rate", 0.0)),
            columns=tuple(str(column) for column in policy.get("columns", [])),
        )
        for policy in config["mask_policies"]
    )


def load_readmission_modelling_frame(
    repo_root: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = pd.read_parquet(repo_root / config["data"]["canonical_path"])
    split = pd.read_parquet(repo_root / config["data"]["split_path"])
    profile = json.loads((repo_root / config["data"]["profile_path"]).read_text(encoding="utf-8"))
    frame = data.merge(
        split[["sample_id", "split"]], on="sample_id", how="inner", validate="one_to_one"
    )
    frame["target"] = frame[str(config["data"]["target"])].astype("int8")
    frame["environment"] = "admission_source_" + frame["admission_source_id"].astype(str)
    return frame, profile


def _prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouping = [
        "experiment",
        "backend",
        "variant",
        "model",
        "weighting",
        "parameter_id",
        "seed",
        "policy",
        "mask_replicate",
    ]
    for keys, group in predictions.groupby(grouping, sort=False):
        records.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                **binary_metrics(group["target"], group["y_score"]),
            }
        )
    return pd.DataFrame(records)


def _predict_classical_policy_bank(
    pipeline: Any,
    validation: pd.DataFrame,
    feature_columns: tuple[str, ...],
    policies: tuple[FeatureMaskPolicy, ...],
    *,
    base_seed: int,
    replicates: int,
) -> pd.DataFrame:
    natural = validation.loc[:, feature_columns].notna().to_numpy()
    records = []
    for policy in policies:
        policy_replicates = 1 if policy.kind in {"natural", "panel"} else replicates
        for replicate in range(policy_replicates):
            observed = apply_feature_mask_policy(
                natural,
                feature_columns,
                policy,
                base_seed=base_seed,
                replicate=replicate,
            )
            masked = validation.loc[:, feature_columns].mask(~observed)
            probabilities = np.asarray(pipeline.predict_proba(masked)[:, 1], dtype=np.float64)
            provenance = ["sample_id", "target", "split", "environment"]
            if "source_line_sha256" in validation:
                provenance.append("source_line_sha256")
            if "patient_nbr" in validation:
                provenance.append("patient_nbr")
            frame = validation.loc[:, provenance].copy()
            frame["policy"] = policy.name
            frame["mask_replicate"] = replicate
            frame["evidence_logit"] = np.log(
                np.clip(probabilities, 1e-7, 1 - 1e-7) / np.clip(1 - probabilities, 1e-7, 1 - 1e-7)
            )
            frame["y_score"] = probabilities
            frame["observed_fraction"] = observed.mean(axis=1)
            frame["observed_mask_sha256"] = observed_mask_hashes(observed, feature_columns)
            records.append(frame)
    return pd.concat(records, ignore_index=True)


def run_readmission_inner(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "readmission_source_only")
    shard_root = run_dir / "prediction_shards"
    shard_root.mkdir()
    frame, profile = load_readmission_modelling_frame(repo_root, config)
    training = frame.loc[frame["split"].isin(config["data"]["source_splits"])].copy()
    validation = frame.loc[frame["split"].eq(config["data"]["validation_split"])].copy()
    features = tuple(str(column) for column in profile["features"])
    continuous = tuple(str(column) for column in profile["continuous_features"])
    categorical = tuple(str(column) for column in profile["categorical_features"])
    policies = _policies(config)
    prediction_paths = []
    fit_records = []
    history_records = []
    device = str(config["device"])

    for experiment in config["experiments"]:
        experiment_name = str(experiment["name"])
        backend = str(experiment.get("backend", "neural"))
        variant = str(experiment.get("variant", "not_applicable"))
        model_name = str(experiment.get("model", "observed_set_transformer"))
        weighting = str(experiment.get("weighting", "not_applicable"))
        base_parameters = (dict(config["common_parameters"]) if backend == "neural" else {}) | dict(
            experiment.get("parameters", {})
        )
        candidates = _parameter_grid(
            base_parameters,
            dict(experiment.get("grid", {})),
        )
        for parameter_id, parameters in enumerate(candidates):
            parameters_json = json.dumps(parameters, sort_keys=True)
            for seed in config["seeds"]:
                fit_started = time.perf_counter()
                best_epoch: int | float
                parameter_count: int | float
                if backend == "neural":
                    result = fit_generic_observed_model(
                        training,
                        validation,
                        feature_columns=features,
                        continuous_columns=continuous,
                        categorical_columns=categorical,
                        evaluation_policies=policies,
                        variant=variant,
                        parameters=parameters,
                        seed=int(seed),
                        device=device,
                    )
                    predictions = predict_generic_policy_bank(
                        result,
                        validation,
                        policies,
                        device=device,
                        base_seed=int(seed) + 200_000,
                        replicates=int(config["mask_replicates"]),
                        batch_size=int(parameters["inference_batch_size"]),
                    )
                    best_epoch = result.best_epoch
                    validation_score = result.validation_score
                    parameter_count = result.parameter_count
                    for history in result.history:
                        history_records.append(
                            {
                                "experiment": experiment_name,
                                "backend": backend,
                                "variant": variant,
                                "model": model_name,
                                "weighting": weighting,
                                "parameter_id": parameter_id,
                                "seed": int(seed),
                                **history,
                            }
                        )
                    del result
                elif backend == "classical":
                    pipeline = build_generic_classical_pipeline(
                        model_name,
                        parameters,
                        int(seed),
                        feature_columns=features,
                        continuous_columns=continuous,
                        categorical_columns=categorical,
                    )
                    weights = group_sample_weights(
                        training,
                        weighting,
                        group_column="environment",
                    )
                    pipeline.fit(
                        training.loc[:, features],
                        training["target"],
                        model__sample_weight=weights,
                    )
                    predictions = _predict_classical_policy_bank(
                        pipeline,
                        validation,
                        features,
                        policies,
                        base_seed=int(seed) + 200_000,
                        replicates=int(config["mask_replicates"]),
                    )
                    provisional = predictions.groupby(
                        ["policy", "mask_replicate"], sort=False
                    ).apply(
                        lambda group: binary_metrics(group["target"], group["y_score"])[
                            "balanced_log_loss"
                        ],
                        include_groups=False,
                    )
                    validation_score = 0.5 * (float(provisional.mean()) + float(provisional.max()))
                    best_epoch = np.nan
                    parameter_count = np.nan
                    del pipeline
                else:
                    raise KeyError(f"Unknown readmission backend: {backend}")
                fit_seconds = time.perf_counter() - fit_started
                predictions["experiment"] = experiment_name
                predictions["backend"] = backend
                predictions["variant"] = variant
                predictions["model"] = model_name
                predictions["weighting"] = weighting
                predictions["parameter_id"] = parameter_id
                predictions["parameters_json"] = parameters_json
                predictions["seed"] = int(seed)
                predictions["config_sha256"] = config_hash(config)
                shard_path = shard_root / (f"{experiment_name}__p{parameter_id}__s{seed}.parquet")
                predictions.to_parquet(shard_path, index=False)
                prediction_paths.append(shard_path)
                fit_records.append(
                    {
                        "experiment": experiment_name,
                        "backend": backend,
                        "variant": variant,
                        "model": model_name,
                        "weighting": weighting,
                        "parameter_id": parameter_id,
                        "parameters_json": parameters_json,
                        "seed": int(seed),
                        "best_epoch": best_epoch,
                        "validation_score": validation_score,
                        "parameter_count": parameter_count,
                        "fit_seconds": fit_seconds,
                    }
                )
                del predictions
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    predictions = pd.concat((pd.read_parquet(path) for path in prediction_paths), ignore_index=True)
    metrics = _prediction_metrics(predictions)
    fits = pd.DataFrame(fit_records)
    selections = []
    for experiment, candidates in fits.groupby("experiment", sort=False):
        summary = candidates.groupby(
            [
                "backend",
                "variant",
                "model",
                "weighting",
                "parameter_id",
                "parameters_json",
            ],
            as_index=False,
            dropna=False,
        ).agg(
            mean_validation_score=("validation_score", "mean"),
            median_best_epoch=("best_epoch", "median"),
            mean_parameter_count=("parameter_count", "mean"),
        )
        selected = summary.sort_values(
            ["mean_validation_score", "mean_parameter_count", "parameter_id"]
        ).iloc[0]
        selections.append({"experiment": experiment, **selected.to_dict()})
    paths = {
        "predictions": run_dir / "validation_predictions.parquet",
        "metrics": run_dir / "validation_metrics.csv",
        "fits": run_dir / "fit_summaries.csv",
        "history": run_dir / "training_history.csv",
        "selections": run_dir / "selected_configurations.csv",
    }
    predictions.to_parquet(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    fits.to_csv(paths["fits"], index=False)
    pd.DataFrame(history_records).to_csv(paths["history"], index=False)
    pd.DataFrame(selections).to_csv(paths["selections"], index=False)
    return paths


def _load_readmission_outer_frames(
    repo_root: Path,
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
]:
    """Load source labels while keeping locked test labels out of memory."""
    profile = json.loads((repo_root / config["data"]["profile_path"]).read_text(encoding="utf-8"))
    split = pd.read_parquet(repo_root / config["data"]["split_path"])
    source_split_names = set(str(value) for value in config["data"]["refit_splits"])
    test_split_names = set(str(value) for value in config["data"]["test_splits"])
    if source_split_names & test_split_names:
        raise ValueError("Refit and test split names overlap")
    source_ids = split.loc[split["split"].isin(source_split_names), "sample_id"].tolist()
    test_ids = split.loc[split["split"].isin(test_split_names), "sample_id"].tolist()
    if set(source_ids) & set(test_ids):
        raise AssertionError("A sample occurs in both refit and locked test sets")

    features = [str(column) for column in profile["features"]]
    shared = [
        "sample_id",
        "patient_nbr",
        "admission_source_id",
        "source_line_sha256",
        *features,
    ]
    target_column = str(config["data"]["target"])
    canonical_path = repo_root / config["data"]["canonical_path"]
    source = pd.read_parquet(
        canonical_path,
        columns=[*shared, target_column],
        filters=[("sample_id", "in", source_ids)],
    ).merge(
        split[["sample_id", "split"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    target_unlabelled = pd.read_parquet(
        canonical_path,
        columns=shared,
        filters=[("sample_id", "in", test_ids)],
    ).merge(
        split[["sample_id", "split"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if set(source["sample_id"]) != set(source_ids):
        raise AssertionError("Source refit sample IDs did not load exactly")
    if set(target_unlabelled["sample_id"]) != set(test_ids):
        raise AssertionError("Locked test sample IDs did not load exactly")
    source["target"] = source[target_column].astype("int8")
    source["environment"] = "admission_source_" + source["admission_source_id"].astype(str)
    target_unlabelled["target"] = 0
    target_unlabelled["environment"] = "admission_source_" + target_unlabelled[
        "admission_source_id"
    ].astype(str)
    label_locator = split.loc[split["sample_id"].isin(test_ids), ["sample_id", "split"]].copy()
    return source, target_unlabelled, profile, label_locator


def _readmission_outer_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouping = [
        "experiment",
        "backend",
        "variant",
        "model",
        "weighting",
        "split",
        "policy",
        "mask_replicate",
    ]
    for keys, group in predictions.groupby(grouping, sort=False, dropna=False):
        records.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                **binary_metrics(group["target"], group["y_score"]),
            }
        )
    return pd.DataFrame(records)


def run_readmission_outer(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    """Run the frozen patient-disjoint ID/OOD test without test-label tuning."""
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "locked_independent_readmission_test")
    source, target_unlabelled, profile, label_locator = _load_readmission_outer_frames(
        repo_root, config
    )
    features = tuple(str(column) for column in profile["features"])
    continuous = tuple(str(column) for column in profile["continuous_features"])
    categorical = tuple(str(column) for column in profile["categorical_features"])
    policies = _policies(config)
    selections = pd.read_csv(repo_root / config["inner_run"] / "selected_configurations.csv")
    device = str(config["device"])
    config_sha256 = config_hash(config)
    shard_root = run_dir / "shards"
    shard_root.mkdir()
    ensemble_paths = []
    seed_paths = []

    for _, selected in selections.iterrows():
        experiment = str(selected["experiment"])
        backend = str(selected["backend"])
        variant = str(selected["variant"])
        model_name = str(selected["model"])
        weighting = str(selected["weighting"])
        parameters = json.loads(str(selected["parameters_json"]))
        parameter_id = int(selected["parameter_id"])
        epochs = (
            max(1, round(float(selected["median_best_epoch"]))) if backend == "neural" else None
        )
        shard = shard_root / experiment
        shard.mkdir()
        seed_records = []
        history_records = []
        for seed in (int(value) for value in config["seeds"]):
            if backend == "neural":
                if epochs is None:
                    raise AssertionError("A neural selection requires a fixed epoch count")
                result = fit_generic_observed_fixed_epochs(
                    source,
                    feature_columns=features,
                    continuous_columns=continuous,
                    categorical_columns=categorical,
                    evaluation_policies=policies,
                    variant=variant,
                    parameters=parameters,
                    seed=seed,
                    device=device,
                    epochs=epochs,
                )
                predictions = predict_generic_policy_bank(
                    result,
                    target_unlabelled,
                    policies,
                    device=device,
                    base_seed=int(config["evaluation_seed"]),
                    replicates=int(config["mask_replicates"]),
                    batch_size=int(parameters["inference_batch_size"]),
                )
                for history in result.history:
                    history_records.append({"seed": seed, **history})
                del result
            elif backend == "classical":
                pipeline = build_generic_classical_pipeline(
                    model_name,
                    parameters,
                    seed,
                    feature_columns=features,
                    continuous_columns=continuous,
                    categorical_columns=categorical,
                )
                weights = group_sample_weights(
                    source,
                    weighting,
                    group_column="environment",
                )
                pipeline.fit(
                    source.loc[:, features],
                    source["target"],
                    model__sample_weight=weights,
                )
                predictions = _predict_classical_policy_bank(
                    pipeline,
                    target_unlabelled,
                    features,
                    policies,
                    base_seed=int(config["evaluation_seed"]),
                    replicates=int(config["mask_replicates"]),
                )
                del pipeline
            else:
                raise KeyError(f"Unknown readmission backend: {backend}")
            predictions = predictions.drop(columns="target")
            predictions["seed"] = seed
            seed_records.append(predictions)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        seed_predictions = pd.concat(seed_records, ignore_index=True)
        grouping = [
            "sample_id",
            "patient_nbr",
            "split",
            "environment",
            "source_line_sha256",
            "policy",
            "mask_replicate",
        ]
        ensemble = seed_predictions.groupby(grouping, as_index=False).agg(
            evidence_logit=("evidence_logit", "mean"),
            y_score=("y_score", "mean"),
            observed_fraction=("observed_fraction", "mean"),
        )
        for frame in (seed_predictions, ensemble):
            frame["experiment"] = experiment
            frame["backend"] = backend
            frame["variant"] = variant
            frame["model"] = model_name
            frame["weighting"] = weighting
            frame["parameter_id"] = parameter_id
            frame["parameters_json"] = json.dumps(parameters, sort_keys=True)
            frame["epochs"] = epochs
            frame["config_sha256"] = config_sha256
        seed_path = shard / "unlabelled_seed_predictions.parquet"
        ensemble_path = shard / "unlabelled_ensemble_predictions.parquet"
        seed_predictions.to_parquet(seed_path, index=False)
        ensemble.to_parquet(ensemble_path, index=False)
        pd.DataFrame(history_records).to_csv(shard / "fit_history.csv", index=False)
        (shard / "selection.json").write_text(
            json.dumps(
                {
                    "experiment": experiment,
                    "backend": backend,
                    "variant": variant,
                    "model": model_name,
                    "weighting": weighting,
                    "parameter_id": parameter_id,
                    "parameters": parameters,
                    "epochs": epochs,
                    "seeds": [int(value) for value in config["seeds"]],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        seed_paths.append(seed_path)
        ensemble_paths.append(ensemble_path)
        del seed_predictions, ensemble, seed_records
        gc.collect()

    # The endpoint column is loaded only after every method and prediction is fixed on disk.
    target_column = str(config["data"]["target"])
    test_ids = label_locator["sample_id"].tolist()
    labels = pd.read_parquet(
        repo_root / config["data"]["canonical_path"],
        columns=["sample_id", target_column],
        filters=[("sample_id", "in", test_ids)],
    ).rename(columns={target_column: "target"})
    labels["target"] = labels["target"].astype("int8")
    predictions = pd.concat(
        (pd.read_parquet(path) for path in ensemble_paths), ignore_index=True
    ).merge(labels, on="sample_id", how="left", validate="many_to_one")
    if predictions["target"].isna().any():
        raise AssertionError("A locked test prediction is missing its endpoint")
    if set(predictions["sample_id"]) != set(test_ids):
        raise AssertionError("Locked test predictions do not cover every test sample")
    paths = {
        "predictions": run_dir / "test_predictions.parquet",
        "metrics": run_dir / "test_metrics.csv",
        "shard_index": run_dir / "shard_index.json",
    }
    predictions.to_parquet(paths["predictions"], index=False)
    _readmission_outer_metrics(predictions).to_csv(paths["metrics"], index=False)
    paths["shard_index"].write_text(
        json.dumps(
            {
                "ensemble_prediction_shards": [
                    str(path.relative_to(repo_root)) for path in ensemble_paths
                ],
                "seed_prediction_shards": [str(path.relative_to(repo_root)) for path in seed_paths],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
