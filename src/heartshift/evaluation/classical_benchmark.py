"""Nested leave-one-hospital-out evaluation for classical controls and baselines."""

from __future__ import annotations

import itertools
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heartshift.config import config_hash
from heartshift.data.uci import sha256_file
from heartshift.masks import (
    apply_mask_policy,
    default_policy_bank,
    mask_frame,
    policy_seed,
)
from heartshift.metrics import binary_metrics
from heartshift.models.classical import build_classical_pipeline, sample_weights, selected_features


def _parameter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = tuple(sorted(grid))
    combinations = itertools.product(*(grid[key] for key in keys))
    return [dict(zip(keys, values, strict=True)) for values in combinations]


def _git_value(repo_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def write_run_manifest(
    repo_root: Path,
    run_dir: Path,
    config: dict[str, Any],
    phase: str,
) -> None:
    canonical_path = config.get("data", {}).get("canonical_path")
    canonical_hash = None
    if canonical_path is not None:
        resolved_canonical_path = repo_root / str(canonical_path)
        if resolved_canonical_path.is_file():
            canonical_hash = sha256_file(resolved_canonical_path)
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "phase": phase,
        "config_sha256": config_hash(config),
        "canonical_data_sha256": canonical_hash,
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_status_porcelain": _git_value(repo_root, "status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "packages": {},
    }
    for package in (
        "catboost",
        "interpret",
        "lightgbm",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "pytabkit",
        "tabpfn",
        "tabicl",
        "tabm",
        "torch",
        "xgboost",
    ):
        try:
            manifest["packages"][package] = version(package)
        except PackageNotFoundError:
            manifest["packages"][package] = "not-installed"
    try:
        import torch

        manifest["torch"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["cuda_version"] = torch.version.cuda
            manifest["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        manifest["torch"] = "not-installed"
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "config.resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fit_predict(
    model_name: str,
    parameters: dict[str, Any],
    weighting: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    view = "core" if model_name == "core_logistic" else "all"
    if model_name == "mask_logistic":
        view = "mask"
    features = selected_features(view)
    pipeline = build_classical_pipeline(model_name, parameters, seed)
    weights = sample_weights(train, weighting)
    unweighted_models = {"tabpfn", "tabicl", "realmlp", "ft_transformer", "tabm"}
    if model_name in unweighted_models:
        if weighting != "pooled":
            raise ValueError(f"{model_name} does not consume sample weights; use pooled weighting")
        pipeline.fit(train.loc[:, features], train["target"])
    else:
        pipeline.fit(train.loc[:, features], train["target"], model__sample_weight=weights)
    fitted_model = pipeline.named_steps["model"]
    convergence_checked_models = {
        "logistic",
        "core_logistic",
        "mask_logistic",
        "elastic_net_logistic",
    }
    if (
        model_name in convergence_checked_models
        and hasattr(fitted_model, "n_iter_")
        and hasattr(fitted_model, "max_iter")
        and int(np.max(fitted_model.n_iter_)) >= int(fitted_model.max_iter)
    ):
        raise RuntimeError(f"{model_name} did not converge for parameters {parameters}")
    probabilities = pipeline.predict_proba(validation.loc[:, features])[:, 1]
    if len(probabilities) != len(validation):
        raise AssertionError("Prediction count changed during evaluation")
    return np.asarray(probabilities, dtype=np.float64)


def run_inner_benchmark(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "inner_source_only")
    data = pd.read_parquet(repo_root / config["data"]["canonical_path"])
    sites = tuple(data["site"].drop_duplicates())
    prediction_records: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    seeds = tuple(int(value) for value in config.get("seeds", [config.get("seed", 5062)]))
    fixed_selections = None
    if config.get("fixed_selection_run") is not None:
        fixed_selection_path = (
            repo_root / str(config["fixed_selection_run"]) / "selected_hyperparameters.csv"
        )
        fixed_selections = pd.read_csv(fixed_selection_path)
        (run_dir / "selection_provenance.json").write_text(
            json.dumps(
                {
                    "source": str(fixed_selection_path.relative_to(repo_root)),
                    "sha256": sha256_file(fixed_selection_path),
                    "mode": "source_only_fixed_candidate_confirmation",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    for outer_target in sites:
        source = data.loc[data["site"].ne(outer_target)].copy()
        if fixed_selections is None:
            model_specs = [
                (
                    str(model["name"]),
                    str(model.get("weighting", "pooled")),
                    _parameter_grid(model.get("grid", {})),
                )
                for model in config["models"]
            ]
        else:
            outer_rows = fixed_selections.loc[fixed_selections["outer_target"].eq(outer_target)]
            model_specs = [
                (
                    str(row["model"]),
                    str(row["weighting"]),
                    [json.loads(str(row["parameters_json"]))],
                )
                for _, row in outer_rows.iterrows()
            ]
        for model_name, weighting, parameter_grid in model_specs:
            for parameter_id, parameters in enumerate(parameter_grid):
                for inner_validation in tuple(source["site"].drop_duplicates()):
                    train = source.loc[source["site"].ne(inner_validation)].copy()
                    validation = source.loc[source["site"].eq(inner_validation)].copy()
                    for seed in seeds:
                        fit_seed = policy_seed(
                            seed,
                            f"{outer_target}|{inner_validation}|{model_name}|{weighting}",
                            0,
                        )
                        fit_started = time.perf_counter()
                        probabilities = _fit_predict(
                            model_name,
                            parameters,
                            weighting,
                            train,
                            validation,
                            fit_seed,
                        )
                        fit_seconds = time.perf_counter() - fit_started
                        fold_predictions = validation.loc[
                            :, ["sample_id", "site", "target", "record_sha256"]
                        ].copy()
                        fold_predictions["outer_target"] = outer_target
                        fold_predictions["inner_validation"] = inner_validation
                        fold_predictions["model"] = model_name
                        fold_predictions["weighting"] = weighting
                        fold_predictions["parameter_id"] = parameter_id
                        fold_predictions["parameters_json"] = json.dumps(parameters, sort_keys=True)
                        fold_predictions["seed"] = seed
                        fold_predictions["fit_seed"] = fit_seed
                        fold_predictions["y_score"] = probabilities
                        fold_predictions["config_sha256"] = config_hash(config)
                        prediction_records.append(fold_predictions)
                        metrics = binary_metrics(validation["target"], probabilities)
                        metric_records.append(
                            {
                                "outer_target": outer_target,
                                "inner_validation": inner_validation,
                                "model": model_name,
                                "weighting": weighting,
                                "parameter_id": parameter_id,
                                "parameters_json": json.dumps(parameters, sort_keys=True),
                                "seed": seed,
                                "fit_seed": fit_seed,
                                "fit_seconds": fit_seconds,
                                **metrics,
                            }
                        )

    predictions = pd.concat(prediction_records, ignore_index=True)
    metrics = pd.DataFrame.from_records(metric_records)
    selections: list[dict[str, Any]] = []
    for (outer_target, model_name, weighting), candidates in metrics.groupby(
        ["outer_target", "model", "weighting"]
    ):
        summaries = candidates.groupby(
            ["parameter_id", "parameters_json", "weighting"], as_index=False
        ).agg(
            mean_balanced_log_loss=("balanced_log_loss", "mean"),
            worst_balanced_log_loss=("balanced_log_loss", "max"),
            mean_roc_auc=("roc_auc", "mean"),
        )
        summaries["selection_score"] = 0.5 * (
            summaries["mean_balanced_log_loss"] + summaries["worst_balanced_log_loss"]
        )
        selected = summaries.sort_values(
            ["selection_score", "mean_roc_auc", "parameter_id"],
            ascending=[True, False, True],
        ).iloc[0]
        selections.append(
            {
                "outer_target": outer_target,
                "model": model_name,
                "weighting": weighting,
                **selected.to_dict(),
            }
        )

    predictions_path = run_dir / "inner_predictions.parquet"
    metrics_path = run_dir / "inner_metrics.csv"
    selections_path = run_dir / "selected_hyperparameters.csv"
    predictions.to_parquet(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(selections).to_csv(selections_path, index=False)
    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "selections": selections_path,
    }


def run_outer_benchmark(
    repo_root: Path,
    config: dict[str, Any],
    inner_run_dir: Path,
    run_dir: Path,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "locked_outer_loho")
    data_path = repo_root / config["data"]["canonical_path"]
    selections = pd.read_csv(inner_run_dir / "selected_hyperparameters.csv")
    seed_prediction_records: list[pd.DataFrame] = []
    seeds = tuple(int(value) for value in config.get("seeds", [config.get("seed", 5062)]))
    outer_mask_replicates = int(config.get("outer_mask_replicates", 1))
    policies = default_policy_bank()
    for _, selected in selections.iterrows():
        outer_target = str(selected["outer_target"])
        model_name = str(selected["model"])
        parameters = json.loads(str(selected["parameters_json"]))
        weighting = str(selected["weighting"])
        source = pd.read_parquet(data_path, filters=[("site", "!=", outer_target)])
        target_unlabelled = pd.read_parquet(
            data_path,
            columns=["sample_id", "site", "record_sha256", *selected_features("all")],
            filters=[("site", "==", outer_target)],
        )
        view = "core" if model_name == "core_logistic" else "all"
        if model_name == "mask_logistic":
            view = "mask"
        features = selected_features(view)
        empirical_pool = source.loc[:, selected_features("all")].notna().to_numpy()
        evaluation_seed = policy_seed(
            int(config.get("evaluation_seed", seeds[0])),
            f"{outer_target}|outer_evaluation",
            0,
        )
        for seed in seeds:
            fit_seed = policy_seed(seed, f"{outer_target}|{model_name}|{weighting}|outer_fit", 0)
            pipeline = build_classical_pipeline(model_name, parameters, fit_seed)
            weights = sample_weights(source, weighting)
            unweighted_models = {
                "tabpfn",
                "tabicl",
                "realmlp",
                "ft_transformer",
                "tabm",
            }
            if model_name in unweighted_models:
                if weighting != "pooled":
                    raise ValueError(
                        f"{model_name} does not consume sample weights; use pooled weighting"
                    )
                pipeline.fit(source.loc[:, features], source["target"])
            else:
                pipeline.fit(
                    source.loc[:, features],
                    source["target"],
                    model__sample_weight=weights,
                )
            for policy in policies:
                repetitions = 1 if policy.kind in {"natural", "panel"} else outer_mask_replicates
                for replicate in range(repetitions):
                    kwargs = (
                        {"empirical_mask_pool": empirical_pool}
                        if policy.kind == "empirical"
                        else {}
                    )
                    observed = apply_mask_policy(
                        target_unlabelled,
                        policy,
                        base_seed=evaluation_seed,
                        replicate=replicate,
                        **kwargs,
                    )
                    masked = mask_frame(target_unlabelled, observed)
                    probabilities = np.asarray(
                        pipeline.predict_proba(masked.loc[:, features])[:, 1],
                        dtype=np.float64,
                    )
                    fold_predictions = target_unlabelled.loc[
                        :, ["sample_id", "site", "record_sha256"]
                    ].copy()
                    fold_predictions["outer_target"] = outer_target
                    fold_predictions["model"] = model_name
                    fold_predictions["weighting"] = weighting
                    fold_predictions["training_seed"] = seed
                    fold_predictions["fit_seed"] = fit_seed
                    fold_predictions["y_score"] = probabilities
                    fold_predictions["policy"] = policy.name
                    fold_predictions["mask_replicate"] = replicate
                    fold_predictions["observed_fraction"] = observed.mean(axis=1)
                    fold_predictions["config_sha256"] = config_hash(config)
                    seed_prediction_records.append(fold_predictions)

    seed_predictions = pd.concat(seed_prediction_records, ignore_index=True)
    grouping = [
        "sample_id",
        "site",
        "record_sha256",
        "outer_target",
        "model",
        "weighting",
        "policy",
        "mask_replicate",
        "config_sha256",
    ]
    predictions = seed_predictions.groupby(grouping, as_index=False).agg(
        y_score=("y_score", "mean"),
        observed_fraction=("observed_fraction", "mean"),
    )
    # Labels are loaded only after every model and mask-policy prediction is fixed.
    labels = pd.read_parquet(data_path, columns=["sample_id", "target"])
    predictions = predictions.merge(labels, on="sample_id", how="left", validate="many_to_one")
    seed_predictions = seed_predictions.merge(
        labels, on="sample_id", how="left", validate="many_to_one"
    )
    if predictions["target"].isna().any() or seed_predictions["target"].isna().any():
        raise AssertionError("An outer prediction is missing its endpoint")
    metric_records = []
    for (outer_target, model_name, weighting, policy, replicate), group in predictions.groupby(
        ["outer_target", "model", "weighting", "policy", "mask_replicate"]
    ):
        metric_records.append(
            {
                "outer_target": outer_target,
                "model": model_name,
                "weighting": weighting,
                "policy": policy,
                "mask_replicate": replicate,
                **binary_metrics(group["target"], group["y_score"]),
            }
        )
    metrics = pd.DataFrame(metric_records)
    predictions_path = run_dir / "outer_predictions.parquet"
    seed_predictions_path = run_dir / "outer_seed_predictions.parquet"
    metrics_path = run_dir / "outer_metrics.csv"
    predictions.to_parquet(predictions_path, index=False)
    seed_predictions.to_parquet(seed_predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    return {
        "predictions": predictions_path,
        "seed_predictions": seed_predictions_path,
        "metrics": metrics_path,
    }
