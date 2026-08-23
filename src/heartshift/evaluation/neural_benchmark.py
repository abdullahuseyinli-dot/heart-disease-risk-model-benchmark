"""Source-only and locked-outer evaluation for PS-MaskDRO ablations."""

from __future__ import annotations

import gc
import itertools
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from heartshift.config import config_hash
from heartshift.data.uci import sha256_file
from heartshift.evaluation.classical_benchmark import write_run_manifest
from heartshift.masks import default_policy_bank, policy_seed
from heartshift.metrics import binary_metrics
from heartshift.models.ps_maskdro import fit_ps_maskdro, predict_policy_bank


def _parameter_grid(
    base: dict[str, Any],
    grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    if not grid:
        return [dict(base)]
    keys = tuple(sorted(grid))
    combinations = itertools.product(*(grid[key] for key in keys))
    return [dict(base) | dict(zip(keys, values, strict=True)) for values in combinations]


def _prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouping = [
        "outer_target",
        "inner_validation",
        "experiment",
        "variant",
        "parameter_id",
        "parameters_json",
        "seed",
        "policy",
        "mask_replicate",
    ]
    for keys, group in predictions.groupby(grouping, dropna=False):
        records.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                **binary_metrics(group["target"], group["y_score"]),
            }
        )
    return pd.DataFrame(records)


def run_neural_inner(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    """Run all architecture/objective choices without opening an outer target."""
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "psmask_inner_source_only")
    shard_root = run_dir / "prediction_shards"
    shard_root.mkdir()
    data = pd.read_parquet(repo_root / config["data"]["canonical_path"])
    sites = tuple(data["site"].drop_duplicates())
    prediction_paths: list[Path] = []
    fit_records: list[dict[str, Any]] = []
    history_records: list[dict[str, Any]] = []
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    fixed_selections: dict[str, pd.DataFrame] = {}
    configured_sources: dict[str, Any] = {}
    if config.get("fixed_selection_run") is not None:
        configured_sources["default"] = config["fixed_selection_run"]
    configured_sources.update(dict(config.get("fixed_selection_runs", {})))
    if configured_sources:
        provenance_records = []
        for source_name, relative_run in configured_sources.items():
            fixed_selection_path = repo_root / str(relative_run) / "selected_configurations.csv"
            fixed_selections[str(source_name)] = pd.read_csv(fixed_selection_path)
            provenance_records.append(
                {
                    "selection_source": str(source_name),
                    "path": str(fixed_selection_path.relative_to(repo_root)),
                    "sha256": sha256_file(fixed_selection_path),
                }
            )
        (run_dir / "selection_provenance.json").write_text(
            json.dumps(
                {
                    "mode": "source_only_fixed_candidate_confirmation",
                    "sources": provenance_records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    for outer_target in sites:
        source = data.loc[data["site"].ne(outer_target)].copy()
        for experiment in config["experiments"]:
            experiment_name = str(experiment["name"])
            variant = str(experiment["variant"])
            if not fixed_selections:
                candidates = _parameter_grid(
                    dict(config["common_parameters"]) | dict(experiment.get("parameters", {})),
                    dict(experiment.get("grid", {})),
                )
            else:
                source_name = experiment_name if experiment_name in fixed_selections else "default"
                if source_name not in fixed_selections:
                    raise KeyError(f"No fixed selection source for {experiment_name}")
                selection_table = fixed_selections[source_name]
                matched = selection_table.loc[
                    selection_table["outer_target"].eq(outer_target)
                    & selection_table["experiment"].eq(experiment_name)
                ]
                if len(matched) != 1:
                    raise AssertionError(
                        "Fixed neural selection must contain one row per outer target/experiment"
                    )
                if str(matched.iloc[0]["variant"]) != variant:
                    raise AssertionError("Fixed neural selection variant changed")
                candidates = [json.loads(str(matched.iloc[0]["parameters_json"]))]
            for parameter_id, parameters in enumerate(candidates):
                parameters_json = json.dumps(parameters, sort_keys=True)
                for inner_validation in tuple(source["site"].drop_duplicates()):
                    training = source.loc[source["site"].ne(inner_validation)].copy()
                    validation = source.loc[source["site"].eq(inner_validation)].copy()
                    empirical_pool = training.loc[:, config["features"]].notna().to_numpy()
                    for seed in config["seeds"]:
                        fit_seed = policy_seed(
                            int(seed),
                            f"{outer_target}|{inner_validation}",
                            0,
                        )
                        fit_started = time.perf_counter()
                        result = fit_ps_maskdro(
                            training,
                            validation,
                            variant=variant,
                            parameters=parameters,
                            seed=fit_seed,
                            device=device,
                        )
                        fit_seconds = time.perf_counter() - fit_started
                        predictions = predict_policy_bank(
                            result.model,
                            result.preprocessor,
                            validation,
                            default_policy_bank(),
                            device=torch.device(device),
                            base_seed=policy_seed(int(seed), str(inner_validation), 99),
                            replicates=int(config["inner_mask_replicates"]),
                            empirical_mask_pool=empirical_pool,
                        )
                        predictions["outer_target"] = outer_target
                        predictions["inner_validation"] = inner_validation
                        predictions["experiment"] = experiment_name
                        predictions["variant"] = variant
                        predictions["parameter_id"] = parameter_id
                        predictions["parameters_json"] = parameters_json
                        predictions["seed"] = int(seed)
                        predictions["fit_seed"] = fit_seed
                        predictions["config_sha256"] = config_hash(config)
                        shard_path = shard_root / (
                            f"{outer_target}__{inner_validation}__{experiment_name}"
                            f"__p{parameter_id}__s{seed}.parquet"
                        )
                        predictions.to_parquet(shard_path, index=False)
                        prediction_paths.append(shard_path)
                        fit_records.append(
                            {
                                "outer_target": outer_target,
                                "inner_validation": inner_validation,
                                "experiment": experiment_name,
                                "variant": variant,
                                "parameter_id": parameter_id,
                                "parameters_json": parameters_json,
                                "seed": int(seed),
                                "fit_seed": fit_seed,
                                "best_epoch": result.best_epoch,
                                "validation_score": result.validation_score,
                                "parameter_count": result.parameter_count,
                                "fit_seconds": fit_seconds,
                            }
                        )
                        for row in result.history:
                            history_records.append(
                                {
                                    "outer_target": outer_target,
                                    "inner_validation": inner_validation,
                                    "experiment": experiment_name,
                                    "variant": variant,
                                    "parameter_id": parameter_id,
                                    "seed": int(seed),
                                    **row,
                                }
                            )
                        del result, predictions
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    predictions = pd.concat((pd.read_parquet(path) for path in prediction_paths), ignore_index=True)
    metrics = _prediction_metrics(predictions)
    fits = pd.DataFrame(fit_records)
    selections = []
    for (outer_target, experiment_name), candidates in fits.groupby(["outer_target", "experiment"]):
        summaries = candidates.groupby(
            ["variant", "parameter_id", "parameters_json"], as_index=False
        ).agg(
            mean_validation_score=("validation_score", "mean"),
            worst_validation_score=("validation_score", "max"),
            median_best_epoch=("best_epoch", "median"),
            mean_parameter_count=("parameter_count", "mean"),
        )
        summaries["selection_score"] = 0.5 * (
            summaries["mean_validation_score"] + summaries["worst_validation_score"]
        )
        selected = summaries.sort_values(
            ["selection_score", "mean_parameter_count", "parameter_id"]
        ).iloc[0]
        selections.append(
            {
                "outer_target": outer_target,
                "experiment": experiment_name,
                **selected.to_dict(),
            }
        )

    paths = {
        "predictions": run_dir / "inner_predictions.parquet",
        "metrics": run_dir / "inner_metrics.csv",
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
