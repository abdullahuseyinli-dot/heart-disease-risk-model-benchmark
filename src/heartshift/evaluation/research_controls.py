"""Fully nested, development-only structured-missingness classical controls."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heartshift.config import config_hash
from heartshift.data.uci import FEATURE_COLUMNS, sha256_file
from heartshift.evaluation.classical_benchmark import write_run_manifest
from heartshift.masks import (
    MaskPolicy,
    apply_mask_policy,
    default_policy_bank,
    mask_frame,
    observed_mask_codes,
    policy_seed,
)
from heartshift.metrics import binary_metrics
from heartshift.models.classical import (
    build_classical_pipeline,
    group_sample_weights,
    sample_weights,
)


def _parameter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    names = tuple(sorted(grid))
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(grid[name] for name in names))
    ]


def augment_training_policies(
    training: pd.DataFrame,
    *,
    augmentation: str,
    training_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Create mask-augmented rows while preserving natural missingness and provenance."""
    if training_replicates < 1:
        raise ValueError("Training augmentation replicates must be positive")
    if augmentation == "natural":
        result = training.copy()
        result["training_policy"] = "natural"
        result["training_mask_replicate"] = 0
        result["robust_environment"] = result["site"].astype(str) + "|natural"
        return result
    if augmentation != "structured_policy_bank":
        raise KeyError(f"Unknown classical augmentation: {augmentation}")
    empirical_pool = training.loc[:, FEATURE_COLUMNS].notna().to_numpy(dtype=bool)
    records = []
    for policy in default_policy_bank():
        replicates = 1 if policy.kind in {"natural", "panel"} else training_replicates
        for replicate in range(replicates):
            kwargs = {"empirical_mask_pool": empirical_pool} if policy.kind == "empirical" else {}
            observed = apply_mask_policy(
                training,
                policy,
                base_seed=seed,
                replicate=replicate,
                **kwargs,
            )
            masked = mask_frame(training, observed)
            masked["training_policy"] = policy.name
            masked["training_mask_replicate"] = replicate
            masked["robust_environment"] = masked["site"].astype(str) + "|" + policy.name
            records.append(masked)
    augmented = pd.concat(records, ignore_index=True)
    original_mask = training.loc[:, FEATURE_COLUMNS].notna().to_numpy(dtype=bool)
    for _, group in augmented.groupby(["training_policy", "training_mask_replicate"]):
        augmented_mask = group.loc[:, FEATURE_COLUMNS].notna().to_numpy(dtype=bool)
        if np.any(augmented_mask & ~original_mask):
            raise AssertionError("Classical augmentation revealed a naturally missing feature")
    return augmented


def _fit_control(
    control: dict[str, Any],
    parameters: dict[str, Any],
    training: pd.DataFrame,
    *,
    seed: int,
    training_replicates: int,
) -> Any:
    augmentation = str(control["augmentation"])
    augmented = augment_training_policies(
        training,
        augmentation=augmentation,
        training_replicates=training_replicates,
        seed=policy_seed(seed, "structured_training", 0),
    )
    model_name = str(control["model"])
    pipeline = build_classical_pipeline(model_name, parameters, seed)
    if augmentation == "structured_policy_bank":
        weights = group_sample_weights(
            augmented,
            "environment_class_balanced",
            group_column="robust_environment",
        )
    else:
        weights = sample_weights(augmented, str(control.get("weighting", "site_class_balanced")))
    pipeline.fit(
        augmented.loc[:, FEATURE_COLUMNS],
        augmented["target"],
        model__sample_weight=weights,
    )
    return pipeline


def _predict_policy_bank(
    pipeline: Any,
    validation: pd.DataFrame,
    policies: tuple[MaskPolicy, ...],
    *,
    empirical_pool: np.ndarray,
    base_seed: int,
    replicates: int,
) -> pd.DataFrame:
    records = []
    for policy in policies:
        policy_replicates = 1 if policy.kind in {"natural", "panel"} else replicates
        for replicate in range(policy_replicates):
            kwargs = {"empirical_mask_pool": empirical_pool} if policy.kind == "empirical" else {}
            observed = apply_mask_policy(
                validation,
                policy,
                base_seed=base_seed,
                replicate=replicate,
                **kwargs,
            )
            masked = mask_frame(validation, observed)
            probability = pipeline.predict_proba(masked.loc[:, FEATURE_COLUMNS])[:, 1]
            provenance = ["sample_id", "site", "record_sha256"]
            if "target" in validation:
                provenance.append("target")
            frame = validation.loc[:, provenance].copy()
            frame["policy"] = policy.name
            frame["mask_replicate"] = replicate
            frame["y_score"] = np.asarray(probability, dtype=np.float64)
            frame["observed_fraction"] = observed.mean(axis=1)
            frame["observed_mask_code"] = observed_mask_codes(observed)
            for feature_index, feature in enumerate(FEATURE_COLUMNS):
                frame[f"observed__{feature}"] = observed[:, feature_index]
            records.append(frame)
    return pd.concat(records, ignore_index=True)


def _cell_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "outer_target",
        "inner_validation",
        "control",
        "parameter_id",
        "parameters_json",
        "seed",
        "policy",
        "mask_replicate",
    ]
    records = []
    for keys, group in predictions.groupby(grouping, sort=False):
        records.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                **binary_metrics(group["target"], group["y_score"]),
            }
        )
    return pd.DataFrame(records)


def _select_nested_controls(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selections = []
    for (outer_target, control), candidates in metrics.groupby(["outer_target", "control"]):
        summary = candidates.groupby(
            ["parameter_id", "parameters_json"], as_index=False
        ).agg(
            mean_balanced_log_loss=("balanced_log_loss", "mean"),
            worst_balanced_log_loss=("balanced_log_loss", "max"),
            mean_roc_auc=("roc_auc", "mean"),
        )
        summary["selection_score"] = 0.5 * (
            summary["mean_balanced_log_loss"] + summary["worst_balanced_log_loss"]
        )
        selected = summary.sort_values(
            ["selection_score", "mean_roc_auc", "parameter_id"],
            ascending=[True, False, True],
        ).iloc[0]
        selections.append({"outer_target": outer_target, "control": control, **selected.to_dict()})
    selection_frame = pd.DataFrame(selections)
    anchor = (
        selection_frame.sort_values(
            ["outer_target", "selection_score", "mean_roc_auc", "control"],
            ascending=[True, True, False, True],
        )
        .groupby("outer_target", as_index=False)
        .first()
        .rename(columns={"control": "selected_anchor"})
    )
    return selection_frame, anchor


def run_research_control_benchmark(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    """Run nested tuning and descriptive outer predictions on consumed UCI data."""
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "consumed_uci_development_controls")
    shard_root = run_dir / "inner_prediction_shards"
    shard_root.mkdir()
    data_path = repo_root / str(config["data"]["canonical_path"])
    data = pd.read_parquet(data_path)
    sites = tuple(str(value) for value in data["site"].drop_duplicates())
    policies = default_policy_bank()
    seeds = tuple(int(value) for value in config["seeds"])
    mask_replicates = int(config["inner_mask_replicates"])
    training_replicates = int(config["training_mask_replicates"])
    fit_records = []
    shard_paths = []
    for outer_target in sites:
        source = data.loc[data["site"].ne(outer_target)].copy()
        for control in config["controls"]:
            control_name = str(control["name"])
            candidates = _parameter_grid(dict(control.get("grid", {})))
            for parameter_id, parameters in enumerate(candidates):
                parameters_json = json.dumps(parameters, sort_keys=True)
                for inner_validation in tuple(source["site"].drop_duplicates()):
                    training = source.loc[source["site"].ne(inner_validation)].copy()
                    validation = source.loc[source["site"].eq(inner_validation)].copy()
                    empirical_pool = training.loc[:, FEATURE_COLUMNS].notna().to_numpy(dtype=bool)
                    for seed in seeds:
                        fit_seed = policy_seed(
                            seed,
                            f"{outer_target}|{inner_validation}|{control_name}|p{parameter_id}",
                            0,
                        )
                        started = time.perf_counter()
                        pipeline = _fit_control(
                            control,
                            parameters,
                            training,
                            seed=fit_seed,
                            training_replicates=training_replicates,
                        )
                        fit_seconds = time.perf_counter() - started
                        predictions = _predict_policy_bank(
                            pipeline,
                            validation,
                            policies,
                            empirical_pool=empirical_pool,
                            base_seed=policy_seed(
                                int(config.get("inner_evaluation_seed", seed)),
                                str(inner_validation),
                                99,
                            ),
                            replicates=mask_replicates,
                        )
                        predictions["outer_target"] = outer_target
                        predictions["inner_validation"] = str(inner_validation)
                        predictions["control"] = control_name
                        predictions["model"] = str(control["model"])
                        predictions["augmentation"] = str(control["augmentation"])
                        predictions["parameter_id"] = parameter_id
                        predictions["parameters_json"] = parameters_json
                        predictions["seed"] = seed
                        predictions["fit_seed"] = fit_seed
                        predictions["config_sha256"] = config_hash(config)
                        shard_path = shard_root / (
                            f"{outer_target}__{inner_validation}__{control_name}"
                            f"__p{parameter_id}__s{seed}.parquet"
                        )
                        predictions.to_parquet(shard_path, index=False)
                        shard_paths.append(shard_path)
                        fit_records.append(
                            {
                                "outer_target": outer_target,
                                "inner_validation": inner_validation,
                                "control": control_name,
                                "parameter_id": parameter_id,
                                "parameters_json": parameters_json,
                                "seed": seed,
                                "fit_seed": fit_seed,
                                "fit_seconds": fit_seconds,
                            }
                        )
    inner_predictions = pd.concat(
        (pd.read_parquet(path) for path in shard_paths), ignore_index=True
    )
    inner_metrics = _cell_metrics(inner_predictions)
    selections, anchors = _select_nested_controls(inner_metrics)
    inner_predictions_path = run_dir / "inner_predictions.parquet"
    inner_metrics_path = run_dir / "inner_metrics.csv"
    selections_path = run_dir / "selected_configurations.csv"
    anchors_path = run_dir / "selected_anchors.csv"
    fits_path = run_dir / "fit_summaries.csv"
    inner_predictions.to_parquet(inner_predictions_path, index=False)
    inner_metrics.to_csv(inner_metrics_path, index=False)
    selections.to_csv(selections_path, index=False)
    anchors.to_csv(anchors_path, index=False)
    pd.DataFrame(fit_records).to_csv(fits_path, index=False)

    # All controls are evaluated for descriptive mechanism analysis only.  The
    # target endpoint is loaded after endpoint-free predictions are on disk.
    outer_shard_root = run_dir / "outer_unlabelled_shards"
    outer_shard_root.mkdir()
    outer_unlabelled_paths = []
    evaluation_seed_base = int(config["outer_evaluation_seed"])
    outer_replicates = int(config["outer_mask_replicates"])
    for _, selection in selections.iterrows():
        outer_target = str(selection["outer_target"])
        control_name = str(selection["control"])
        control = next(value for value in config["controls"] if value["name"] == control_name)
        parameters = json.loads(str(selection["parameters_json"]))
        source = pd.read_parquet(data_path, filters=[("site", "!=", outer_target)])
        target_unlabelled = pd.read_parquet(
            data_path,
            columns=["sample_id", "site", "record_sha256", *FEATURE_COLUMNS],
            filters=[("site", "==", outer_target)],
        )
        empirical_pool = source.loc[:, FEATURE_COLUMNS].notna().to_numpy(dtype=bool)
        evaluation_seed = policy_seed(
            evaluation_seed_base, f"{outer_target}|outer_evaluation", 0
        )
        for seed in seeds:
            fit_seed = policy_seed(seed, f"{outer_target}|{control_name}|outer_fit", 0)
            pipeline = _fit_control(
                control,
                parameters,
                source,
                seed=fit_seed,
                training_replicates=training_replicates,
            )
            predictions = _predict_policy_bank(
                pipeline,
                target_unlabelled,
                policies,
                empirical_pool=empirical_pool,
                base_seed=evaluation_seed,
                replicates=outer_replicates,
            )
            predictions["outer_target"] = outer_target
            predictions["control"] = control_name
            predictions["model"] = str(control["model"])
            predictions["augmentation"] = str(control["augmentation"])
            predictions["parameter_id"] = int(selection["parameter_id"])
            predictions["parameters_json"] = json.dumps(parameters, sort_keys=True)
            predictions["training_seed"] = seed
            predictions["fit_seed"] = fit_seed
            predictions["config_sha256"] = config_hash(config)
            shard_path = outer_shard_root / f"{outer_target}__{control_name}__s{seed}.parquet"
            predictions.to_parquet(shard_path, index=False)
            outer_unlabelled_paths.append(shard_path)
    outer_unlabelled = pd.concat(
        (pd.read_parquet(path) for path in outer_unlabelled_paths), ignore_index=True
    )
    outer_unlabelled_path = run_dir / "outer_unlabelled_predictions.parquet"
    outer_unlabelled.to_parquet(outer_unlabelled_path, index=False)
    labels = pd.read_parquet(data_path, columns=["sample_id", "target"])
    outer_predictions = outer_unlabelled.merge(
        labels, on="sample_id", how="left", validate="many_to_one"
    )
    if outer_predictions["target"].isna().any():
        raise AssertionError("A development-control target label is missing")
    outer_predictions_path = run_dir / "outer_predictions.parquet"
    outer_metrics_path = run_dir / "outer_metrics.csv"
    outer_predictions.to_parquet(outer_predictions_path, index=False)
    outer_metric_records = []
    grouping = [
        "outer_target",
        "control",
        "training_seed",
        "policy",
        "mask_replicate",
    ]
    for keys, group in outer_predictions.groupby(grouping, sort=False):
        outer_metric_records.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                **binary_metrics(group["target"], group["y_score"]),
            }
        )
    pd.DataFrame(outer_metric_records).to_csv(outer_metrics_path, index=False)
    audit_path = run_dir / "evidence_audit.json"
    artifact_paths = {
        "inner_predictions": inner_predictions_path,
        "inner_metrics": inner_metrics_path,
        "selections": selections_path,
        "anchors": anchors_path,
        "fits": fits_path,
        "outer_unlabelled": outer_unlabelled_path,
        "outer_predictions": outer_predictions_path,
        "outer_metrics": outer_metrics_path,
    }
    audit_path.write_text(
        json.dumps(
            {
                "status": "complete_consumed_uci_development_only",
                "new_confirmatory_claim_allowed": False,
                "fully_nested_selection": True,
                "individual_seed_predictions_retained": True,
                "sha256": {path.name: sha256_file(path) for path in artifact_paths.values()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_paths | {"audit": audit_path}
