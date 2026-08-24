"""End-to-end model smoke test on the public eICU demo split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heartshift.config import config_hash
from heartshift.data.uci import sha256_file
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
    fit_generic_observed_model,
    predict_generic_policy_bank,
)


def _mask_generic_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    observed: np.ndarray,
) -> pd.DataFrame:
    result = frame.copy()
    for index, feature in enumerate(feature_columns):
        result[feature] = result[feature].mask(~observed[:, index])
    if np.any(result.loc[:, feature_columns].notna().to_numpy() & ~observed):
        raise AssertionError("Generic smoke masking retained a hidden value")
    return result


def _policies(config: dict[str, Any]) -> tuple[FeatureMaskPolicy, ...]:
    return tuple(
        FeatureMaskPolicy(
            name=str(value["name"]),
            kind=str(value["kind"]),
            rate=float(value.get("rate", 0.0)),
            columns=tuple(str(column) for column in value.get("columns", [])),
        )
        for value in config["policies"]
    )


def _classical_predictions(
    training: pd.DataFrame,
    target: pd.DataFrame,
    *,
    model_name: str,
    parameters: dict[str, Any],
    feature_columns: tuple[str, ...],
    continuous_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    policies: tuple[FeatureMaskPolicy, ...],
    seed: int,
    mask_replicates: int,
) -> pd.DataFrame:
    pipeline = build_generic_classical_pipeline(
        model_name,
        parameters,
        seed,
        feature_columns=feature_columns,
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    weights = group_sample_weights(
        training, "environment_class_balanced", group_column="environment"
    )
    pipeline.fit(
        training.loc[:, feature_columns],
        training["target"],
        model__sample_weight=weights,
    )
    _, natural = (
        np.zeros((len(target), len(feature_columns))),
        target.loc[:, feature_columns].notna().to_numpy(dtype=bool),
    )
    records = []
    for policy in policies:
        replicates = 1 if policy.kind in {"natural", "panel"} else mask_replicates
        for replicate in range(replicates):
            observed = apply_feature_mask_policy(
                natural,
                feature_columns,
                policy,
                base_seed=seed + 100_000,
                replicate=replicate,
            )
            masked = _mask_generic_frame(target, feature_columns, observed)
            frame = target.loc[:, ["sample_id", "environment", "target"]].copy()
            frame["policy"] = policy.name
            frame["mask_replicate"] = replicate
            frame["y_score"] = pipeline.predict_proba(masked.loc[:, feature_columns])[:, 1]
            frame["observed_fraction"] = observed.mean(axis=1)
            frame["observed_mask_sha256"] = observed_mask_hashes(observed, feature_columns)
            frame["experiment"] = f"classical:{model_name}"
            records.append(frame)
    return pd.concat(records, ignore_index=True)


def run_external_demo_smoke(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    """Exercise classical and observed-set backbones; never produce a science claim."""
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "public_eicu_demo_pipeline_smoke")
    data_dir = repo_root / str(config["data_dir"])
    profile = json.loads((data_dir / "profile.json").read_text(encoding="utf-8"))
    if profile.get("scientific_evaluation_allowed") is not False:
        raise AssertionError("External demo smoke requires an explicit no-science profile")
    canonical = pd.read_parquet(data_dir / "canonical.parquet")
    manifest = pd.read_parquet(data_dir / "hospital_split.parquet")
    encounter_column = str(config["encounter_id_column"])
    data = canonical.merge(
        manifest.loc[:, ["encounter_id", "split", "quarantine_reason"]],
        left_on=encounter_column,
        right_on="encounter_id",
        how="left",
        validate="one_to_one",
    )
    training = data.loc[data["split"].eq("development")].copy()
    validation = data.loc[data["split"].eq("architecture_selection")].copy()
    target = data.loc[data["split"].eq("locked_confirmation")].copy()
    if any(frame.empty for frame in (training, validation, target)):
        raise AssertionError("Public demo smoke lacks one active split role")
    feature_columns = tuple(str(value) for value in config["feature_columns"])
    continuous_columns = tuple(str(value) for value in config["continuous_columns"])
    categorical_columns = tuple(str(value) for value in config["categorical_columns"])
    policies = _policies(config)
    seed = int(config["seed"])
    prediction_records = []
    for model in config["classical_models"]:
        prediction_records.append(
            _classical_predictions(
                training,
                target,
                model_name=str(model["name"]),
                parameters=dict(model.get("parameters", {})),
                feature_columns=feature_columns,
                continuous_columns=continuous_columns,
                categorical_columns=categorical_columns,
                policies=policies,
                seed=seed,
                mask_replicates=int(config["mask_replicates"]),
            )
        )
    for backbone in config["observed_backbones"]:
        parameters = dict(config["observed_parameters"])
        parameters["backbone"] = str(backbone)
        result = fit_generic_observed_model(
            training,
            validation,
            feature_columns=feature_columns,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            evaluation_policies=policies,
            variant="prior_separated",
            parameters=parameters,
            seed=seed,
            device=str(config["device"]),
        )
        predictions = predict_generic_policy_bank(
            result,
            target,
            policies,
            device=str(config["device"]),
            base_seed=seed + 100_000,
            replicates=int(config["mask_replicates"]),
            batch_size=int(parameters.get("inference_batch_size", 512)),
        )
        predictions["experiment"] = f"observed:{backbone}"
        prediction_records.append(predictions)
    predictions = pd.concat(prediction_records, ignore_index=True)
    metric_records = []
    for keys, group in predictions.groupby(["experiment", "policy", "mask_replicate"], sort=False):
        metric_records.append(
            {
                "experiment": keys[0],
                "policy": keys[1],
                "mask_replicate": keys[2],
                **binary_metrics(group["target"], group["y_score"]),
            }
        )
    metrics = pd.DataFrame(metric_records)
    predictions_path = run_dir / "sample_predictions.parquet"
    metrics_path = run_dir / "policy_metrics.csv"
    result_path = run_dir / "smoke_result.json"
    audit_path = run_dir / "evidence_audit.json"
    predictions.to_parquet(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    result_path.write_text(
        json.dumps(
            {
                "status": "passed_end_to_end_public_demo_smoke",
                "scientific_result": False,
                "confirmation_claim_allowed": False,
                "models": sorted(predictions["experiment"].unique()),
                "target_encounters": int(target["sample_id"].nunique()),
                "profile_sha256": sha256_file(data_dir / "profile.json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "status": "complete_demo_smoke_evidence",
                "config_sha256": config_hash(config),
                "sha256": {
                    path.name: sha256_file(path)
                    for path in (predictions_path, metrics_path, result_path)
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "result": result_path,
        "audit": audit_path,
    }
