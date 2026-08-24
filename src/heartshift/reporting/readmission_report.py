"""Prediction-level reporting for the patient-disjoint readmission shift task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heartshift.metrics import binary_metrics
from heartshift.reporting.outer_report import _weighted_balanced_log_loss


def readmission_cell_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
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
    records = []
    for keys, group in predictions.groupby(grouping, sort=False, dropna=False):
        records.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                **binary_metrics(group["target"], group["y_score"]),
            }
        )
    return pd.DataFrame(records)


def readmission_primary_estimands(metrics: pd.DataFrame) -> pd.DataFrame:
    policy_means = metrics.groupby(["experiment", "split", "policy"], as_index=False).agg(
        balanced_log_loss=("balanced_log_loss", "mean"),
        balanced_brier=("balanced_brier", "mean"),
        roc_auc=("roc_auc", "mean"),
    )
    records = []
    for experiment, values in policy_means.groupby("experiment", sort=False):
        id_natural = values.loc[
            values["split"].eq("id_test") & values["policy"].eq("natural")
        ].iloc[0]
        ood_natural = values.loc[
            values["split"].eq("ood_test") & values["policy"].eq("natural")
        ].iloc[0]
        ood = values.loc[values["split"].eq("ood_test")]
        records.append(
            {
                "experiment": experiment,
                "id_natural_balanced_log_loss": float(id_natural["balanced_log_loss"]),
                "ood_natural_balanced_log_loss": float(ood_natural["balanced_log_loss"]),
                "ood_minus_id_natural_balanced_log_loss": float(
                    ood_natural["balanced_log_loss"] - id_natural["balanced_log_loss"]
                ),
                "ood_worst_mask_balanced_log_loss": float(ood["balanced_log_loss"].max()),
                "ood_worst_mask_balanced_brier": float(ood["balanced_brier"].max()),
                "id_natural_roc_auc": float(id_natural["roc_auc"]),
                "ood_natural_roc_auc": float(ood_natural["roc_auc"]),
            }
        )
    return pd.DataFrame(records).sort_values("ood_worst_mask_balanced_log_loss")


def paired_readmission_bootstrap(
    predictions: pd.DataFrame,
    *,
    reference_experiment: str,
    comparison_experiments: list[str],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster-bootstrap patients for the OOD worst-mask estimand."""
    methods = [reference_experiment, *comparison_experiments]
    selected = predictions.loc[
        predictions["experiment"].isin(methods) & predictions["split"].eq("ood_test")
    ].copy()
    if set(selected["experiment"]) != set(methods):
        missing = set(methods) - set(selected["experiment"])
        raise KeyError(f"Readmission bootstrap experiments missing: {sorted(missing)}")
    key_columns = ["sample_id", "patient_nbr", "policy", "mask_replicate"]
    reference_keys = selected.loc[
        selected["experiment"].eq(reference_experiment), key_columns
    ].sort_values(key_columns)
    for experiment in comparison_experiments:
        keys = selected.loc[selected["experiment"].eq(experiment), key_columns].sort_values(
            key_columns
        )
        if not reference_keys.reset_index(drop=True).equals(keys.reset_index(drop=True)):
            raise AssertionError(f"Readmission prediction keys differ for experiment {experiment}")

    patient_ids = (
        selected.loc[selected["experiment"].eq(reference_experiment), "patient_nbr"]
        .drop_duplicates()
        .to_numpy()
    )
    cells = {}
    for experiment, method_data in selected.groupby("experiment", sort=False):
        method_cells = []
        for (policy, replicate), group in method_data.groupby(
            ["policy", "mask_replicate"], sort=False
        ):
            method_cells.append(
                (
                    str(policy),
                    int(replicate),
                    group["patient_nbr"].to_numpy(),
                    group["target"].to_numpy(dtype=np.int8),
                    group["y_score"].to_numpy(dtype=np.float64),
                )
            )
        cells[str(experiment)] = method_cells

    rng = np.random.default_rng(seed)
    records = []
    for replicate_index in range(repetitions):
        draws = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        unique, draw_counts = np.unique(draws, return_counts=True)
        patient_counts = dict(zip(unique, draw_counts, strict=True))
        estimates = {}
        for experiment, method_cells in cells.items():
            cell_losses = []
            for policy, replicate, patients, target, score in method_cells:
                counts = np.fromiter(
                    (patient_counts.get(value, 0) for value in patients),
                    dtype=np.float64,
                    count=len(patients),
                )
                cell_losses.append(
                    (
                        policy,
                        replicate,
                        _weighted_balanced_log_loss(target, score, counts),
                    )
                )
            cell_frame = pd.DataFrame(
                cell_losses, columns=["policy", "replicate", "balanced_log_loss"]
            )
            estimates[experiment] = float(
                cell_frame.groupby("policy")["balanced_log_loss"].mean().max()
            )
        for experiment in comparison_experiments:
            records.append(
                {
                    "bootstrap_replicate": replicate_index,
                    "experiment": experiment,
                    "reference_experiment": reference_experiment,
                    "difference_candidate_minus_reference": (
                        estimates[experiment] - estimates[reference_experiment]
                    ),
                }
            )
    replicates = pd.DataFrame(records)
    intervals = replicates.groupby(["experiment", "reference_experiment"], as_index=False).agg(
        mean_difference=("difference_candidate_minus_reference", "mean"),
        ci_025=("difference_candidate_minus_reference", lambda values: values.quantile(0.025)),
        ci_975=("difference_candidate_minus_reference", lambda values: values.quantile(0.975)),
        probability_better=(
            "difference_candidate_minus_reference",
            lambda values: float(np.mean(values < 0.0)),
        ),
    )
    return replicates, intervals


def build_readmission_report(
    repo_root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions = pd.read_parquet(repo_root / config["predictions_path"])
    required = {
        "sample_id",
        "patient_nbr",
        "target",
        "experiment",
        "split",
        "policy",
        "mask_replicate",
        "y_score",
    }
    if missing := required - set(predictions.columns):
        raise AssertionError(f"Readmission prediction columns missing: {sorted(missing)}")
    metrics = readmission_cell_metrics(predictions)
    primary = readmission_primary_estimands(metrics)
    replicates, intervals = paired_readmission_bootstrap(
        predictions,
        reference_experiment=str(config["bootstrap"]["reference_experiment"]),
        comparison_experiments=[
            str(value) for value in config["bootstrap"]["comparison_experiments"]
        ],
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    paths = {
        "cell_metrics": output_dir / "split_policy_metrics.csv",
        "primary": output_dir / "primary_estimands.csv",
        "bootstrap_replicates": output_dir / "patient_cluster_bootstrap.parquet",
        "bootstrap_intervals": output_dir / "patient_cluster_bootstrap_intervals.csv",
        "manifest": output_dir / "report_manifest.json",
    }
    metrics.to_csv(paths["cell_metrics"], index=False)
    primary.to_csv(paths["primary"], index=False)
    replicates.to_parquet(paths["bootstrap_replicates"], index=False)
    intervals.to_csv(paths["bootstrap_intervals"], index=False)
    paths["manifest"].write_text(
        json.dumps(
            {
                "status": "derived_from_locked_sample_level_predictions",
                "bootstrap_unit": "patient_cluster",
                "endpoint": "any_readmission_TableShift_compatibility",
                "prediction_rows": len(predictions),
                "experiments": sorted(predictions["experiment"].unique()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
