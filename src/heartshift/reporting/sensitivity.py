"""Versioned post-outcome sensitivity analyses for consumed heart evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heartshift.data.uci import sha256_file
from heartshift.metrics import balanced_log_loss
from heartshift.reporting.outer_report import (
    PRIMARY_TRACK,
    cell_metrics,
    paired_primary_stratified_bootstrap,
    primary_estimands,
)


def site_policy_class_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    """Expose the class component driving each balanced-loss cell."""
    selected = predictions.loc[predictions["track"].eq(PRIMARY_TRACK)]
    records = []
    grouping = ["method", "outer_target", "policy", "mask_replicate"]
    for keys, group in selected.groupby(grouping, sort=False):
        score = np.clip(group["score"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7)
        target = group["target"].to_numpy(dtype=np.int8)
        point = -(target * np.log(score) + (1 - target) * np.log(1 - score))
        record = dict(zip(grouping, keys, strict=True))
        for label in (0, 1):
            values = point[target == label]
            record[f"class_{label}_n"] = len(values)
            record[f"class_{label}_log_loss"] = float(values.mean()) if len(values) else np.nan
        record["balanced_log_loss"] = float(
            np.nanmean([record["class_0_log_loss"], record["class_1_log_loss"]])
        )
        records.append(record)
    return pd.DataFrame(records)


def site_worst_contrasts(
    metrics: pd.DataFrame,
    *,
    reference_method: str,
    comparison_methods: list[str],
) -> pd.DataFrame:
    policy = (
        metrics.loc[metrics["track"].eq(PRIMARY_TRACK)]
        .groupby(["method", "outer_target", "policy"], as_index=False)
        .agg(balanced_log_loss=("balanced_log_loss", "mean"))
    )
    worst = policy.groupby(["method", "outer_target"], as_index=False).agg(
        site_worst_balanced_log_loss=("balanced_log_loss", "max")
    )
    reference = worst.loc[worst["method"].eq(reference_method)].rename(
        columns={"site_worst_balanced_log_loss": "reference_site_worst_balanced_log_loss"}
    )
    records = []
    for method in comparison_methods:
        candidate = worst.loc[worst["method"].eq(method)].rename(
            columns={"site_worst_balanced_log_loss": "candidate_site_worst_balanced_log_loss"}
        )
        merged = candidate.merge(
            reference.loc[:, ["outer_target", "reference_site_worst_balanced_log_loss"]],
            on="outer_target",
            validate="one_to_one",
        )
        merged["reference_method"] = reference_method
        merged["difference_candidate_minus_reference"] = (
            merged["candidate_site_worst_balanced_log_loss"]
            - merged["reference_site_worst_balanced_log_loss"]
        )
        records.append(merged)
    return pd.concat(records, ignore_index=True)


def leave_one_site_out_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    policy = (
        metrics.loc[metrics["track"].eq(PRIMARY_TRACK)]
        .groupby(["method", "outer_target", "policy"], as_index=False)
        .agg(balanced_log_loss=("balanced_log_loss", "mean"))
    )
    worst = policy.groupby(["method", "outer_target"], as_index=False).agg(
        site_worst_balanced_log_loss=("balanced_log_loss", "max")
    )
    records = []
    for omitted in sorted(worst["outer_target"].unique()):
        ranking = (
            worst.loc[worst["outer_target"].ne(omitted)]
            .groupby("method", as_index=False)
            .agg(macro_site_worst_balanced_log_loss=("site_worst_balanced_log_loss", "mean"))
            .sort_values(["macro_site_worst_balanced_log_loss", "method"])
            .reset_index(drop=True)
        )
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
        ranking.insert(0, "omitted_site", omitted)
        records.append(ranking)
    return pd.concat(records, ignore_index=True)


def _normalise_seed_predictions(path: Path, kind: str, namespace: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if kind == "neural":
        frame["method"] = namespace + ":" + frame["experiment"].astype(str)
    elif kind == "classical":
        base = namespace + ":" + frame["model"].astype(str) + ":" + frame["weighting"].astype(str)
        if "calibration" in frame:
            frame["method"] = np.where(
                frame["calibration"].astype(str).eq("raw"),
                base,
                base + ":" + frame["calibration"].astype(str),
            )
        else:
            frame["method"] = base
    else:
        raise KeyError(f"Unknown seed prediction kind: {kind}")
    score_column = "y_score" if "y_score" in frame else "y_score_zero_shot"
    return frame.rename(columns={"training_seed": "seed", score_column: "score"}).loc[
        :,
        [
            "sample_id",
            "outer_target",
            "target",
            "policy",
            "mask_replicate",
            "method",
            "seed",
            "score",
        ],
    ]


def seed_level_primary_estimands(
    repo_root: Path,
    specs: list[dict[str, Any]],
    ensemble_primary: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        _normalise_seed_predictions(
            repo_root / str(spec["path"]), str(spec["kind"]), str(spec["namespace"])
        )
        for spec in specs
    ]
    seed_predictions = pd.concat(frames, ignore_index=True)
    records = []
    for (method, seed), group in seed_predictions.groupby(["method", "seed"], sort=False):
        cells = []
        for (site, policy, replicate), cell in group.groupby(
            ["outer_target", "policy", "mask_replicate"], sort=False
        ):
            cells.append(
                {
                    "outer_target": site,
                    "policy": policy,
                    "mask_replicate": replicate,
                    "balanced_log_loss": balanced_log_loss(cell["target"], cell["score"]),
                }
            )
        cell_frame = pd.DataFrame(cells)
        policy_mean = cell_frame.groupby(["outer_target", "policy"], as_index=False).agg(
            balanced_log_loss=("balanced_log_loss", "mean")
        )
        site_worst = policy_mean.groupby("outer_target")["balanced_log_loss"].max()
        records.append(
            {
                "method": method,
                "seed": int(seed),
                "seed_macro_site_worst_balanced_log_loss": float(site_worst.mean()),
            }
        )
    result = pd.DataFrame(records)
    summary = result.groupby("method", as_index=False).agg(
        individual_seed_mean=("seed_macro_site_worst_balanced_log_loss", "mean"),
        individual_seed_sd=("seed_macro_site_worst_balanced_log_loss", "std"),
        individual_seed_min=("seed_macro_site_worst_balanced_log_loss", "min"),
        individual_seed_max=("seed_macro_site_worst_balanced_log_loss", "max"),
        seed_count=("seed", "nunique"),
    )
    ensemble = ensemble_primary.loc[
        :, ["method", "macro_site_worst_mask_balanced_log_loss"]
    ].rename(
        columns={"macro_site_worst_mask_balanced_log_loss": "ensemble_primary_balanced_log_loss"}
    )
    return summary.merge(ensemble, on="method", how="left", validate="one_to_one")


def build_heart_sensitivity_report(
    repo_root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Build a non-confirmatory correction package without changing v5 evidence."""
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions = pd.read_parquet(repo_root / str(config["normalized_predictions_path"]))
    metrics = cell_metrics(predictions)
    primary = primary_estimands(metrics)
    reference = str(config["reference_method"])
    comparisons = [str(value) for value in config["comparison_methods"]]
    bootstrap_replicates, bootstrap_intervals = paired_primary_stratified_bootstrap(
        predictions,
        reference_method=reference,
        comparison_methods=comparisons,
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    outputs = {
        "bootstrap_replicates": output_dir / "stratified_bootstrap_replicates.parquet",
        "bootstrap_intervals": output_dir / "stratified_bootstrap_intervals.csv",
        "site_contrasts": output_dir / "site_worst_contrasts.csv",
        "site_jackknife": output_dir / "leave_one_site_out_ranking.csv",
        "class_losses": output_dir / "site_policy_class_losses.csv",
        "seed_sensitivity": output_dir / "seed_ensemble_sensitivity.csv",
        "manifest": output_dir / "report_manifest.json",
        "audit": output_dir / "evidence_audit.json",
    }
    bootstrap_replicates.to_parquet(outputs["bootstrap_replicates"], index=False)
    bootstrap_intervals.to_csv(outputs["bootstrap_intervals"], index=False)
    site_worst_contrasts(
        metrics, reference_method=reference, comparison_methods=comparisons
    ).to_csv(outputs["site_contrasts"], index=False)
    leave_one_site_out_ranking(metrics).to_csv(outputs["site_jackknife"], index=False)
    site_policy_class_losses(predictions).to_csv(outputs["class_losses"], index=False)
    seed_level_primary_estimands(repo_root, config["seed_prediction_runs"], primary).to_csv(
        outputs["seed_sensitivity"], index=False
    )
    outputs["manifest"].write_text(
        json.dumps(
            {
                "status": "post_outcome_consumed_evidence_sensitivity_only",
                "source_report": str(config["normalized_predictions_path"]),
                "bootstrap_unit": "record_stratified_by_observed_hospital_and_outcome_class",
                "hospital_inference": "conditional_on_four_observed_hospitals",
                "fitted_model_uncertainty_in_bootstrap": False,
                "future_hospital_inference": False,
                "exact_cross_method_mask_identity_available_in_historical_artifacts": False,
                "effect_estimate": "original_observed_paired_contrast",
                "bootstrap_repetitions": int(config["bootstrap"]["repetitions"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hashes = {
        path.name: sha256_file(path)
        for name, path in outputs.items()
        if name != "audit" and path.is_file()
    }
    outputs["audit"].write_text(
        json.dumps(
            {
                "audit_status": "complete_consumed_sensitivity_evidence",
                "artifact_count": len(hashes),
                "sha256": hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs
