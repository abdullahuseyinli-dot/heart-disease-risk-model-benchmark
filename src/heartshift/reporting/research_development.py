"""Reconstruct the consumed-data backbone/control/router development benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from heartshift.config import config_hash
from heartshift.data.uci import sha256_file
from heartshift.metrics import balanced_log_loss
from heartshift.reporting.outer_report import (
    PRIMARY_TRACK,
    cell_metrics,
    paired_primary_stratified_bootstrap,
    primary_estimands,
)
from heartshift.reporting.sensitivity import leave_one_site_out_ranking

EVALUATION_KEYS = ("sample_id", "outer_target", "policy", "mask_replicate")


def _normalise_controls(run_dir: Path, seeds: tuple[int, ...]) -> pd.DataFrame:
    frame = pd.read_parquet(run_dir / "outer_predictions.parquet")
    frame = frame.loc[frame["training_seed"].isin(seeds)].copy()
    key = [*EVALUATION_KEYS, "control"]
    audit = frame.groupby(key, sort=False).agg(
        seed_count=("training_seed", "nunique"),
        target_values=("target", "nunique"),
        mask_values=("observed_mask_code", "nunique"),
        fraction_min=("observed_fraction", "min"),
        fraction_max=("observed_fraction", "max"),
    )
    if audit["seed_count"].ne(len(seeds)).any():
        raise AssertionError("A control outer key is missing a declared seed")
    if audit["target_values"].ne(1).any() or audit["mask_values"].ne(1).any():
        raise AssertionError("Control seeds disagree on endpoints or exact masks")
    if (audit["fraction_max"] - audit["fraction_min"]).max() > 1e-12:
        raise AssertionError("Control seeds disagree on observed fractions")
    grouped = frame.groupby(
        [
            *EVALUATION_KEYS,
            "target",
            "observed_fraction",
            "observed_mask_code",
            "control",
        ],
        as_index=False,
    ).agg(score=("y_score", "mean"))
    grouped["method"] = "control:" + grouped["control"].astype(str)
    grouped["track"] = PRIMARY_TRACK
    return grouped.loc[
        :,
        [
            *EVALUATION_KEYS,
            "target",
            "observed_fraction",
            "observed_mask_code",
            "method",
            "track",
            "score",
        ],
    ]


def _normalise_neural(
    run_dir: Path,
    logit_ensembles: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    frame = pd.read_parquet(run_dir / "outer_predictions.parquet").copy()
    frame["method"] = "neural:" + frame["experiment"].astype(str)
    frame["track"] = PRIMARY_TRACK
    frame["score"] = frame["y_score_zero_shot"].astype(float)
    columns = [
        *EVALUATION_KEYS,
        "target",
        "observed_fraction",
        "observed_mask_code",
        "method",
        "track",
        "score",
    ]
    records = [frame.loc[:, columns]]
    for ensemble_name, experiments in logit_ensembles.items():
        selected = frame.loc[frame["experiment"].isin(experiments)].copy()
        if set(selected["experiment"].astype(str)) != set(experiments):
            raise KeyError(f"Neural ensemble {ensemble_name} is missing an experiment")
        audit = selected.groupby(list(EVALUATION_KEYS), sort=False).agg(
            experiment_count=("experiment", "nunique"),
            target_values=("target", "nunique"),
            mask_values=("observed_mask_code", "nunique"),
            fraction_min=("observed_fraction", "min"),
            fraction_max=("observed_fraction", "max"),
        )
        if audit["experiment_count"].ne(len(experiments)).any():
            raise AssertionError(f"Neural ensemble {ensemble_name} is incomplete")
        if audit["target_values"].ne(1).any() or audit["mask_values"].ne(1).any():
            raise AssertionError(f"Neural ensemble {ensemble_name} disagrees on endpoints or masks")
        if (audit["fraction_max"] - audit["fraction_min"]).max() > 1e-12:
            raise AssertionError(f"Neural ensemble {ensemble_name} disagrees on observed fractions")
        selected["expert_logit"] = logit(
            np.clip(selected["score"].to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6)
        )
        ensemble = selected.groupby(
            [
                *EVALUATION_KEYS,
                "target",
                "observed_fraction",
                "observed_mask_code",
            ],
            as_index=False,
        ).agg(mean_logit=("expert_logit", "mean"))
        ensemble["method"] = f"neural_ensemble:{ensemble_name}"
        ensemble["track"] = PRIMARY_TRACK
        ensemble["score"] = expit(ensemble.pop("mean_logit"))
        records.append(ensemble.loc[:, columns])
    return pd.concat(records, ignore_index=True)


def _normalise_router(run_dir: Path, methods: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_parquet(run_dir / "outer_predictions.parquet")
    frame = frame.loc[frame["method"].isin(methods)].copy()
    if set(frame["method"].astype(str)) != set(methods):
        raise KeyError("A declared router-report method is missing")
    frame["method"] = "router:" + frame["method"].astype(str)
    frame["track"] = PRIMARY_TRACK
    frame["score"] = frame["y_score"].astype(float)
    return frame.loc[
        :,
        [
            *EVALUATION_KEYS,
            "target",
            "observed_fraction",
            "observed_mask_code",
            "method",
            "track",
            "score",
        ],
    ]


def _audit_paired_predictions(predictions: pd.DataFrame) -> None:
    if predictions.duplicated([*EVALUATION_KEYS, "method", "track"]).any():
        raise AssertionError("Development report contains duplicate method keys")
    if not np.isfinite(predictions["score"]).all():
        raise AssertionError("Development report contains non-finite probabilities")
    audit = predictions.groupby(list(EVALUATION_KEYS), sort=False).agg(
        method_count=("method", "nunique"),
        target_values=("target", "nunique"),
        mask_values=("observed_mask_code", "nunique"),
        fraction_min=("observed_fraction", "min"),
        fraction_max=("observed_fraction", "max"),
    )
    if audit["method_count"].ne(predictions["method"].nunique()).any():
        raise AssertionError("Development methods do not share complete paired keys")
    if audit["target_values"].ne(1).any() or audit["mask_values"].ne(1).any():
        raise AssertionError("Development methods disagree on endpoints or exact masks")
    if (audit["fraction_max"] - audit["fraction_min"]).max() > 1e-12:
        raise AssertionError("Development methods disagree on observed fractions")


def _seed_primary(
    frame: pd.DataFrame,
    *,
    namespace: str,
    method_column: str,
    seed_column: str,
    score_column: str,
) -> pd.DataFrame:
    records = []
    for (method, seed), group in frame.groupby([method_column, seed_column], sort=False):
        cells = []
        for (site, policy, replicate), cell in group.groupby(
            ["outer_target", "policy", "mask_replicate"], sort=False
        ):
            cells.append(
                {
                    "site": site,
                    "policy": policy,
                    "replicate": replicate,
                    "balanced_log_loss": balanced_log_loss(cell["target"], cell[score_column]),
                }
            )
        cell_frame = pd.DataFrame(cells)
        policy = cell_frame.groupby(["site", "policy"], as_index=False).agg(
            balanced_log_loss=("balanced_log_loss", "mean")
        )
        site_worst = policy.groupby("site")["balanced_log_loss"].max()
        records.append(
            {
                "method": f"{namespace}:{method}",
                "seed": int(seed),
                "macro_site_worst_balanced_log_loss": float(site_worst.mean()),
            }
        )
    return pd.DataFrame(records)


def _resource_summary(control_run: Path, neural_inner_run: Path) -> pd.DataFrame:
    controls = pd.read_csv(control_run / "fit_summaries.csv")
    control_summary = controls.groupby("control", as_index=False).agg(
        fits=("fit_seconds", "size"),
        mean_fit_seconds=("fit_seconds", "mean"),
        median_fit_seconds=("fit_seconds", "median"),
    )
    control_summary["family"] = "control"
    control_summary = control_summary.rename(columns={"control": "method"})
    neural = pd.read_csv(neural_inner_run / "fit_summaries.csv")
    neural_summary = neural.groupby("experiment", as_index=False).agg(
        fits=("fit_seconds", "size"),
        mean_fit_seconds=("fit_seconds", "mean"),
        median_fit_seconds=("fit_seconds", "median"),
        mean_parameter_count=("parameter_count", "mean"),
        median_best_epoch=("best_epoch", "median"),
    )
    neural_summary["family"] = "neural"
    neural_summary = neural_summary.rename(columns={"experiment": "method"})
    return pd.concat([control_summary, neural_summary], ignore_index=True, sort=False)


def build_research_development_report(
    repo_root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Build a hash-audited post-outcome report from individual predictions."""
    output_dir.mkdir(parents=True, exist_ok=False)
    control_run = repo_root / str(config["control_run"])
    neural_inner_run = repo_root / str(config["neural_inner_run"])
    neural_outer_run = repo_root / str(config["neural_outer_run"])
    router_run = repo_root / str(config["router_run"])
    seeds = tuple(int(value) for value in config["seeds"])
    predictions = pd.concat(
        [
            _normalise_controls(control_run, seeds),
            _normalise_neural(
                neural_outer_run,
                {
                    str(name): tuple(str(value) for value in experiments)
                    for name, experiments in config.get("neural_logit_ensembles", {}).items()
                },
            ),
            _normalise_router(
                router_run,
                tuple(str(value) for value in config["router_methods"]),
            ),
        ],
        ignore_index=True,
    )
    _audit_paired_predictions(predictions)
    metrics = cell_metrics(predictions)
    primary = primary_estimands(metrics)
    contrast_replicates = []
    contrast_intervals = []
    for contrast in config["contrasts"]:
        reference = str(contrast["reference_method"])
        comparisons = [str(value) for value in contrast["comparison_methods"]]
        replicates, intervals = paired_primary_stratified_bootstrap(
            predictions,
            reference_method=reference,
            comparison_methods=comparisons,
            repetitions=int(config["bootstrap_repetitions"]),
            seed=int(config["bootstrap_seed"]),
        )
        replicates["contrast_id"] = str(contrast["id"])
        intervals["contrast_id"] = str(contrast["id"])
        contrast_replicates.append(replicates)
        contrast_intervals.append(intervals)
    control_seeds = pd.read_parquet(control_run / "outer_predictions.parquet")
    neural_seeds = pd.read_parquet(neural_outer_run / "outer_seed_predictions.parquet")
    seed_sensitivity = pd.concat(
        [
            _seed_primary(
                control_seeds.loc[control_seeds["training_seed"].isin(seeds)],
                namespace="control",
                method_column="control",
                seed_column="training_seed",
                score_column="y_score",
            ),
            _seed_primary(
                neural_seeds.loc[neural_seeds["training_seed"].isin(seeds)],
                namespace="neural",
                method_column="experiment",
                seed_column="training_seed",
                score_column="y_score",
            ),
        ],
        ignore_index=True,
    )
    paths = {
        "predictions": output_dir / "normalized_predictions.parquet",
        "metrics": output_dir / "site_policy_metrics.csv",
        "primary": output_dir / "primary_estimands.csv",
        "contrasts": output_dir / "paired_contrast_intervals.csv",
        "contrast_replicates": output_dir / "paired_contrast_replicates.parquet",
        "site_jackknife": output_dir / "leave_one_site_out_ranking.csv",
        "seed_sensitivity": output_dir / "individual_seed_sensitivity.csv",
        "resources": output_dir / "resource_summary.csv",
        "neural_selections": output_dir / "source_selected_neural_experts.csv",
        "router_selections": output_dir / "source_selected_router_candidates.csv",
        "manifest": output_dir / "report_manifest.json",
        "audit": output_dir / "evidence_audit.json",
    }
    predictions.to_parquet(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    primary.to_csv(paths["primary"], index=False)
    pd.concat(contrast_intervals, ignore_index=True).to_csv(paths["contrasts"], index=False)
    pd.concat(contrast_replicates, ignore_index=True).to_parquet(
        paths["contrast_replicates"], index=False
    )
    leave_one_site_out_ranking(metrics).to_csv(paths["site_jackknife"], index=False)
    seed_sensitivity.to_csv(paths["seed_sensitivity"], index=False)
    _resource_summary(control_run, neural_inner_run).to_csv(paths["resources"], index=False)
    pd.read_csv(router_run / "source_selected_neural_experts.csv").to_csv(
        paths["neural_selections"], index=False
    )
    pd.read_csv(router_run / "source_selected_router_candidates.csv").to_csv(
        paths["router_selections"], index=False
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "status": str(config["status"]),
                "new_confirmatory_claim_allowed": False,
                "outer_outcomes_historically_consumed": True,
                "exact_cross_method_mask_identity_required": True,
                "individual_seed_predictions_retained": True,
                "inference": (
                    "conditional on four observed hospitals and fitted models; "
                    "post-outcome descriptive only"
                ),
                "config_sha256": config_hash(config),
                "method_count": int(predictions["method"].nunique()),
                "prediction_rows": len(predictions),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hashes = {
        path.name: sha256_file(path)
        for name, path in paths.items()
        if name != "audit" and path.is_file()
    }
    paths["audit"].write_text(
        json.dumps(
            {
                "status": "complete_prediction_reconstructed_development_report",
                "sha256": hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
