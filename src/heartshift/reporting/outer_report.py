"""Aggregate locked HeartShift outputs from immutable patient-level predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heartshift.metrics import binary_metrics

PRIMARY_TRACK = "dg_zero_shot"
MCAR_RATES = {"natural": 0.0, "mcar_10": 0.1, "mcar_30": 0.3, "mcar_50": 0.5}


def _normalise_classical(path: Path, namespace: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    base_method = (
        namespace + ":" + frame["model"].astype(str) + ":" + frame["weighting"].astype(str)
    )
    calibration = (
        frame["calibration"].astype(str)
        if "calibration" in frame
        else pd.Series("raw", index=frame.index, dtype="string")
    )
    frame["method"] = np.where(
        calibration.eq("raw"),
        base_method,
        base_method + ":" + calibration,
    )
    frame["track"] = PRIMARY_TRACK
    frame["score"] = frame["y_score"].astype(float)
    columns = [
        "sample_id",
        "outer_target",
        "target",
        "policy",
        "mask_replicate",
        "observed_fraction",
        "method",
        "track",
        "score",
    ]
    for identity_column in ("observed_mask_code", "observed_mask_sha256"):
        if identity_column in frame:
            columns.append(identity_column)
    return frame.loc[:, columns]


def _normalise_neural(path: Path, namespace: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["method"] = namespace + ":" + frame["experiment"].astype(str)
    tracks: list[tuple[str, str, str | None]] = [(PRIMARY_TRACK, "y_score_zero_shot", None)]
    if "y_score_uda_mlls" in frame:
        tracks.extend(
            [
                ("uda_mlls", "y_score_uda_mlls", "adaptation_allowed"),
                ("uda_soft_bbse", "y_score_uda_soft_bbse", "adaptation_allowed"),
            ]
        )
    records = []
    for track, score_column, gate_column in tracks:
        selected = frame
        if gate_column is not None:
            selected = selected.loc[selected[gate_column].astype(bool)]
        selected = selected.loc[selected[score_column].notna()].copy()
        selected["track"] = track
        selected["score"] = selected[score_column].astype(float)
        columns = [
            "sample_id",
            "outer_target",
            "target",
            "policy",
            "mask_replicate",
            "observed_fraction",
            "method",
            "track",
            "score",
        ]
        for identity_column in ("observed_mask_code", "observed_mask_sha256"):
            if identity_column in selected:
                columns.append(identity_column)
        records.append(selected.loc[:, columns])
    return pd.concat(records, ignore_index=True)


def load_heart_outer_predictions(
    repo_root: Path,
    run_specs: list[dict[str, Any]],
    *,
    require_exact_mask_identity: bool = False,
) -> pd.DataFrame:
    records = []
    for spec in run_specs:
        path = repo_root / str(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Locked prediction artifact is missing: {path}")
        kind = str(spec["kind"])
        namespace = str(spec["namespace"])
        if kind == "classical":
            records.append(_normalise_classical(path, namespace))
        elif kind == "neural":
            records.append(_normalise_neural(path, namespace))
        else:
            raise KeyError(f"Unknown outer prediction kind: {kind}")
    predictions = pd.concat(records, ignore_index=True)
    if predictions.duplicated(
        [
            "sample_id",
            "outer_target",
            "policy",
            "mask_replicate",
            "method",
            "track",
        ]
    ).any():
        raise AssertionError("Normalized outer predictions contain duplicate evaluation rows")
    if not np.isfinite(predictions["score"]).all():
        raise AssertionError("Normalized outer predictions contain non-finite scores")
    paired = predictions.loc[predictions["track"].eq(PRIMARY_TRACK)]
    audit = paired.groupby(
        ["sample_id", "outer_target", "policy", "mask_replicate"], sort=False
    ).agg(
        target_values=("target", "nunique"),
        minimum_observed_fraction=("observed_fraction", "min"),
        maximum_observed_fraction=("observed_fraction", "max"),
    )
    if audit["target_values"].ne(1).any():
        raise AssertionError("Methods disagree on an outer target endpoint")
    if (audit["maximum_observed_fraction"] - audit["minimum_observed_fraction"]).max() > 1e-12:
        raise AssertionError("Methods were not evaluated under identical patient masks")
    identity_columns = [
        column
        for column in ("observed_mask_code", "observed_mask_sha256")
        if column in paired and paired[column].notna().all()
    ]
    if require_exact_mask_identity and not identity_columns:
        raise AssertionError(
            "Exact mask identity is required but at least one prediction artifact lacks it"
        )
    for identity_column in identity_columns:
        identity_audit = paired.groupby(
            ["sample_id", "outer_target", "policy", "mask_replicate"], sort=False
        )[identity_column].nunique(dropna=False)
        if identity_audit.gt(1).any():
            raise AssertionError(f"Methods disagree on exact patient masks in {identity_column}")
    return predictions


def cell_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    grouping = ["method", "track", "outer_target", "policy", "mask_replicate"]
    records = []
    for keys, group in predictions.groupby(grouping, sort=False):
        records.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                "mean_observed_fraction": float(group["observed_fraction"].mean()),
                **binary_metrics(group["target"], group["score"]),
            }
        )
    return pd.DataFrame(records)


def primary_estimands(metrics: pd.DataFrame) -> pd.DataFrame:
    zero_shot = metrics.loc[metrics["track"].eq(PRIMARY_TRACK)].copy()
    policy_means = (
        zero_shot.groupby(["method", "outer_target", "policy"], as_index=False)
        .agg(
            balanced_log_loss=("balanced_log_loss", "mean"),
            balanced_brier=("balanced_brier", "mean"),
            roc_auc=("roc_auc", "mean"),
            aurc=("aurc", "mean"),
        )
        .sort_values(["method", "outer_target", "policy"])
    )
    records = []
    for method, method_metrics in policy_means.groupby("method", sort=False):
        site_worst = method_metrics.groupby("outer_target", as_index=False).agg(
            worst_mask_balanced_log_loss=("balanced_log_loss", "max"),
            worst_mask_balanced_brier=("balanced_brier", "max"),
            worst_mask_aurc=("aurc", "max"),
        )
        natural = method_metrics.loc[method_metrics["policy"].eq("natural")]
        mcar = method_metrics.loc[method_metrics["policy"].isin(MCAR_RATES)].copy()
        mcar["deletion_rate"] = mcar["policy"].map(MCAR_RATES).astype(float)
        curve_areas = []
        degradation = []
        for _, curve in mcar.groupby("outer_target"):
            curve = curve.sort_values("deletion_rate")
            if set(curve["deletion_rate"]) == set(MCAR_RATES.values()):
                curve_areas.append(
                    float(
                        np.trapezoid(curve["balanced_log_loss"], curve["deletion_rate"])
                        / max(MCAR_RATES.values())
                    )
                )
                degradation.append(
                    float(
                        curve.loc[curve["deletion_rate"].eq(0.5), "balanced_log_loss"].iloc[0]
                        - curve.loc[curve["deletion_rate"].eq(0.0), "balanced_log_loss"].iloc[0]
                    )
                )
        records.append(
            {
                "method": method,
                "macro_site_worst_mask_balanced_log_loss": float(
                    site_worst["worst_mask_balanced_log_loss"].mean()
                ),
                "worst_site_mask_balanced_log_loss": float(
                    method_metrics["balanced_log_loss"].max()
                ),
                "macro_site_worst_mask_balanced_brier": float(
                    site_worst["worst_mask_balanced_brier"].mean()
                ),
                "macro_site_worst_mask_aurc": float(site_worst["worst_mask_aurc"].mean()),
                "macro_natural_roc_auc": float(natural["roc_auc"].mean()),
                "macro_mcar_balanced_log_loss_auc": float(np.mean(curve_areas)),
                "macro_mcar50_minus_natural_balanced_log_loss": float(np.mean(degradation)),
            }
        )
    return pd.DataFrame(records).sort_values(
        [
            "macro_site_worst_mask_balanced_log_loss",
            "worst_site_mask_balanced_log_loss",
        ]
    )


def adaptation_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    adapted = metrics.loc[metrics["track"].ne(PRIMARY_TRACK)].copy()
    if adapted.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "track",
                "accepted_site_policy_cells",
                "mean_log_loss",
                "mean_balanced_log_loss",
                "mean_roc_auc",
            ]
        )
    policy_means = adapted.groupby(
        ["method", "track", "outer_target", "policy"], as_index=False
    ).agg(
        log_loss=("log_loss", "mean"),
        balanced_log_loss=("balanced_log_loss", "mean"),
        roc_auc=("roc_auc", "mean"),
    )
    return policy_means.groupby(["method", "track"], as_index=False).agg(
        accepted_site_policy_cells=("policy", "size"),
        mean_log_loss=("log_loss", "mean"),
        mean_balanced_log_loss=("balanced_log_loss", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
    )


def descriptive_sex_subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report natural-policy strata without presenting them as a fairness guarantee."""
    if "sex" not in predictions:
        raise KeyError("Sex subgroup reporting requires a sex column")
    selected = predictions.loc[
        predictions["track"].eq(PRIMARY_TRACK)
        & predictions["policy"].eq("natural")
        & predictions["sex"].notna()
    ]
    records = []
    for keys, group in selected.groupby(["method", "outer_target", "sex"], sort=False):
        if group["target"].nunique() == 2:
            metrics = binary_metrics(group["target"], group["score"])
            estimable = True
        else:
            metrics = {
                "n": float(len(group)),
                "prevalence": float(group["target"].mean()),
            }
            estimable = False
        records.append(
            {
                "method": keys[0],
                "outer_target": keys[1],
                "sex": keys[2],
                "both_classes_observed": estimable,
                **metrics,
            }
        )
    return pd.DataFrame(records)


def _weighted_balanced_log_loss(
    target: np.ndarray,
    score: np.ndarray,
    count: np.ndarray,
) -> float:
    score = np.clip(score, 1e-7, 1 - 1e-7)
    point = -(target * np.log(score) + (1 - target) * np.log(1 - score))
    class_means = []
    for label in (0, 1):
        selected = target == label
        denominator = float(count[selected].sum())
        if denominator <= 0:
            return float("nan")
        class_means.append(float(np.sum(count[selected] * point[selected]) / denominator))
    return float(np.mean(class_means))


def paired_primary_bootstrap(
    predictions: pd.DataFrame,
    *,
    reference_method: str,
    comparison_methods: list[str],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired record bootstrap conditional on the four observed hospitals."""
    zero_shot = predictions.loc[predictions["track"].eq(PRIMARY_TRACK)].copy()
    methods = [reference_method, *comparison_methods]
    selected = zero_shot.loc[zero_shot["method"].isin(methods)]
    if set(selected["method"]) != set(methods):
        missing = set(methods) - set(selected["method"])
        raise KeyError(f"Bootstrap methods missing from predictions: {sorted(missing)}")
    key_columns = ["sample_id", "outer_target", "policy", "mask_replicate"]
    reference_keys = selected.loc[selected["method"].eq(reference_method), key_columns].sort_values(
        key_columns
    )
    for method in comparison_methods:
        method_keys = selected.loc[selected["method"].eq(method), key_columns].sort_values(
            key_columns
        )
        if not reference_keys.reset_index(drop=True).equals(method_keys.reset_index(drop=True)):
            raise AssertionError(f"Paired bootstrap keys differ for method {method}")

    cells: dict[str, list[tuple[str, str, int, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for method, method_data in selected.groupby("method", sort=False):
        method_cells = []
        for (site, policy, replicate), group in method_data.groupby(
            ["outer_target", "policy", "mask_replicate"], sort=False
        ):
            method_cells.append(
                (
                    str(site),
                    str(policy),
                    int(replicate),
                    group["sample_id"].to_numpy(),
                    group["target"].to_numpy(dtype=np.int8),
                    group["score"].to_numpy(dtype=np.float64),
                )
            )
        cells[str(method)] = method_cells

    site_samples = {
        str(site): values["sample_id"].drop_duplicates().to_numpy()
        for site, values in selected.loc[selected["method"].eq(reference_method)].groupby(
            "outer_target"
        )
    }
    rng = np.random.default_rng(seed)
    replicate_records = []
    for bootstrap_replicate in range(repetitions):
        counts_by_site = {}
        for site, sample_ids in site_samples.items():
            draws = rng.choice(sample_ids, size=len(sample_ids), replace=True)
            unique, counts = np.unique(draws, return_counts=True)
            counts_by_site[site] = dict(zip(unique, counts, strict=True))
        estimates = {}
        for method, method_cells in cells.items():
            cell_losses = []
            for site, policy, replicate, sample_ids, target, score in method_cells:
                sample_counts = counts_by_site[site]
                count = np.fromiter(
                    (sample_counts.get(value, 0) for value in sample_ids),
                    dtype=np.float64,
                    count=len(sample_ids),
                )
                loss = _weighted_balanced_log_loss(target, score, count)
                cell_losses.append((site, policy, replicate, loss))
            cell_frame = pd.DataFrame(
                cell_losses,
                columns=["site", "policy", "replicate", "balanced_log_loss"],
            )
            policy_means = cell_frame.groupby(["site", "policy"], as_index=False).agg(
                balanced_log_loss=("balanced_log_loss", "mean")
            )
            site_worst = policy_means.groupby("site")["balanced_log_loss"].max()
            estimates[method] = float(site_worst.mean())
        for method in comparison_methods:
            replicate_records.append(
                {
                    "bootstrap_replicate": bootstrap_replicate,
                    "method": method,
                    "reference_method": reference_method,
                    "difference_candidate_minus_reference": (
                        estimates[method] - estimates[reference_method]
                    ),
                }
            )
    replicates = pd.DataFrame(replicate_records)
    intervals = (
        replicates.groupby(["method", "reference_method"], as_index=False)
        .agg(
            mean_difference=("difference_candidate_minus_reference", "mean"),
            ci_025=("difference_candidate_minus_reference", lambda values: values.quantile(0.025)),
            ci_975=("difference_candidate_minus_reference", lambda values: values.quantile(0.975)),
            probability_better=(
                "difference_candidate_minus_reference",
                lambda values: float(np.mean(values < 0.0)),
            ),
        )
        .sort_values("mean_difference")
    )
    return replicates, intervals


def paired_primary_stratified_bootstrap(
    predictions: pd.DataFrame,
    *,
    reference_method: str,
    comparison_methods: list[str],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired site-by-class record bootstrap with explicit observed contrasts.

    This version never drops a hospital because a rare class disappeared from a
    resample. It remains conditional on the observed hospitals and fitted models;
    it is not a future-hospital or refit bootstrap.
    """
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    zero_shot = predictions.loc[predictions["track"].eq(PRIMARY_TRACK)].copy()
    methods = [reference_method, *comparison_methods]
    selected = zero_shot.loc[zero_shot["method"].isin(methods)]
    if set(selected["method"]) != set(methods):
        missing = set(methods) - set(selected["method"])
        raise KeyError(f"Bootstrap methods missing from predictions: {sorted(missing)}")

    key_columns = ["sample_id", "outer_target", "policy", "mask_replicate"]
    reference_keys = selected.loc[selected["method"].eq(reference_method), key_columns].sort_values(
        key_columns
    )
    for method in comparison_methods:
        method_keys = selected.loc[selected["method"].eq(method), key_columns].sort_values(
            key_columns
        )
        if not reference_keys.reset_index(drop=True).equals(method_keys.reset_index(drop=True)):
            raise AssertionError(f"Paired bootstrap keys differ for method {method}")

    endpoint_audit = selected.groupby(["outer_target", "sample_id"], sort=False)["target"].nunique()
    if endpoint_audit.ne(1).any():
        raise AssertionError("A record has inconsistent endpoint values")
    reference_rows = selected.loc[selected["method"].eq(reference_method)]
    site_class_samples: dict[tuple[str, int], np.ndarray] = {}
    for (site, label), group in reference_rows.groupby(["outer_target", "target"], sort=False):
        sample_ids = group["sample_id"].drop_duplicates().to_numpy()
        if not len(sample_ids):
            raise ValueError(f"Site {site} has no records for outcome class {label}")
        site_class_samples[(str(site), int(label))] = sample_ids
    sites = {str(value) for value in reference_rows["outer_target"].unique()}
    if set(site for site, _ in site_class_samples) != sites or any(
        (site, label) not in site_class_samples for site in sites for label in (0, 1)
    ):
        raise ValueError("Every bootstrapped site must contain both outcome classes")

    observed = primary_estimands(cell_metrics(selected)).set_index("method")[
        "macro_site_worst_mask_balanced_log_loss"
    ]
    observed_differences = {
        method: float(observed[method] - observed[reference_method])
        for method in comparison_methods
    }
    # All methods share the same ordered evaluation rows.  Holding their losses in
    # columns keeps resampling paired, while a multinomial draw is exactly the
    # count-vector representation of sampling records with replacement.  This is
    # orders of magnitude faster than constructing dictionaries/data frames in
    # every replicate and does not alter the bootstrap distribution or estimand.
    method_order = [reference_method, *comparison_methods]
    ordered_keys = [*key_columns, "target"]
    reference = (
        selected.loc[selected["method"].eq(reference_method), ordered_keys]
        .sort_values(key_columns)
        .reset_index(drop=True)
    )
    scores = []
    for method in method_order:
        method_rows = (
            selected.loc[selected["method"].eq(method), [*key_columns, "score"]]
            .sort_values(key_columns)
            .reset_index(drop=True)
        )
        if not reference.loc[:, key_columns].equals(method_rows.loc[:, key_columns]):
            raise AssertionError(f"Paired bootstrap row order differs for method {method}")
        scores.append(method_rows["score"].to_numpy(dtype=np.float64))
    score_matrix = np.column_stack(scores)
    target = reference["target"].to_numpy(dtype=np.int8)
    clipped = np.clip(score_matrix, 1e-7, 1 - 1e-7)
    point_losses = -(
        target[:, None] * np.log(clipped) + (1 - target[:, None]) * np.log(1 - clipped)
    )

    rng = np.random.default_rng(seed)
    stratum_counts: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, raw_sample_ids in sorted(site_class_samples.items()):
        sample_ids = np.sort(raw_sample_ids)
        n_records = len(sample_ids)
        probabilities = np.full(n_records, 1.0 / n_records)
        counts = rng.multinomial(n_records, probabilities, size=repetitions).astype(
            np.float64, copy=False
        )
        stratum_counts[key] = (sample_ids, counts)

    site_estimates = []
    for site in sorted(sites):
        site_rows = reference["outer_target"].astype(str).eq(site).to_numpy()
        policy_estimates = []
        policies = sorted(reference.loc[site_rows, "policy"].astype(str).unique())
        for policy in policies:
            policy_rows = site_rows & reference["policy"].astype(str).eq(policy).to_numpy()
            replicate_estimates = []
            replicate_values = sorted(reference.loc[policy_rows, "mask_replicate"].unique())
            for replicate in replicate_values:
                cell_rows = policy_rows & reference["mask_replicate"].eq(replicate).to_numpy()
                class_estimates = []
                for label in (0, 1):
                    row_indices = np.flatnonzero(cell_rows & (target == label))
                    order = np.argsort(reference.loc[row_indices, "sample_id"].to_numpy())
                    row_indices = row_indices[order]
                    expected_ids, counts = stratum_counts[(site, label)]
                    actual_ids = reference.loc[row_indices, "sample_id"].to_numpy()
                    if not np.array_equal(actual_ids, expected_ids):
                        raise AssertionError(
                            f"Evaluation cell {site}/{policy}/{replicate}/{label} has "
                            "incomplete or duplicated patient keys"
                        )
                    class_estimates.append(counts @ point_losses[row_indices] / len(row_indices))
                replicate_estimates.append(0.5 * (class_estimates[0] + class_estimates[1]))
            policy_estimates.append(np.mean(replicate_estimates, axis=0))
        site_estimates.append(np.max(policy_estimates, axis=0))
    estimates = np.mean(site_estimates, axis=0)
    if estimates.shape != (repetitions, len(method_order)) or not np.isfinite(estimates).all():
        raise AssertionError("Vectorized stratified bootstrap produced invalid estimates")

    differences = estimates[:, 1:] - estimates[:, [0]]
    replicates = pd.DataFrame(
        {
            "bootstrap_replicate": np.repeat(np.arange(repetitions), len(comparison_methods)),
            "method": np.tile(comparison_methods, repetitions),
            "reference_method": reference_method,
            "difference_candidate_minus_reference": differences.reshape(-1),
        }
    )
    interval_records = []
    for (method, reference), group in replicates.groupby(
        ["method", "reference_method"], sort=False
    ):
        values = group["difference_candidate_minus_reference"]
        observed_difference = observed_differences[str(method)]
        percentile_low = float(values.quantile(0.025))
        percentile_high = float(values.quantile(0.975))
        bootstrap_mean = float(values.mean())
        interval_records.append(
            {
                "method": method,
                "reference_method": reference,
                "observed_difference": observed_difference,
                "bootstrap_mean_difference": bootstrap_mean,
                "bootstrap_bias": bootstrap_mean - observed_difference,
                "percentile_ci_025": percentile_low,
                "percentile_ci_975": percentile_high,
                "basic_ci_025": 2.0 * observed_difference - percentile_high,
                "basic_ci_975": 2.0 * observed_difference - percentile_low,
                "probability_better_descriptive": float(np.mean(values < 0.0)),
            }
        )
    intervals = pd.DataFrame(interval_records).sort_values("observed_difference")
    return replicates, intervals


def build_heart_outer_report(
    repo_root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions = load_heart_outer_predictions(
        repo_root,
        config["prediction_runs"],
        require_exact_mask_identity=bool(config.get("require_exact_mask_identity", False)),
    )
    demographics = pd.read_parquet(
        repo_root / config["canonical_data_path"], columns=["sample_id", "sex"]
    )
    predictions = predictions.merge(
        demographics, on="sample_id", how="left", validate="many_to_one"
    )
    metrics = cell_metrics(predictions)
    primary = primary_estimands(metrics)
    adaptation = adaptation_summary(metrics)
    sex_subgroups = descriptive_sex_subgroup_metrics(predictions)
    replicates, intervals = paired_primary_bootstrap(
        predictions,
        reference_method=str(config["bootstrap"]["reference_method"]),
        comparison_methods=[str(value) for value in config["bootstrap"]["comparison_methods"]],
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    paths = {
        "predictions": output_dir / "normalized_outer_predictions.parquet",
        "cell_metrics": output_dir / "site_policy_metrics.csv",
        "primary": output_dir / "primary_estimands.csv",
        "adaptation": output_dir / "adaptation_summary.csv",
        "sex_subgroups": output_dir / "descriptive_sex_subgroup_metrics.csv",
        "bootstrap_replicates": output_dir / "paired_bootstrap_replicates.parquet",
        "bootstrap_intervals": output_dir / "paired_bootstrap_intervals.csv",
        "manifest": output_dir / "report_manifest.json",
    }
    predictions.to_parquet(paths["predictions"], index=False)
    metrics.to_csv(paths["cell_metrics"], index=False)
    primary.to_csv(paths["primary"], index=False)
    adaptation.to_csv(paths["adaptation"], index=False)
    sex_subgroups.to_csv(paths["sex_subgroups"], index=False)
    replicates.to_parquet(paths["bootstrap_replicates"], index=False)
    intervals.to_csv(paths["bootstrap_intervals"], index=False)
    paths["manifest"].write_text(
        json.dumps(
            {
                "status": "derived_from_locked_sample_level_predictions",
                "bootstrap_unit": "patient_within_observed_outer_hospital",
                "hospital_inference": "conditional_on_four_observed_hospitals",
                "primary_track": PRIMARY_TRACK,
                "prediction_rows": len(predictions),
                "methods": sorted(predictions["method"].unique()),
                "tracks": sorted(predictions["track"].unique()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
