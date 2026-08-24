"""Post-outcome stability report for an exact neural seed extension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heartshift.config import config_hash
from heartshift.data.uci import FEATURE_COLUMNS, sha256_file
from heartshift.reporting.outer_report import (
    cell_metrics,
    paired_primary_stratified_bootstrap,
    primary_estimands,
)
from heartshift.reporting.research_development import (
    _audit_paired_predictions,
    _normalise_controls,
    _normalise_neural,
    _seed_primary,
)
from heartshift.reporting.sensitivity import leave_one_site_out_ranking

SEED_KEY = (
    "sample_id",
    "outer_target",
    "experiment",
    "policy",
    "mask_replicate",
    "training_seed",
)
MASK_COLUMNS = tuple(f"observed__{feature}" for feature in FEATURE_COLUMNS)
REPRODUCTION_COLUMNS = (
    *SEED_KEY,
    "record_sha256",
    "observed_fraction",
    "observed_mask_code",
    *MASK_COLUMNS,
    "target",
    "fit_seed",
    "epochs",
    "y_score",
)


def exact_seed_reproduction(
    historical_path: Path,
    extension_path: Path,
    historical_seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Require the historical seed subset to reproduce at value level."""
    historical = pd.read_parquet(historical_path, columns=list(REPRODUCTION_COLUMNS))
    extension = pd.read_parquet(extension_path, columns=list(REPRODUCTION_COLUMNS))
    extension = extension.loc[extension["training_seed"].isin(historical_seeds)].copy()
    historical = historical.sort_values(list(SEED_KEY)).reset_index(drop=True)
    extension = extension.sort_values(list(SEED_KEY)).reset_index(drop=True)
    if len(historical) != len(extension):
        raise AssertionError("Historical and extension seed subsets have different row counts")
    for column in SEED_KEY:
        if not historical[column].equals(extension[column]):
            raise AssertionError(f"Historical seed key changed in {column}")
    exact_columns = (
        "record_sha256",
        "observed_mask_code",
        *MASK_COLUMNS,
        "target",
        "fit_seed",
        "epochs",
    )
    mismatch_counts = {
        column: int(historical[column].ne(extension[column]).sum()) for column in exact_columns
    }
    fraction_difference = np.abs(
        historical["observed_fraction"].to_numpy(dtype=np.float64)
        - extension["observed_fraction"].to_numpy(dtype=np.float64)
    )
    score_difference = np.abs(
        historical["y_score"].to_numpy(dtype=np.float64)
        - extension["y_score"].to_numpy(dtype=np.float64)
    )
    if any(mismatch_counts.values()):
        raise AssertionError("Historical seed provenance did not reproduce exactly")
    if fraction_difference.max(initial=0.0) != 0.0:
        raise AssertionError("Historical observed fractions did not reproduce exactly")
    if score_difference.max(initial=0.0) != 0.0:
        raise AssertionError("Historical probabilities did not reproduce exactly")
    return {
        "status": "exact_value_reproduction",
        "rows": len(historical),
        "historical_seeds": list(historical_seeds),
        "mismatch_counts": mismatch_counts,
        "maximum_observed_fraction_difference": float(fraction_difference.max(initial=0.0)),
        "maximum_probability_difference": float(score_difference.max(initial=0.0)),
    }


def _rename_neural_namespace(frame: pd.DataFrame, namespace: str) -> pd.DataFrame:
    result = frame.copy()
    result["method"] = result["method"].str.replace("neural:", f"{namespace}:", regex=False)
    return result


def _three_vs_ten(primary: pd.DataFrame) -> pd.DataFrame:
    historical = primary.loc[primary["method"].str.startswith("psmask3:")].copy()
    extension = primary.loc[primary["method"].str.startswith("psmask10:")].copy()
    historical["experiment"] = historical["method"].str.removeprefix("psmask3:")
    extension["experiment"] = extension["method"].str.removeprefix("psmask10:")
    columns = [
        "macro_site_worst_mask_balanced_log_loss",
        "worst_site_mask_balanced_log_loss",
        "macro_natural_roc_auc",
    ]
    table = historical.loc[:, ["experiment", *columns]].merge(
        extension.loc[:, ["experiment", *columns]],
        on="experiment",
        suffixes=("_3seed", "_10seed"),
        validate="one_to_one",
    )
    table["delta_10seed_minus_3seed_primary"] = (
        table["macro_site_worst_mask_balanced_log_loss_10seed"]
        - table["macro_site_worst_mask_balanced_log_loss_3seed"]
    )
    return table.sort_values("macro_site_worst_mask_balanced_log_loss_10seed")


def _paired_contrasts(
    predictions: pd.DataFrame,
    experiments: tuple[str, ...],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    replicate_records = []
    interval_records = []

    comparisons = [f"psmask10:{experiment}" for experiment in experiments]
    replicates, intervals = paired_primary_stratified_bootstrap(
        predictions,
        reference_method="control:random_forest_natural",
        comparison_methods=comparisons,
        repetitions=repetitions,
        seed=seed,
    )
    replicates["contrast_id"] = "ten_seed_family_vs_random_forest"
    intervals["contrast_id"] = "ten_seed_family_vs_random_forest"
    replicate_records.append(replicates)
    interval_records.append(intervals)

    for offset, experiment in enumerate(experiments, start=1):
        replicates, intervals = paired_primary_stratified_bootstrap(
            predictions,
            reference_method=f"psmask3:{experiment}",
            comparison_methods=[f"psmask10:{experiment}"],
            repetitions=repetitions,
            seed=seed + offset,
        )
        contrast_id = f"ten_seed_minus_three_seed:{experiment}"
        replicates["contrast_id"] = contrast_id
        intervals["contrast_id"] = contrast_id
        replicate_records.append(replicates)
        interval_records.append(intervals)

    replicates, intervals = paired_primary_stratified_bootstrap(
        predictions,
        reference_method="psmask10:v2_prior_separated",
        comparison_methods=["psmask10:v4_structured_policy_bank"],
        repetitions=repetitions,
        seed=seed + len(experiments) + 1,
    )
    replicates["contrast_id"] = "ten_seed_v4_minus_v2"
    intervals["contrast_id"] = "ten_seed_v4_minus_v2"
    replicate_records.append(replicates)
    interval_records.append(intervals)
    return (
        pd.concat(replicate_records, ignore_index=True),
        pd.concat(interval_records, ignore_index=True),
    )


def multiplicity_sensitivity(
    replicates: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    contrast_id: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add Bonferroni percentile and joint max-error descriptive intervals."""
    family = replicates.loc[replicates["contrast_id"].eq(contrast_id)].copy()
    observed = intervals.loc[intervals["contrast_id"].eq(contrast_id)].set_index("method")[
        "observed_difference"
    ]
    methods = tuple(sorted(str(value) for value in family["method"].unique()))
    if not methods or set(methods) != set(observed.index.astype(str)):
        raise AssertionError("Multiplicity family and observed contrasts are misaligned")
    family_size = len(methods)
    lower_probability = alpha / (2 * family_size)
    upper_probability = 1.0 - lower_probability
    wide = family.pivot(
        index="bootstrap_replicate",
        columns="method",
        values="difference_candidate_minus_reference",
    ).loc[:, list(methods)]
    if wide.isna().any().any():
        raise AssertionError("Multiplicity family is missing a paired bootstrap replicate")
    maximum_error = (wide - observed.loc[list(methods)]).abs().max(axis=1)
    joint_critical_value = float(maximum_error.quantile(1.0 - alpha))
    records = []
    for method in methods:
        values = wide[method]
        point = float(observed.loc[method])
        records.append(
            {
                "contrast_id": contrast_id,
                "method": method,
                "family_size": family_size,
                "alpha": alpha,
                "bonferroni_percentile_ci_lower": float(values.quantile(lower_probability)),
                "bonferroni_percentile_ci_upper": float(values.quantile(upper_probability)),
                "simultaneous_max_error_critical_value": joint_critical_value,
                "simultaneous_max_error_ci_lower": point - joint_critical_value,
                "simultaneous_max_error_ci_upper": point + joint_critical_value,
            }
        )
    return pd.DataFrame(records)


def build_seed_extension_report(
    repo_root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Build an exact, explicitly post-outcome 3-versus-10-seed report."""
    output_dir.mkdir(parents=True, exist_ok=False)
    historical_run = repo_root / str(config["historical_run"])
    extension_run = repo_root / str(config["extension_run"])
    control_run = repo_root / str(config["control_run"])
    seeds = tuple(int(value) for value in config["seeds"])
    historical_seeds = tuple(int(value) for value in config["historical_seeds"])
    experiments = tuple(str(value) for value in config["experiments"])

    reproduction = exact_seed_reproduction(
        historical_run / "outer_seed_predictions.parquet",
        extension_run / "outer_seed_predictions.parquet",
        historical_seeds,
    )
    historical = _rename_neural_namespace(_normalise_neural(historical_run, {}), "psmask3")
    extension = _rename_neural_namespace(_normalise_neural(extension_run, {}), "psmask10")
    historical = historical.loc[
        historical["method"].isin(f"psmask3:{value}" for value in experiments)
    ]
    extension = extension.loc[
        extension["method"].isin(f"psmask10:{value}" for value in experiments)
    ]
    controls = _normalise_controls(control_run, seeds)
    control_methods = tuple(str(value) for value in config["control_methods"])
    controls = controls.loc[controls["method"].isin(control_methods)]
    predictions = pd.concat([historical, extension, controls], ignore_index=True)
    _audit_paired_predictions(predictions)

    metrics = cell_metrics(predictions)
    primary = primary_estimands(metrics)
    comparison = _three_vs_ten(primary)
    extension_seed_frame = pd.read_parquet(extension_run / "outer_seed_predictions.parquet")
    seed_sensitivity = _seed_primary(
        extension_seed_frame,
        namespace="psmask10",
        method_column="experiment",
        seed_column="training_seed",
        score_column="y_score",
    )
    repetitions = int(config["bootstrap"]["repetitions"])
    bootstrap_seed = int(config["bootstrap"]["seed"])
    bootstrap_replicates, bootstrap_intervals = _paired_contrasts(
        predictions, experiments, repetitions, bootstrap_seed
    )
    site_jackknife = leave_one_site_out_ranking(metrics)
    multiplicity = multiplicity_sensitivity(
        bootstrap_replicates,
        bootstrap_intervals,
        contrast_id="ten_seed_family_vs_random_forest",
    )

    outputs = {
        "predictions": output_dir / "normalized_predictions.parquet",
        "metrics": output_dir / "site_policy_metrics.csv",
        "primary": output_dir / "primary_estimands.csv",
        "comparison": output_dir / "three_vs_ten_seed_comparison.csv",
        "seed_sensitivity": output_dir / "ten_seed_sensitivity.csv",
        "bootstrap_replicates": output_dir / "paired_bootstrap_replicates.parquet",
        "bootstrap_intervals": output_dir / "paired_bootstrap_intervals.csv",
        "site_jackknife": output_dir / "leave_one_site_out_ranking.csv",
        "multiplicity": output_dir / "posthoc_multiplicity_sensitivity.csv",
        "reproduction": output_dir / "exact_reproduction.json",
        "manifest": output_dir / "report_manifest.json",
        "audit": output_dir / "evidence_audit.json",
    }
    predictions.to_parquet(outputs["predictions"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    primary.to_csv(outputs["primary"], index=False)
    comparison.to_csv(outputs["comparison"], index=False)
    seed_sensitivity.to_csv(outputs["seed_sensitivity"], index=False)
    bootstrap_replicates.to_parquet(outputs["bootstrap_replicates"], index=False)
    bootstrap_intervals.to_csv(outputs["bootstrap_intervals"], index=False)
    site_jackknife.to_csv(outputs["site_jackknife"], index=False)
    multiplicity.to_csv(outputs["multiplicity"], index=False)
    outputs["reproduction"].write_text(
        json.dumps(reproduction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs["manifest"].write_text(
        json.dumps(
            {
                "status": "post_outcome_consumed_uci_seed_sensitivity_only",
                "new_confirmatory_claim_allowed": False,
                "outer_outcomes_historically_consumed": True,
                "extension_initiated_after_review_of_consumed_outer_results": True,
                "exact_historical_seed_reproduction_required": True,
                "exact_cross_method_mask_identity_required": True,
                "conditional_inference": "four_observed_hospitals_and_fitted_models",
                "historical_seed_count": len(historical_seeds),
                "extension_seed_count": len(seeds),
                "method_count": int(predictions["method"].nunique()),
                "prediction_rows": len(predictions),
                "bootstrap_repetitions_per_contrast": repetitions,
                "config_sha256": config_hash(config),
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
                "status": "complete_post_outcome_seed_extension_report",
                "sha256": hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs
