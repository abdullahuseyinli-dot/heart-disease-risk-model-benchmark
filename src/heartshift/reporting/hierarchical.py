"""Hospital/patient hierarchical inference for future multi-centre confirmation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _weighted_balanced_loss(
    target: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
) -> float:
    probability = np.clip(score, 1e-7, 1 - 1e-7)
    point_loss = -(target * np.log(probability) + (1 - target) * np.log(1 - probability))
    class_means = []
    for label in (0, 1):
        selected = target == label
        denominator = weight[selected].sum()
        if denominator <= 0:
            raise ValueError("A hierarchical bootstrap cell lost an outcome class")
        class_means.append(float(np.sum(weight[selected] * point_loss[selected]) / denominator))
    return float(np.mean(class_means))


def _method_hospital_score(
    method_rows: pd.DataFrame,
    patient_counts: dict[Any, int],
) -> float:
    cells = []
    for (policy, replicate), cell in method_rows.groupby(["policy", "mask_replicate"], sort=False):
        del policy, replicate
        weights = cell["patient_id"].map(patient_counts).fillna(0).to_numpy(dtype=np.float64)
        cells.append(
            _weighted_balanced_loss(
                cell["target"].to_numpy(dtype=np.int8),
                cell["score"].to_numpy(dtype=np.float64),
                weights,
            )
        )
    cell_frame = method_rows.loc[:, ["policy", "mask_replicate"]].drop_duplicates().copy()
    cell_frame["balanced_log_loss"] = cells
    policy_means = cell_frame.groupby("policy")["balanced_log_loss"].mean()
    return float(policy_means.max())


def _ensemble_seed_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if "training_seed" not in predictions:
        return predictions.copy()
    grouping = [
        "sample_id",
        "patient_id",
        "hospital_id",
        "target",
        "policy",
        "mask_replicate",
        "method",
    ]
    for optional in ("observed_mask_code", "observed_mask_sha256"):
        if optional in predictions:
            grouping.append(optional)
    return predictions.groupby(grouping, as_index=False).agg(score=("score", "mean"))


def _observed_primary(predictions: pd.DataFrame) -> pd.Series:
    hospital_scores = []
    for (method, hospital), group in predictions.groupby(["method", "hospital_id"]):
        counts = {patient: 1 for patient in group["patient_id"].unique()}
        hospital_scores.append(
            {
                "method": method,
                "hospital_id": hospital,
                "score": _method_hospital_score(group, counts),
            }
        )
    return pd.DataFrame(hospital_scores).groupby("method")["score"].mean()


def hierarchical_hospital_patient_bootstrap(
    predictions: pd.DataFrame,
    *,
    reference_method: str,
    comparison_methods: list[str],
    repetitions: int,
    seed: int,
    include_training_seed_layer: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample hospitals, then outcome-stratified patient clusters within hospital.

    The hospital layer supports inference beyond the observed hospital mix. If
    individual-seed predictions are supplied, one common seed is sampled in each
    replicate before the hospital/patient draw, retaining fitted-model variation.
    """
    required = {
        "sample_id",
        "patient_id",
        "hospital_id",
        "target",
        "policy",
        "mask_replicate",
        "method",
        "score",
    }
    missing = required - set(predictions)
    if missing:
        raise KeyError(f"Hierarchical inference columns are missing: {sorted(missing)}")
    if repetitions < 1:
        raise ValueError("Hierarchical bootstrap repetitions must be positive")
    methods = [reference_method, *comparison_methods]
    selected = predictions.loc[predictions["method"].isin(methods)].copy()
    if set(selected["method"]) != set(methods):
        raise KeyError("One or more hierarchical-bootstrap methods are absent")
    if not np.isfinite(selected["score"]).all():
        raise ValueError("Hierarchical-bootstrap scores must be finite")
    endpoint_audit = selected.groupby(["hospital_id", "sample_id"])["target"].nunique()
    if endpoint_audit.ne(1).any():
        raise AssertionError("An external sample has inconsistent endpoints")
    patient_hospital = selected.groupby("patient_id")["hospital_id"].nunique()
    if patient_hospital.gt(1).any():
        raise AssertionError("A patient crosses hospitals in active confirmation predictions")
    key_columns = ["sample_id", "hospital_id", "policy", "mask_replicate"]
    if "training_seed" in selected:
        key_columns.append("training_seed")
    reference_keys = selected.loc[selected["method"].eq(reference_method), key_columns].sort_values(
        key_columns
    )
    for method in comparison_methods:
        candidate_keys = selected.loc[selected["method"].eq(method), key_columns].sort_values(
            key_columns
        )
        if not reference_keys.reset_index(drop=True).equals(candidate_keys.reset_index(drop=True)):
            raise AssertionError(f"Hierarchical paired keys differ for {method}")
    observed = _observed_primary(_ensemble_seed_predictions(selected))
    observed_differences = {
        method: float(observed[method] - observed[reference_method])
        for method in comparison_methods
    }
    seeds = (
        sorted(int(value) for value in selected["training_seed"].unique())
        if "training_seed" in selected and include_training_seed_layer
        else []
    )
    if seeds:
        seed_sets = selected.groupby("method")["training_seed"].apply(
            lambda values: set(int(value) for value in values)
        )
        if any(values != set(seeds) for values in seed_sets):
            raise AssertionError("Methods do not share the same training-seed layer")
    base = selected if seeds else _ensemble_seed_predictions(selected)
    hospitals = np.asarray(sorted(base["hospital_id"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    replicate_records = []
    for bootstrap_replicate in range(repetitions):
        sampled_seed = int(rng.choice(seeds)) if seeds else None
        current = (
            base.loc[base["training_seed"].eq(sampled_seed)] if sampled_seed is not None else base
        )
        hospital_draws = rng.choice(hospitals, size=len(hospitals), replace=True)
        method_estimates: dict[str, list[float]] = {method: [] for method in methods}
        for hospital in hospital_draws:
            hospital_rows = current.loc[current["hospital_id"].eq(hospital)]
            reference_rows = hospital_rows.loc[hospital_rows["method"].eq(reference_method)]
            patient_labels = reference_rows.groupby("patient_id")["target"].max()
            patient_counts: dict[Any, int] = {}
            for label in (0, 1):
                pool = patient_labels.loc[patient_labels.eq(label)].index.to_numpy()
                if not len(pool):
                    raise ValueError(
                        f"Hospital {hospital} has no patient cluster in endpoint stratum {label}"
                    )
                draws = rng.choice(pool, size=len(pool), replace=True)
                unique, counts = np.unique(draws, return_counts=True)
                patient_counts.update(dict(zip(unique, counts, strict=True)))
            for method in methods:
                method_rows = hospital_rows.loc[hospital_rows["method"].eq(method)]
                method_estimates[method].append(_method_hospital_score(method_rows, patient_counts))
        estimates = {method: float(np.mean(values)) for method, values in method_estimates.items()}
        for method in comparison_methods:
            replicate_records.append(
                {
                    "bootstrap_replicate": bootstrap_replicate,
                    "sampled_training_seed": sampled_seed,
                    "method": method,
                    "reference_method": reference_method,
                    "difference_candidate_minus_reference": (
                        estimates[method] - estimates[reference_method]
                    ),
                }
            )
    replicates = pd.DataFrame(replicate_records)
    intervals = []
    for (method, reference), group in replicates.groupby(
        ["method", "reference_method"], sort=False
    ):
        values = group["difference_candidate_minus_reference"]
        observed_difference = observed_differences[str(method)]
        low, high = (float(values.quantile(value)) for value in (0.025, 0.975))
        intervals.append(
            {
                "method": method,
                "reference_method": reference,
                "observed_difference": observed_difference,
                "bootstrap_mean_difference": float(values.mean()),
                "percentile_ci_025": low,
                "percentile_ci_975": high,
                "basic_ci_025": 2.0 * observed_difference - high,
                "basic_ci_975": 2.0 * observed_difference - low,
                "probability_better_descriptive": float(np.mean(values < 0.0)),
                "hospital_count": len(hospitals),
                "training_seed_layer_included": bool(seeds),
            }
        )
    return replicates, pd.DataFrame(intervals)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p-values for prespecified secondary tests."""
    if not p_values or any(not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("Holm adjustment requires non-empty p-values in [0, 1]")
    ordered = sorted(p_values, key=p_values.get)  # type: ignore[arg-type]
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (total - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted
