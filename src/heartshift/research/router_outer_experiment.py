"""Source-selected support-router development with endpoint-isolated outer inference.

The four UCI outer outcomes have been consumed by earlier project versions, so this
module cannot create new confirmation.  It nevertheless enforces the prospective
mechanics needed for later external confirmation: every architecture, anchor,
blend, router checkpoint, and target probability is fixed on source data and
written to disk before target endpoints are loaded.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit

from heartshift.config import config_hash
from heartshift.data.uci import FEATURE_COLUMNS, sha256_file
from heartshift.evaluation.classical_benchmark import write_run_manifest
from heartshift.metrics import binary_metrics
from heartshift.models.support_shrinkage import (
    fit_support_aware_router,
    predict_support_aware_router,
)
from heartshift.reporting.hierarchical import hierarchical_hospital_patient_bootstrap
from heartshift.reporting.outer_report import (
    PRIMARY_TRACK,
    cell_metrics,
    paired_primary_stratified_bootstrap,
    primary_estimands,
)
from heartshift.research.router_experiment import (
    KEY_COLUMNS,
    _fixed_blend_selection,
    _group_codes,
    _robust_score,
)

MASK_COLUMNS = tuple(f"observed__{feature}" for feature in FEATURE_COLUMNS)
OUTER_KEY_COLUMNS = ("sample_id", "outer_target", "policy", "mask_replicate")


def select_neural_experiments(
    selections: pd.DataFrame,
    experiment_groups: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Choose one backbone per expert family using source-only inner scores."""
    required = {
        "outer_target",
        "experiment",
        "variant",
        "parameter_id",
        "selection_score",
        "mean_parameter_count",
    }
    if missing := required - set(selections):
        raise KeyError(f"Neural selection columns are missing: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    for outer_target in sorted(str(value) for value in selections["outer_target"].unique()):
        target_rows = selections.loc[selections["outer_target"].eq(outer_target)]
        for expert_name, candidates in experiment_groups.items():
            matched = target_rows.loc[target_rows["experiment"].isin(candidates)].copy()
            if set(matched["experiment"].astype(str)) != set(candidates):
                raise AssertionError(
                    f"Incomplete source-only backbone candidates for {outer_target}/{expert_name}"
                )
            selected = matched.sort_values(
                ["selection_score", "mean_parameter_count", "experiment"],
                ascending=[True, True, True],
            ).iloc[0]
            records.append(
                {
                    "outer_target": outer_target,
                    "expert_name": str(expert_name),
                    "selected_experiment": str(selected["experiment"]),
                    "variant": str(selected["variant"]),
                    "parameter_id": int(selected["parameter_id"]),
                    "selection_score": float(selected["selection_score"]),
                    "mean_parameter_count": float(selected["mean_parameter_count"]),
                    "candidate_experiments": json.dumps(sorted(candidates)),
                }
            )
    result = pd.DataFrame(records)
    expected = len(selections["outer_target"].unique()) * len(experiment_groups)
    if len(result) != expected:
        raise AssertionError("Source-only neural expert selection is incomplete")
    return result


def _audit_seed_bank(
    frame: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    seed_column: str,
    expected_seed_count: int,
    endpoint_required: bool,
) -> None:
    aggregations: dict[str, tuple[str, str]] = {
        "mask_values": ("observed_mask_code", "nunique"),
        "seed_count": (seed_column, "nunique"),
        "observed_fraction_min": ("observed_fraction", "min"),
        "observed_fraction_max": ("observed_fraction", "max"),
    }
    if endpoint_required:
        aggregations["target_values"] = ("target", "nunique")
    audit = frame.groupby(list(key_columns), sort=False).agg(**aggregations)
    if audit.empty or audit["mask_values"].ne(1).any():
        raise AssertionError("An expert seed bank disagrees on exact masks")
    if audit["seed_count"].ne(expected_seed_count).any():
        raise AssertionError("An expert evaluation key is missing a declared seed")
    if (
        audit["observed_fraction_max"] - audit["observed_fraction_min"]
    ).max() > 1e-12:
        raise AssertionError("An expert seed bank used different evaluation masks")
    if endpoint_required and audit["target_values"].ne(1).any():
        raise AssertionError("An expert seed bank disagrees on source endpoints")


def _ensemble_neural_source(
    run_dir: Path,
    selected_experiments: pd.DataFrame,
    prediction_seeds: tuple[int, ...],
) -> pd.DataFrame:
    columns = [
        *KEY_COLUMNS,
        "site",
        "target",
        "observed_fraction",
        "observed_mask_code",
        *MASK_COLUMNS,
        "experiment",
        "parameter_id",
        "seed",
        "y_score",
    ]
    predictions = pd.read_parquet(run_dir / "inner_predictions.parquet", columns=columns)
    predictions = predictions.loc[predictions["seed"].isin(prediction_seeds)]
    records = []
    for selection in selected_experiments.itertuples(index=False):
        selected = predictions.loc[
            predictions["outer_target"].eq(selection.outer_target)
            & predictions["experiment"].eq(selection.selected_experiment)
            & predictions["parameter_id"].eq(selection.parameter_id)
        ].copy()
        if selected.empty:
            raise AssertionError("A selected neural source expert has no predictions")
        _audit_seed_bank(
            selected,
            key_columns=KEY_COLUMNS,
            seed_column="seed",
            expected_seed_count=len(prediction_seeds),
            endpoint_required=True,
        )
        grouped = selected.groupby(
            [*KEY_COLUMNS, "site", "target", "observed_mask_code", *MASK_COLUMNS],
            as_index=False,
        ).agg(
            y_score=("y_score", "mean"),
            observed_fraction=("observed_fraction", "mean"),
        )
        grouped["expert_name"] = selection.expert_name
        grouped["selected_experiment"] = selection.selected_experiment
        records.append(grouped)
    return pd.concat(records, ignore_index=True)


def _ensemble_anchor_source(
    run_dir: Path,
    prediction_seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors = pd.read_csv(run_dir / "selected_anchors.csv")
    required = {"outer_target", "selected_anchor", "parameter_id"}
    if missing := required - set(anchors):
        raise KeyError(f"Anchor selection columns are missing: {sorted(missing)}")
    columns = [
        *KEY_COLUMNS,
        "site",
        "target",
        "observed_fraction",
        "observed_mask_code",
        *MASK_COLUMNS,
        "control",
        "parameter_id",
        "seed",
        "y_score",
    ]
    predictions = pd.read_parquet(run_dir / "inner_predictions.parquet", columns=columns)
    predictions = predictions.loc[predictions["seed"].isin(prediction_seeds)]
    records = []
    for anchor in anchors.itertuples(index=False):
        selected = predictions.loc[
            predictions["outer_target"].eq(anchor.outer_target)
            & predictions["control"].eq(anchor.selected_anchor)
            & predictions["parameter_id"].eq(anchor.parameter_id)
        ].copy()
        if selected.empty:
            raise AssertionError("A selected classical anchor has no source predictions")
        _audit_seed_bank(
            selected,
            key_columns=KEY_COLUMNS,
            seed_column="seed",
            expected_seed_count=len(prediction_seeds),
            endpoint_required=True,
        )
        grouped = selected.groupby(
            [*KEY_COLUMNS, "site", "target", "observed_mask_code", *MASK_COLUMNS],
            as_index=False,
        ).agg(
            y_score=("y_score", "mean"),
            observed_fraction=("observed_fraction", "mean"),
        )
        grouped["expert_name"] = "stable_anchor"
        grouped["selected_anchor"] = anchor.selected_anchor
        grouped["selected_anchor_parameter_id"] = int(anchor.parameter_id)
        records.append(grouped)
    return pd.concat(records, ignore_index=True), anchors


def _align_expert_frames(
    frames: dict[str, pd.DataFrame],
    *,
    key_columns: tuple[str, ...],
    endpoint_required: bool,
) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one expert frame is required")
    first_name = next(iter(frames))
    base_columns = [
        *key_columns,
        "site",
        "observed_fraction",
        "observed_mask_code",
        *MASK_COLUMNS,
    ]
    if "record_sha256" in frames[first_name]:
        base_columns.append("record_sha256")
    if endpoint_required:
        base_columns.append("target")
    table = frames[first_name].loc[:, [*base_columns, "y_score"]].rename(
        columns={"y_score": f"expert__{first_name}"}
    )
    if table.duplicated(list(key_columns)).any():
        raise AssertionError(f"Expert {first_name} has duplicate evaluation keys")
    expected_rows = len(table)
    for name, frame in list(frames.items())[1:]:
        if len(frame) != expected_rows:
            raise AssertionError(f"Expert {name} has an incomplete evaluation bank")
        candidate_columns = [
            *key_columns,
            "site",
            "observed_fraction",
            "observed_mask_code",
            *MASK_COLUMNS,
            "y_score",
        ]
        if "record_sha256" in base_columns:
            candidate_columns.append("record_sha256")
        if endpoint_required:
            candidate_columns.append("target")
        renames = {
            column: f"{column}__{name}"
            for column in candidate_columns
            if column not in key_columns and column != "y_score"
        }
        renames["y_score"] = f"expert__{name}"
        candidate = frame.loc[:, candidate_columns].rename(columns=renames)
        if candidate.duplicated(list(key_columns)).any():
            raise AssertionError(f"Expert {name} has duplicate evaluation keys")
        table = table.merge(
            candidate,
            on=list(key_columns),
            how="inner",
            validate="one_to_one",
        )
        checks = ["site", "observed_mask_code", *MASK_COLUMNS]
        if "record_sha256" in base_columns:
            checks.append("record_sha256")
        if endpoint_required:
            checks.append("target")
        for column in checks:
            if not table[column].eq(table[f"{column}__{name}"]).all():
                raise AssertionError(f"Expert {name} disagrees on aligned {column}")
        if not np.allclose(
            table["observed_fraction"],
            table[f"observed_fraction__{name}"],
            atol=1e-12,
        ):
            raise AssertionError(f"Expert {name} disagrees on observed fraction")
        table = table.drop(
            columns=[
                *[f"{column}__{name}" for column in checks],
                f"observed_fraction__{name}",
            ]
        )
    if len(table) != expected_rows:
        raise AssertionError("Expert alignment dropped evaluation rows")
    return table


def assemble_source_experts(
    neural_run_dir: Path,
    control_run_dir: Path,
    *,
    experiment_groups: dict[str, tuple[str, ...]],
    prediction_seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    """Assemble source-OOF experts and record all source-only selections."""
    selections = pd.read_csv(neural_run_dir / "selected_configurations.csv")
    selected_experiments = select_neural_experiments(selections, experiment_groups)
    neural = _ensemble_neural_source(
        neural_run_dir, selected_experiments, prediction_seeds
    )
    frames = {
        expert_name: neural.loc[neural["expert_name"].eq(expert_name)].copy()
        for expert_name in experiment_groups
    }
    anchor, anchors = _ensemble_anchor_source(control_run_dir, prediction_seeds)
    frames["stable_anchor"] = anchor
    table = _align_expert_frames(
        frames, key_columns=KEY_COLUMNS, endpoint_required=True
    )
    return table, tuple(frames), selected_experiments, anchors


def _load_neural_outer_experts(
    run_dir: Path,
    selected_experiments: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for selection in selected_experiments.itertuples(index=False):
        path = (
            run_dir
            / "shards"
            / f"{selection.outer_target}__{selection.selected_experiment}"
            / "unlabelled_outer_predictions.parquet"
        )
        frame = pd.read_parquet(path)
        if "target" in frame:
            raise AssertionError("Neural endpoint-free shard unexpectedly contains target")
        if not frame["outer_target"].eq(selection.outer_target).all():
            raise AssertionError("Neural outer shard target-site provenance changed")
        frame = frame.rename(columns={"y_score_zero_shot": "y_score"})
        frame["expert_name"] = selection.expert_name
        frame["selected_experiment"] = selection.selected_experiment
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def _load_anchor_outer_experts(
    run_dir: Path,
    anchors: pd.DataFrame,
    prediction_seeds: tuple[int, ...],
) -> pd.DataFrame:
    predictions = pd.read_parquet(run_dir / "outer_unlabelled_predictions.parquet")
    if "target" in predictions:
        raise AssertionError("Classical endpoint-free predictions unexpectedly contain target")
    predictions = predictions.loc[predictions["training_seed"].isin(prediction_seeds)]
    records = []
    for anchor in anchors.itertuples(index=False):
        selected = predictions.loc[
            predictions["outer_target"].eq(anchor.outer_target)
            & predictions["control"].eq(anchor.selected_anchor)
            & predictions["parameter_id"].eq(anchor.parameter_id)
        ].copy()
        if selected.empty:
            raise AssertionError("A selected anchor has no endpoint-free outer predictions")
        _audit_seed_bank(
            selected,
            key_columns=OUTER_KEY_COLUMNS,
            seed_column="training_seed",
            expected_seed_count=len(prediction_seeds),
            endpoint_required=False,
        )
        grouping = [
            *OUTER_KEY_COLUMNS,
            "site",
            "record_sha256",
            "observed_mask_code",
            *MASK_COLUMNS,
        ]
        grouped = selected.groupby(grouping, as_index=False).agg(
            y_score=("y_score", "mean"),
            observed_fraction=("observed_fraction", "mean"),
        )
        grouped["expert_name"] = "stable_anchor"
        grouped["selected_anchor"] = anchor.selected_anchor
        grouped["selected_anchor_parameter_id"] = int(anchor.parameter_id)
        records.append(grouped)
    return pd.concat(records, ignore_index=True)


def assemble_outer_experts(
    neural_outer_run_dir: Path,
    control_run_dir: Path,
    *,
    selected_experiments: pd.DataFrame,
    anchors: pd.DataFrame,
    expert_names: tuple[str, ...],
    prediction_seeds: tuple[int, ...],
) -> pd.DataFrame:
    """Align endpoint-free outer experts under exact feature-level masks."""
    neural = _load_neural_outer_experts(neural_outer_run_dir, selected_experiments)
    anchor = _load_anchor_outer_experts(control_run_dir, anchors, prediction_seeds)
    frames = {
        name: (
            anchor.copy()
            if name == "stable_anchor"
            else neural.loc[neural["expert_name"].eq(name)].copy()
        )
        for name in expert_names
    }
    return _align_expert_frames(
        frames, key_columns=OUTER_KEY_COLUMNS, endpoint_required=False
    )


def _router_extra_features(
    feature_mode: str,
    probabilities: np.ndarray,
) -> np.ndarray | None:
    if feature_mode == "support_only":
        return None
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.asarray(logit(clipped), dtype=np.float64)
    if feature_mode == "expert_logit_context":
        return logits
    if feature_mode == "anchor_relative_context":
        anchor = logits[:, [-1]]
        return np.concatenate([anchor, logits[:, :-1] - anchor], axis=1)
    raise KeyError(f"Unknown router feature mode: {feature_mode}")


def _outer_cell_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouping = ["outer_target", "method", "policy", "mask_replicate"]
    for keys, group in predictions.groupby(grouping, sort=False):
        records.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                **binary_metrics(group["target"], group["y_score"]),
            }
        )
    return pd.DataFrame(records)


def _normalise_outer_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    normalised = predictions.rename(columns={"y_score": "score"}).copy()
    normalised["track"] = PRIMARY_TRACK
    columns = [
        "sample_id",
        "outer_target",
        "target",
        "policy",
        "mask_replicate",
        "observed_fraction",
        "observed_mask_code",
        "method",
        "track",
        "score",
    ]
    result = normalised.loc[:, columns]
    if result.duplicated(
        ["sample_id", "outer_target", "policy", "mask_replicate", "method"]
    ).any():
        raise AssertionError("An outer method has duplicate paired evaluation keys")
    if not np.isfinite(result["score"]).all():
        raise AssertionError("An outer method produced a non-finite probability")
    audit = result.groupby(
        ["sample_id", "outer_target", "policy", "mask_replicate"], sort=False
    ).agg(
        methods=("method", "nunique"),
        target_values=("target", "nunique"),
        mask_values=("observed_mask_code", "nunique"),
        fraction_min=("observed_fraction", "min"),
        fraction_max=("observed_fraction", "max"),
    )
    expected_methods = result["method"].nunique()
    if audit["methods"].ne(expected_methods).any():
        raise AssertionError("Outer methods do not share complete paired keys")
    if audit["target_values"].ne(1).any() or audit["mask_values"].ne(1).any():
        raise AssertionError("Outer methods disagree on endpoints or exact masks")
    if (audit["fraction_max"] - audit["fraction_min"]).max() > 1e-12:
        raise AssertionError("Outer methods disagree on observed fractions")
    return result


def _router_gate(
    normalised: pd.DataFrame,
    ensemble_weights: pd.DataFrame,
    *,
    natural_auroc_margin: float,
) -> dict[str, Any]:
    metrics = cell_metrics(normalised)
    primary = primary_estimands(metrics).set_index("method")
    natural = metrics.loc[
        metrics["track"].eq(PRIMARY_TRACK) & metrics["policy"].eq("natural")
    ].groupby("method", as_index=True)["roc_auc"].mean()
    router_name = "support_aware_router"
    router_bll = float(
        primary.loc[router_name, "macro_site_worst_mask_balanced_log_loss"]
    )
    fixed_primary = primary.drop(index=router_name)
    strongest_method = str(
        fixed_primary["macro_site_worst_mask_balanced_log_loss"].idxmin()
    )
    reference_bll = float(
        fixed_primary.loc[
            strongest_method, "macro_site_worst_mask_balanced_log_loss"
        ]
    )
    maximum_weight = ensemble_weights[
        [column for column in ensemble_weights if column.startswith("weight__")]
    ].max(axis=1)
    checks: dict[str, dict[str, Any]] = {
        "descriptive_robust_improvement_over_strongest_fixed": {
            "router": router_bll,
            "strongest_method": strongest_method,
            "reference": reference_bll,
            "passed": bool(router_bll < reference_bll),
        },
        "descriptive_natural_auroc_noninferiority": {
            "router": float(natural[router_name]),
            "strongest_method": strongest_method,
            "reference": float(natural[strongest_method]),
            "margin": natural_auroc_margin,
            "passed": bool(
                natural[router_name]
                >= natural[strongest_method] - natural_auroc_margin
            ),
        },
        "router_not_collapsed": {
            "mean_maximum_ensemble_weight": float(maximum_weight.mean()),
            "threshold": 0.95,
            "passed": bool(maximum_weight.mean() < 0.95),
        },
    }
    return {
        "status": (
            "passed_all_development_gates"
            if all(check["passed"] for check in checks.values())
            else "failed_one_or_more_development_gates"
        ),
        "new_confirmatory_claim_allowed": False,
        "checks": checks,
        "primary_estimands": primary.reset_index().to_dict(orient="records"),
    }


def _free_router(result: Any) -> None:
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_outer_router_experiment(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    """Select on source OOF data, predict endpoints blind, then score descriptively."""
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(
        repo_root,
        run_dir,
        config,
        "consumed_uci_source_selected_outer_router_development",
    )
    prediction_seeds = tuple(int(value) for value in config["prediction_seeds"])
    router_seeds = tuple(int(value) for value in config["router_seeds"])
    experiment_groups = {
        str(name): tuple(str(value) for value in values)
        for name, values in config["neural_experiment_groups"].items()
    }
    source, expert_names, selected_experiments, anchors = assemble_source_experts(
        repo_root / str(config["neural_inner_run"]),
        repo_root / str(config["control_run"]),
        experiment_groups=experiment_groups,
        prediction_seeds=prediction_seeds,
    )
    outer = assemble_outer_experts(
        repo_root / str(config["neural_outer_run"]),
        repo_root / str(config["control_run"]),
        selected_experiments=selected_experiments,
        anchors=anchors,
        expert_names=expert_names,
        prediction_seeds=prediction_seeds,
    )
    source_path = run_dir / "aligned_source_oof_experts.parquet"
    outer_experts_path = run_dir / "aligned_endpoint_free_outer_experts.parquet"
    selections_path = run_dir / "source_selected_neural_experts.csv"
    anchors_path = run_dir / "source_selected_anchors.csv"
    source.to_parquet(source_path, index=False)
    outer.to_parquet(outer_experts_path, index=False)
    selected_experiments.to_csv(selections_path, index=False)
    anchors.to_csv(anchors_path, index=False)

    probability_columns = [f"expert__{name}" for name in expert_names]
    member_predictions: list[pd.DataFrame] = []
    member_weights: list[pd.DataFrame] = []
    validation_records: list[dict[str, Any]] = []
    history_records: list[dict[str, Any]] = []
    fixed_records: list[pd.DataFrame] = []
    blend_records: list[dict[str, Any]] = []
    candidate_selection_records: list[dict[str, Any]] = []

    for outer_target, source_context in source.groupby("outer_target", sort=True):
        target_context = outer.loc[outer["outer_target"].eq(outer_target)].copy()
        if target_context.empty:
            raise AssertionError(f"No endpoint-free target experts for {outer_target}")
        source_probability = source_context.loc[:, probability_columns].to_numpy()
        target_probability = target_context.loc[:, probability_columns].to_numpy()
        convex_weights, convex_id = _fixed_blend_selection(
            source_context,
            source_probability,
            step=float(config["convex_grid_step"]),
        )
        expert_source_scores = [
            _robust_score(source_context, source_probability[:, index])
            for index in range(len(expert_names))
        ]
        best_expert_index = int(np.argmin(expert_source_scores))
        blend_records.append(
            {
                "outer_target": outer_target,
                "convex_selection_id": convex_id,
                "selected_best_expert": expert_names[best_expert_index],
                **{
                    f"convex_weight__{name}": float(convex_weights[index])
                    for index, name in enumerate(expert_names)
                },
            }
        )
        target_logits = logit(np.clip(target_probability, 1e-6, 1 - 1e-6))
        fixed_scores = {
            **{
                f"expert:{name}": target_probability[:, index]
                for index, name in enumerate(expert_names)
            },
            "equal_logit_blend": expit(target_logits.mean(axis=1)),
            "selected_best_expert": target_probability[:, best_expert_index],
            "selected_convex_blend": expit(target_logits @ convex_weights),
        }
        provenance = target_context.loc[
            :,
            [
                "sample_id",
                "outer_target",
                "site",
                "record_sha256",
                "policy",
                "mask_replicate",
                "observed_fraction",
                "observed_mask_code",
            ],
        ]
        for method, scores in fixed_scores.items():
            frame = provenance.copy()
            frame["method"] = method
            frame["y_score"] = scores
            fixed_records.append(frame)

        sites = sorted(str(value) for value in source_context["site"].unique())
        if len(sites) != 3:
            raise ValueError("UCI router source selection requires three source hospitals")
        target_masks = target_context.loc[:, list(MASK_COLUMNS)].to_numpy(dtype=bool)
        for candidate in config["router_candidates"]:
            candidate_name = str(candidate["name"])
            feature_mode = str(candidate["feature_mode"])
            parameters = dict(config["router"]) | dict(candidate.get("parameters", {}))
            for validation_site in sites:
                training = source_context.loc[source_context["site"].ne(validation_site)]
                validation = source_context.loc[source_context["site"].eq(validation_site)]
                training_probability = training.loc[:, probability_columns].to_numpy()
                validation_probability = validation.loc[:, probability_columns].to_numpy()
                training_extra = _router_extra_features(feature_mode, training_probability)
                validation_extra = _router_extra_features(feature_mode, validation_probability)
                target_extra = _router_extra_features(feature_mode, target_probability)
                for router_seed in router_seeds:
                    result = fit_support_aware_router(
                        training_probability,
                        training.loc[:, list(MASK_COLUMNS)].to_numpy(dtype=bool),
                        training["target"].to_numpy(),
                        _group_codes(training),
                        training["policy"].eq("natural").to_numpy(),
                        validation_probability,
                        validation.loc[:, list(MASK_COLUMNS)].to_numpy(dtype=bool),
                        validation["target"].to_numpy(),
                        _group_codes(validation),
                        expert_names=expert_names,
                        parameters=parameters,
                        seed=router_seed,
                        device=str(config["device"]),
                        training_extra_features=training_extra,
                        validation_extra_features=validation_extra,
                    )
                    validation_prediction, _ = predict_support_aware_router(
                        result,
                        validation_probability,
                        validation.loc[:, list(MASK_COLUMNS)].to_numpy(dtype=bool),
                        extra_features=validation_extra,
                    )
                    target_prediction, routing_weights = predict_support_aware_router(
                        result,
                        target_probability,
                        target_masks,
                        extra_features=target_extra,
                    )
                    validation_records.append(
                        {
                            "outer_target": outer_target,
                            "router_candidate": candidate_name,
                            "feature_mode": feature_mode,
                            "validation_site": validation_site,
                            "router_seed": router_seed,
                            "balanced_validation_score": _robust_score(
                                validation, validation_prediction
                            ),
                            "training_objective_validation_score": result.validation_score,
                            "best_epoch": result.best_epoch,
                            "parameter_count": result.parameter_count,
                        }
                    )
                    member = provenance.copy()
                    member["router_candidate"] = candidate_name
                    member["feature_mode"] = feature_mode
                    member["validation_site"] = validation_site
                    member["router_seed"] = router_seed
                    member["y_score"] = target_prediction
                    member_predictions.append(member)
                    weight_frame = provenance.loc[
                        :,
                        [
                            "sample_id",
                            "outer_target",
                            "policy",
                            "mask_replicate",
                            "observed_mask_code",
                        ],
                    ].copy()
                    weight_frame["router_candidate"] = candidate_name
                    weight_frame["feature_mode"] = feature_mode
                    weight_frame["validation_site"] = validation_site
                    weight_frame["router_seed"] = router_seed
                    for index, name in enumerate(expert_names):
                        weight_frame[f"weight__{name}"] = routing_weights[:, index]
                    member_weights.append(weight_frame)
                    for history in result.history:
                        history_records.append(
                            {
                                "outer_target": outer_target,
                                "router_candidate": candidate_name,
                                "feature_mode": feature_mode,
                                "validation_site": validation_site,
                                "router_seed": router_seed,
                                "best_epoch": result.best_epoch,
                                **history,
                            }
                        )
                    _free_router(result)

        target_validation = pd.DataFrame(validation_records)
        target_validation = target_validation.loc[
            target_validation["outer_target"].eq(outer_target)
        ]
        summary = target_validation.groupby(
            ["router_candidate", "feature_mode"], as_index=False
        ).agg(
            mean_balanced_validation_score=("balanced_validation_score", "mean"),
            worst_balanced_validation_score=("balanced_validation_score", "max"),
            mean_best_epoch=("best_epoch", "mean"),
            parameter_count=("parameter_count", "first"),
        )
        summary["selection_score"] = 0.5 * (
            summary["mean_balanced_validation_score"]
            + summary["worst_balanced_validation_score"]
        )
        selected_candidate = summary.sort_values(
            ["selection_score", "parameter_count", "router_candidate"]
        ).iloc[0]
        candidate_selection_records.append(
            {"outer_target": outer_target, **selected_candidate.to_dict()}
        )

    all_members = pd.concat(member_predictions, ignore_index=True)
    all_weights = pd.concat(member_weights, ignore_index=True)
    candidate_selections = pd.DataFrame(candidate_selection_records)
    selected_keys = candidate_selections.loc[
        :, ["outer_target", "router_candidate"]
    ]
    selected_members = all_members.merge(
        selected_keys,
        on=["outer_target", "router_candidate"],
        how="inner",
        validate="many_to_one",
    )
    selected_weights = all_weights.merge(
        selected_keys,
        on=["outer_target", "router_candidate"],
        how="inner",
        validate="many_to_one",
    )
    expected_members = 3 * len(router_seeds)
    member_counts = selected_members.groupby(
        ["sample_id", "outer_target", "policy", "mask_replicate"], sort=False
    ).size()
    weight_counts = selected_weights.groupby(
        ["sample_id", "outer_target", "policy", "mask_replicate"], sort=False
    ).size()
    if member_counts.ne(expected_members).any() or weight_counts.ne(expected_members).any():
        raise AssertionError("A selected router ensemble is missing a declared member")
    router_ensemble = selected_members.groupby(
        [
            "sample_id",
            "outer_target",
            "site",
            "record_sha256",
            "policy",
            "mask_replicate",
            "observed_fraction",
            "observed_mask_code",
        ],
        as_index=False,
    ).agg(y_score=("y_score", "mean"))
    router_ensemble["method"] = "support_aware_router"
    weight_columns = [f"weight__{name}" for name in expert_names]
    ensemble_weights = selected_weights.groupby(
        ["sample_id", "outer_target", "policy", "mask_replicate", "observed_mask_code"],
        as_index=False,
    )[weight_columns].mean()
    endpoint_free_predictions = pd.concat(
        [*fixed_records, router_ensemble], ignore_index=True
    )

    paths = {
        "source_experts": source_path,
        "outer_experts": outer_experts_path,
        "neural_selections": selections_path,
        "anchor_selections": anchors_path,
        "blend_selections": run_dir / "source_selected_fixed_blends.csv",
        "router_validation": run_dir / "source_router_validation_scores.csv",
        "router_candidate_selections": run_dir / "source_selected_router_candidates.csv",
        "router_history": run_dir / "router_training_history.csv",
        "endpoint_free_members": run_dir / "endpoint_free_router_member_predictions.parquet",
        "endpoint_free_predictions": run_dir / "endpoint_free_outer_predictions.parquet",
        "member_weights": run_dir / "router_member_weights.parquet",
        "ensemble_weights": run_dir / "router_ensemble_weights.parquet",
        "predictions": run_dir / "outer_predictions.parquet",
        "member_predictions": run_dir / "outer_router_member_predictions.parquet",
        "metrics": run_dir / "outer_metrics.csv",
        "primary": run_dir / "primary_estimands.csv",
        "member_summary": run_dir / "outer_router_member_summary.csv",
        "bootstrap_replicates": run_dir / "conditional_bootstrap_replicates.parquet",
        "bootstrap_intervals": run_dir / "conditional_bootstrap_intervals.csv",
        "hierarchical_replicates": run_dir / "hierarchical_bootstrap_replicates.parquet",
        "hierarchical_intervals": run_dir / "hierarchical_bootstrap_intervals.csv",
        "gates": run_dir / "router_gates.json",
        "audit": run_dir / "evidence_audit.json",
    }
    pd.DataFrame(blend_records).to_csv(paths["blend_selections"], index=False)
    pd.DataFrame(validation_records).to_csv(paths["router_validation"], index=False)
    candidate_selections.to_csv(paths["router_candidate_selections"], index=False)
    pd.DataFrame(history_records).to_csv(paths["router_history"], index=False)
    all_members.to_parquet(paths["endpoint_free_members"], index=False)
    endpoint_free_predictions.to_parquet(paths["endpoint_free_predictions"], index=False)
    all_weights.to_parquet(paths["member_weights"], index=False)
    ensemble_weights.to_parquet(paths["ensemble_weights"], index=False)
    if "target" in endpoint_free_predictions or "target" in all_members:
        raise AssertionError("A target endpoint entered endpoint-free router predictions")

    # Historical target endpoints are opened only after every model and probability
    # has been selected and durably persisted.  This is still descriptive because
    # earlier repository versions already consumed these outcomes.
    labels = pd.read_parquet(
        repo_root / str(config["data"]["canonical_path"]),
        columns=["sample_id", "site", "target"],
    ).rename(columns={"site": "outer_target"})
    labels = labels.loc[:, ["sample_id", "outer_target", "target"]]
    predictions = endpoint_free_predictions.merge(
        labels,
        on=["sample_id", "outer_target"],
        how="left",
        validate="many_to_one",
    )
    labelled_members = all_members.merge(
        labels,
        on=["sample_id", "outer_target"],
        how="left",
        validate="many_to_one",
    )
    if predictions["target"].isna().any() or labelled_members["target"].isna().any():
        raise AssertionError("A persisted router prediction is missing its endpoint")
    predictions.to_parquet(paths["predictions"], index=False)
    labelled_members.to_parquet(paths["member_predictions"], index=False)
    metrics = _outer_cell_metrics(predictions)
    metrics.to_csv(paths["metrics"], index=False)
    normalised = _normalise_outer_predictions(predictions)
    primary = primary_estimands(cell_metrics(normalised))
    primary.to_csv(paths["primary"], index=False)

    member_summary_records = []
    for keys, group in labelled_members.groupby(
        ["outer_target", "router_candidate", "validation_site", "router_seed"],
        sort=False,
    ):
        natural = group.loc[group["policy"].eq("natural")]
        member_summary_records.append(
            {
                **dict(
                    zip(
                        [
                            "outer_target",
                            "router_candidate",
                            "validation_site",
                            "router_seed",
                        ],
                        keys,
                        strict=True,
                    )
                ),
                "site_worst_balanced_log_loss": _robust_score(
                    group, group["y_score"].to_numpy()
                ),
                "natural_roc_auc": binary_metrics(
                    natural["target"], natural["y_score"]
                )["roc_auc"],
            }
        )
    pd.DataFrame(member_summary_records).to_csv(paths["member_summary"], index=False)

    reference_method = str(config["inference"]["reference_method"])
    comparison_methods = [
        str(value) for value in config["inference"]["comparison_methods"]
    ]
    bootstrap_replicates, bootstrap_intervals = paired_primary_stratified_bootstrap(
        normalised,
        reference_method=reference_method,
        comparison_methods=comparison_methods,
        repetitions=int(config["inference"]["conditional_repetitions"]),
        seed=int(config["inference"]["seed"]),
    )
    bootstrap_replicates.to_parquet(paths["bootstrap_replicates"], index=False)
    bootstrap_intervals.to_csv(paths["bootstrap_intervals"], index=False)
    hierarchical_input = normalised.rename(
        columns={"outer_target": "hospital_id"}
    ).copy()
    hierarchical_input["patient_id"] = hierarchical_input["sample_id"]
    hierarchical_replicates, hierarchical_intervals = (
        hierarchical_hospital_patient_bootstrap(
            hierarchical_input,
            reference_method=reference_method,
            comparison_methods=comparison_methods,
            repetitions=int(config["inference"]["hierarchical_repetitions"]),
            seed=int(config["inference"]["seed"]),
            include_training_seed_layer=False,
        )
    )
    hierarchical_replicates.to_parquet(paths["hierarchical_replicates"], index=False)
    hierarchical_intervals.to_csv(paths["hierarchical_intervals"], index=False)
    gates = _router_gate(
        normalised,
        ensemble_weights,
        natural_auroc_margin=float(config["natural_auroc_margin"]),
    )
    paths["gates"].write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hashes = {
        path.name: sha256_file(path)
        for name, path in paths.items()
        if name != "audit" and path.is_file()
    }
    paths["audit"].write_text(
        json.dumps(
            {
                "status": f"complete_{config['status']}",
                "new_confirmatory_claim_allowed": False,
                "outer_outcomes_historically_consumed": True,
                "current_run_target_endpoint_loaded_after_all_predictions": True,
                "source_only_model_and_router_selection": True,
                "source_meta_evaluation_claim": False,
                "conditional_inference": "four_observed_hospitals_and_fitted_models",
                "hierarchical_inference_limitation": "only_four_observed_hospitals",
                "config_sha256": config_hash(config),
                "sha256": hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
