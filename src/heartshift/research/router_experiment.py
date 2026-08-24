"""Cross-fitted source-only evaluation of the support-aware expert router."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from heartshift.config import config_hash
from heartshift.data.uci import FEATURE_COLUMNS, sha256_file
from heartshift.evaluation.classical_benchmark import write_run_manifest
from heartshift.metrics import balanced_log_loss, binary_metrics
from heartshift.models.support_shrinkage import (
    fit_support_aware_router,
    predict_support_aware_router,
)

KEY_COLUMNS = (
    "sample_id",
    "outer_target",
    "inner_validation",
    "policy",
    "mask_replicate",
)


def _selected_neural_predictions(
    run_dir: Path,
    experiments: dict[str, str],
    prediction_seeds: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    predictions = pd.read_parquet(run_dir / "inner_predictions.parquet")
    selections = pd.read_csv(run_dir / "selected_configurations.csv")
    predictions = predictions.loc[predictions["seed"].isin(prediction_seeds)]
    mask_columns = [f"observed__{feature}" for feature in FEATURE_COLUMNS]
    outputs = {}
    for expert_name, experiment in experiments.items():
        records = []
        for outer_target in predictions["outer_target"].unique():
            selected = selections.loc[
                selections["outer_target"].eq(outer_target)
                & selections["experiment"].eq(experiment)
            ]
            if len(selected) != 1:
                raise AssertionError(
                    f"Neural selection is not unique for {outer_target}/{experiment}"
                )
            parameter_id = int(selected.iloc[0]["parameter_id"])
            records.append(
                predictions.loc[
                    predictions["outer_target"].eq(outer_target)
                    & predictions["experiment"].eq(experiment)
                    & predictions["parameter_id"].eq(parameter_id)
                ]
            )
        selected_predictions = pd.concat(records, ignore_index=True)
        audit = selected_predictions.groupby(list(KEY_COLUMNS), sort=False).agg(
            target_values=("target", "nunique"),
            observed_fraction_min=("observed_fraction", "min"),
            observed_fraction_max=("observed_fraction", "max"),
            mask_values=("observed_mask_code", "nunique"),
            seed_count=("seed", "nunique"),
        )
        if audit["target_values"].ne(1).any() or audit["mask_values"].ne(1).any():
            raise AssertionError("Neural expert seeds disagree on endpoints or exact masks")
        if (audit["observed_fraction_max"] - audit["observed_fraction_min"]).max() > 1e-12:
            raise AssertionError(
                "Neural expert seeds used different evaluation masks; ensembling is invalid"
            )
        if audit["seed_count"].ne(len(prediction_seeds)).any():
            raise AssertionError("A neural expert evaluation key is missing a declared seed")
        outputs[expert_name] = selected_predictions.groupby(
            [
                *KEY_COLUMNS,
                "site",
                "target",
                "observed_mask_code",
                *mask_columns,
            ],
            as_index=False,
        ).agg(
            y_score=("y_score", "mean"),
            observed_fraction=("observed_fraction", "mean"),
        )
    return outputs


def _anchor_candidate_predictions(
    run_dir: Path,
    prediction_seeds: tuple[int, ...],
) -> pd.DataFrame:
    predictions = pd.read_parquet(run_dir / "inner_predictions.parquet")
    predictions = predictions.loc[predictions["seed"].isin(prediction_seeds)]
    mask_columns = [f"observed__{feature}" for feature in FEATURE_COLUMNS]
    candidate_keys = [*KEY_COLUMNS, "control", "parameter_id"]
    audit = predictions.groupby(candidate_keys, sort=False).agg(
        target_values=("target", "nunique"),
        mask_values=("observed_mask_code", "nunique"),
        seed_count=("seed", "nunique"),
    )
    if audit["target_values"].ne(1).any() or audit["mask_values"].ne(1).any():
        raise AssertionError("Anchor seeds disagree on targets or exact masks")
    if audit["seed_count"].ne(len(prediction_seeds)).any():
        raise AssertionError("An anchor evaluation key is missing a declared seed")
    return predictions.groupby(
        [
            *KEY_COLUMNS,
            "site",
            "target",
            "observed_mask_code",
            *mask_columns,
            "control",
            "parameter_id",
            "parameters_json",
        ],
        as_index=False,
    ).agg(
        y_score=("y_score", "mean"),
        observed_fraction=("observed_fraction", "mean"),
    )


def assemble_router_table(
    neural_run_dir: Path,
    control_run_dir: Path,
    *,
    neural_experiments: dict[str, str],
    prediction_seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, tuple[str, ...], pd.DataFrame]:
    """Align neural experts and retain all nested anchor candidates."""
    neural = _selected_neural_predictions(neural_run_dir, neural_experiments, prediction_seeds)
    neural_names = tuple(neural_experiments)
    if not neural_names:
        raise ValueError("At least one fixed neural expert is required")
    first_name = neural_names[0]
    table = neural[first_name].rename(columns={"y_score": f"expert__{first_name}"})
    for name in neural_names[1:]:
        frame = neural[name]
        candidate = frame.rename(
            columns={
                "y_score": f"expert__{name}",
                "target": f"target__{name}",
                "site": f"site__{name}",
                "observed_fraction": f"observed_fraction__{name}",
                "observed_mask_code": f"observed_mask_code__{name}",
                **{
                    column: f"{column}__{name}"
                    for column in [f"observed__{feature}" for feature in FEATURE_COLUMNS]
                },
            }
        )
        table = table.merge(candidate, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
        if not table["target"].eq(table[f"target__{name}"]).all():
            raise AssertionError(f"Expert {name} endpoints do not align")
        if not table["site"].eq(table[f"site__{name}"]).all():
            raise AssertionError(f"Expert {name} hospitals do not align")
        if not np.allclose(
            table["observed_fraction"], table[f"observed_fraction__{name}"], atol=1e-12
        ):
            raise AssertionError(f"Expert {name} evaluation masks do not align")
        if not table["observed_mask_code"].eq(table[f"observed_mask_code__{name}"]).all():
            raise AssertionError(f"Expert {name} exact mask codes do not align")
        for column in [f"observed__{feature}" for feature in FEATURE_COLUMNS]:
            if not table[column].eq(table[f"{column}__{name}"]).all():
                raise AssertionError(f"Expert {name} feature-level masks do not align")
        table = table.drop(
            columns=[
                f"target__{name}",
                f"site__{name}",
                f"observed_fraction__{name}",
                f"observed_mask_code__{name}",
                *[f"observed__{feature}__{name}" for feature in FEATURE_COLUMNS],
            ]
        )
    anchor_candidates = _anchor_candidate_predictions(control_run_dir, prediction_seeds)
    expected_rows = len(neural[first_name])
    if len(table) != expected_rows:
        raise AssertionError("Neural expert alignment dropped evaluation rows")
    neural_key_masks = table.loc[:, [*KEY_COLUMNS, "observed_mask_code"]].drop_duplicates()
    anchor_key_masks = anchor_candidates.loc[
        :, [*KEY_COLUMNS, "observed_mask_code"]
    ].drop_duplicates()
    mask_audit = neural_key_masks.merge(
        anchor_key_masks,
        on=list(KEY_COLUMNS),
        how="outer",
        suffixes=("__neural", "__anchor"),
        indicator=True,
    )
    if (
        not mask_audit["_merge"].eq("both").all()
        or not mask_audit["observed_mask_code__neural"]
        .eq(mask_audit["observed_mask_code__anchor"])
        .all()
    ):
        raise AssertionError("Neural and anchor-candidate exact mask banks do not align")
    return table, neural_names, anchor_candidates


def _group_codes(frame: pd.DataFrame) -> np.ndarray:
    groups = (
        frame["site"].astype(str)
        + "|"
        + frame["policy"].astype(str)
        + "|y="
        + frame["target"].astype(str)
    )
    return np.asarray(pd.Categorical(groups).codes, dtype=np.int64)


def _robust_score(frame: pd.DataFrame, score: np.ndarray) -> float:
    working = frame.loc[:, ["site", "policy", "mask_replicate", "target"]].copy()
    working["score"] = score
    cells = []
    for _, group in working.groupby(["site", "policy", "mask_replicate"], sort=False):
        cells.append(balanced_log_loss(group["target"], group["score"]))
    return float(0.5 * (np.mean(cells) + np.max(cells)))


def _simplex_grid(n_experts: int, step: float) -> np.ndarray:
    units = round(1.0 / step)
    if units < 1 or not np.isclose(units * step, 1.0):
        raise ValueError("Convex grid step must divide one")
    weights = []
    for values in itertools.product(range(units + 1), repeat=n_experts):
        if sum(values) == units:
            weights.append(np.asarray(values, dtype=np.float64) / units)
    return np.stack(weights)


def _fixed_blend_selection(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    step: float,
) -> tuple[np.ndarray, int]:
    grid = _simplex_grid(probabilities.shape[1], step)
    expert_logits = logit(np.clip(probabilities, 1e-6, 1 - 1e-6))
    scores = [_robust_score(validation, expit(expert_logits @ weights)) for weights in grid]
    selected = int(np.argmin(scores))
    return grid[selected], selected


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "outer_target",
        "meta_training_site",
        "meta_early_stop_site",
        "meta_evaluation_site",
        "router_seed",
        "method",
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


def _router_gate(metrics: pd.DataFrame, weights: pd.DataFrame) -> dict[str, Any]:
    policy = metrics.groupby(
        ["router_seed", "method", "outer_target", "meta_evaluation_site", "policy"],
        as_index=False,
    ).agg(
        balanced_log_loss=("balanced_log_loss", "mean"),
        roc_auc=("roc_auc", "mean"),
    )
    robust = policy.groupby(
        ["router_seed", "method", "outer_target", "meta_evaluation_site"], as_index=False
    ).agg(worst_policy_bll=("balanced_log_loss", "max"))
    summary = robust.groupby("method", as_index=False).agg(
        mean_site_worst_bll=("worst_policy_bll", "mean")
    )
    natural = (
        policy.loc[policy["policy"].eq("natural")]
        .groupby("method", as_index=False)
        .agg(natural_auroc=("roc_auc", "mean"))
    )
    summary = summary.merge(natural, on="method", validate="one_to_one").set_index("method")
    router = summary.loc["support_aware_router"]
    competitors = summary.drop(index="support_aware_router")
    strongest = competitors["mean_site_worst_bll"].idxmin()
    maximum_weight = weights[[column for column in weights if column.startswith("weight__")]].max(
        axis=1
    )
    checks: dict[str, dict[str, Any]] = {
        "robust_improvement_over_strongest_fixed": {
            "router": float(router["mean_site_worst_bll"]),
            "strongest_method": str(strongest),
            "strongest": float(competitors.loc[strongest, "mean_site_worst_bll"]),
            "passed": bool(
                router["mean_site_worst_bll"] < competitors.loc[strongest, "mean_site_worst_bll"]
            ),
        },
        "natural_auroc_noninferiority": {
            "router": float(router["natural_auroc"]),
            "strongest": float(competitors.loc[strongest, "natural_auroc"]),
            "margin": 0.01,
            "passed": bool(
                router["natural_auroc"] >= competitors.loc[strongest, "natural_auroc"] - 0.01
            ),
        },
        "router_not_collapsed": {
            "mean_maximum_weight": float(maximum_weight.mean()),
            "threshold": 0.95,
            "passed": bool(maximum_weight.mean() < 0.95),
        },
    }
    return {
        "status": "passed_all_source_router_gates"
        if all(value["passed"] for value in checks.values())
        else "failed_one_or_more_source_router_gates",
        "checks": checks,
        "method_summary": summary.reset_index().to_dict(orient="records"),
    }


def run_router_experiment(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "consumed_source_oof_router_development")
    table, neural_expert_names, anchor_candidates = assemble_router_table(
        repo_root / str(config["neural_run"]),
        repo_root / str(config["control_run"]),
        neural_experiments={
            str(name): str(experiment) for name, experiment in config["neural_experiments"].items()
        },
        prediction_seeds=tuple(int(value) for value in config["prediction_seeds"]),
    )
    expert_names = (*neural_expert_names, "stable_anchor")
    table_path = run_dir / "aligned_source_oof_experts.parquet"
    table.to_parquet(table_path, index=False)
    anchor_candidates_path = run_dir / "aligned_anchor_candidates.parquet"
    anchor_candidates.to_parquet(anchor_candidates_path, index=False)
    mask_columns = [f"observed__{feature}" for feature in FEATURE_COLUMNS]
    prediction_records = []
    weight_records = []
    history_records = []
    anchor_selection_records = []
    router_seeds = tuple(int(value) for value in config["router_seeds"])
    probability_columns = [f"expert__{name}" for name in expert_names]
    for outer_target, context in table.groupby("outer_target", sort=True):
        outer_anchor_candidates = anchor_candidates.loc[
            anchor_candidates["outer_target"].eq(outer_target)
        ]
        sites = sorted(str(value) for value in context["inner_validation"].unique())
        if len(sites) != 3:
            raise ValueError("UCI source router development requires three source hospitals")
        for evaluation_index, evaluation_site in enumerate(sites):
            early_stop_site = sites[(evaluation_index + 1) % len(sites)]
            training_site = sites[(evaluation_index + 2) % len(sites)]
            early_anchor = outer_anchor_candidates.loc[
                outer_anchor_candidates["inner_validation"].eq(early_stop_site)
            ]
            anchor_scores: list[dict[str, Any]] = []
            for (control, parameter_id), candidate in early_anchor.groupby(
                ["control", "parameter_id"], sort=True
            ):
                anchor_scores.append(
                    {
                        "control": str(control),
                        "parameter_id": int(parameter_id),
                        "parameters_json": str(candidate.iloc[0]["parameters_json"]),
                        "early_stop_robust_score": _robust_score(
                            candidate, candidate["y_score"].to_numpy()
                        ),
                    }
                )
            if not anchor_scores:
                raise AssertionError("No nested anchor candidate is available")
            selected_anchor = sorted(
                anchor_scores,
                key=lambda row: (
                    float(row["early_stop_robust_score"]),
                    str(row["control"]),
                    int(row["parameter_id"]),
                ),
            )[0]
            anchor_selection_records.append(
                {
                    "outer_target": outer_target,
                    "meta_training_site": training_site,
                    "meta_early_stop_site": early_stop_site,
                    "meta_evaluation_site": evaluation_site,
                    **selected_anchor,
                }
            )
            selected_anchor_predictions = outer_anchor_candidates.loc[
                outer_anchor_candidates["control"].eq(selected_anchor["control"])
                & outer_anchor_candidates["parameter_id"].eq(selected_anchor["parameter_id"])
            ].rename(
                columns={
                    "y_score": "expert__stable_anchor",
                    "target": "target__stable_anchor",
                    "site": "site__stable_anchor",
                    "observed_fraction": "observed_fraction__stable_anchor",
                    "observed_mask_code": "observed_mask_code__stable_anchor",
                    **{column: f"{column}__stable_anchor" for column in mask_columns},
                }
            )
            cycle_context = context.merge(
                selected_anchor_predictions.loc[
                    :,
                    [
                        *KEY_COLUMNS,
                        "expert__stable_anchor",
                        "target__stable_anchor",
                        "site__stable_anchor",
                        "observed_fraction__stable_anchor",
                        "observed_mask_code__stable_anchor",
                        *[f"{column}__stable_anchor" for column in mask_columns],
                    ],
                ],
                on=list(KEY_COLUMNS),
                how="inner",
                validate="one_to_one",
            )
            if len(cycle_context) != len(context):
                raise AssertionError("Nested anchor selection dropped source OOF rows")
            if (
                not cycle_context["target"].eq(cycle_context["target__stable_anchor"]).all()
                or not cycle_context["site"].eq(cycle_context["site__stable_anchor"]).all()
            ):
                raise AssertionError("Nested anchor endpoints or hospitals do not align")
            if (
                not cycle_context["observed_mask_code"]
                .eq(cycle_context["observed_mask_code__stable_anchor"])
                .all()
            ):
                raise AssertionError("Nested anchor exact mask codes do not align")
            for column in mask_columns:
                if not cycle_context[column].eq(cycle_context[f"{column}__stable_anchor"]).all():
                    raise AssertionError("Nested anchor feature-level masks do not align")
            cycle_context = cycle_context.drop(
                columns=[
                    "target__stable_anchor",
                    "site__stable_anchor",
                    "observed_fraction__stable_anchor",
                    "observed_mask_code__stable_anchor",
                    *[f"{column}__stable_anchor" for column in mask_columns],
                ]
            )
            training = cycle_context.loc[cycle_context["inner_validation"].eq(training_site)].copy()
            early_stop = cycle_context.loc[
                cycle_context["inner_validation"].eq(early_stop_site)
            ].copy()
            evaluation = cycle_context.loc[
                cycle_context["inner_validation"].eq(evaluation_site)
            ].copy()
            training_probability = training.loc[:, probability_columns].to_numpy()
            early_probability = early_stop.loc[:, probability_columns].to_numpy()
            evaluation_probability = evaluation.loc[:, probability_columns].to_numpy()
            convex_weights, convex_id = _fixed_blend_selection(
                early_stop,
                early_probability,
                step=float(config["convex_grid_step"]),
            )
            early_expert_scores = [
                _robust_score(early_stop, early_probability[:, index])
                for index in range(len(expert_names))
            ]
            best_expert_index = int(np.argmin(early_expert_scores))
            for router_seed in router_seeds:
                result = fit_support_aware_router(
                    training_probability,
                    training.loc[:, mask_columns].to_numpy(dtype=bool),
                    training["target"].to_numpy(),
                    _group_codes(training),
                    training["policy"].eq("natural").to_numpy(),
                    early_probability,
                    early_stop.loc[:, mask_columns].to_numpy(dtype=bool),
                    early_stop["target"].to_numpy(),
                    _group_codes(early_stop),
                    expert_names=expert_names,
                    parameters=dict(config["router"]),
                    seed=router_seed,
                    device=str(config["device"]),
                )
                routed, routing_weights = predict_support_aware_router(
                    result,
                    evaluation_probability,
                    evaluation.loc[:, mask_columns].to_numpy(dtype=bool),
                )
                expert_logits = logit(np.clip(evaluation_probability, 1e-6, 1 - 1e-6))
                method_scores = {
                    **{
                        f"expert:{name}": evaluation_probability[:, index]
                        for index, name in enumerate(expert_names)
                    },
                    "equal_logit_blend": expit(expert_logits.mean(axis=1)),
                    "selected_best_expert": evaluation_probability[:, best_expert_index],
                    "selected_convex_blend": expit(expert_logits @ convex_weights),
                    "support_aware_router": routed,
                }
                provenance = evaluation.loc[
                    :,
                    [
                        "sample_id",
                        "site",
                        "target",
                        "policy",
                        "mask_replicate",
                        "observed_fraction",
                        "observed_mask_code",
                    ],
                ]
                for method, score in method_scores.items():
                    frame = provenance.copy()
                    frame["outer_target"] = outer_target
                    frame["meta_training_site"] = training_site
                    frame["meta_early_stop_site"] = early_stop_site
                    frame["meta_evaluation_site"] = evaluation_site
                    frame["router_seed"] = router_seed
                    frame["method"] = method
                    frame["y_score"] = score
                    frame["convex_selection_id"] = convex_id
                    frame["selected_anchor_control"] = selected_anchor["control"]
                    frame["selected_anchor_parameter_id"] = selected_anchor["parameter_id"]
                    prediction_records.append(frame)
                weight_frame = provenance.loc[
                    :, ["sample_id", "policy", "mask_replicate", "observed_mask_code"]
                ].copy()
                weight_frame["outer_target"] = outer_target
                weight_frame["meta_evaluation_site"] = evaluation_site
                weight_frame["router_seed"] = router_seed
                for index, name in enumerate(expert_names):
                    weight_frame[f"weight__{name}"] = routing_weights[:, index]
                weight_records.append(weight_frame)
                for row in result.history:
                    history_records.append(
                        {
                            "outer_target": outer_target,
                            "meta_training_site": training_site,
                            "meta_early_stop_site": early_stop_site,
                            "meta_evaluation_site": evaluation_site,
                            "router_seed": router_seed,
                            "best_epoch": result.best_epoch,
                            "validation_score": result.validation_score,
                            **row,
                        }
                    )
    predictions = pd.concat(prediction_records, ignore_index=True)
    weights = pd.concat(weight_records, ignore_index=True)
    metrics = _metric_rows(predictions)
    gates = _router_gate(metrics, weights)
    paths = {
        "table": table_path,
        "anchor_candidates": anchor_candidates_path,
        "predictions": run_dir / "source_meta_evaluation_predictions.parquet",
        "weights": run_dir / "routing_weights.parquet",
        "metrics": run_dir / "source_meta_evaluation_metrics.csv",
        "history": run_dir / "router_training_history.csv",
        "anchor_selections": run_dir / "nested_anchor_selections.csv",
        "gates": run_dir / "router_gates.json",
        "audit": run_dir / "evidence_audit.json",
    }
    predictions.to_parquet(paths["predictions"], index=False)
    weights.to_parquet(paths["weights"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    pd.DataFrame(history_records).to_csv(paths["history"], index=False)
    pd.DataFrame(anchor_selection_records).to_csv(paths["anchor_selections"], index=False)
    paths["gates"].write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {
        path.name: sha256_file(path)
        for name, path in paths.items()
        if name != "audit" and path.is_file()
    }
    paths["audit"].write_text(
        json.dumps(
            {
                "status": "complete_consumed_source_oof_development_only",
                "new_confirmatory_claim_allowed": False,
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
