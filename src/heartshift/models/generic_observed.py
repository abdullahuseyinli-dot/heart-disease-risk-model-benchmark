"""Mini-batch observed-set learning for larger heterogeneous clinical tables."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch import nn

from heartshift.masks import policy_seed
from heartshift.models.observed_set import NeuralPreprocessor, ObservedFeatureSetEncoder
from heartshift.models.ps_maskdro import (
    _set_reproducibility,
    mirrams_objective,
    ps_maskdro_objective,
)


@dataclass(frozen=True)
class FeatureMaskPolicy:
    name: str
    kind: str
    rate: float = 0.0
    columns: tuple[str, ...] = ()


@dataclass
class GenericFitResult:
    model: ObservedFeatureSetEncoder
    preprocessor: NeuralPreprocessor
    best_epoch: int
    validation_score: float
    history: list[dict[str, float]]
    parameter_count: int


def apply_feature_mask_policy(
    natural: np.ndarray,
    feature_columns: tuple[str, ...],
    policy: FeatureMaskPolicy,
    *,
    base_seed: int,
    replicate: int,
) -> np.ndarray:
    """Apply a generic intervention that can only hide naturally observed values."""
    keep = natural.copy()
    if policy.kind == "natural":
        return keep
    rng = np.random.default_rng(policy_seed(base_seed, policy.name, replicate))
    if policy.kind == "mcar":
        keep &= rng.random(keep.shape) >= policy.rate
    elif policy.kind == "panel":
        for column in policy.columns:
            keep[:, feature_columns.index(column)] = False
    else:
        raise KeyError(f"Unknown generic policy kind: {policy.kind}")
    empty = ~keep.any(axis=1)
    for row in np.flatnonzero(empty):
        observed = np.flatnonzero(natural[row])
        if len(observed):
            keep[row, observed[0]] = True
    if np.any(keep & ~natural):
        raise AssertionError("A generic mask policy revealed unavailable information")
    return keep


def _environment_codes(data: pd.DataFrame, minimum_environment_size: int) -> np.ndarray:
    counts = data["environment"].value_counts()
    retained = set(counts.loc[counts >= minimum_environment_size].index)
    collapsed = data["environment"].where(data["environment"].isin(retained), "other")
    return np.asarray(pd.Categorical(collapsed).codes, dtype=np.int64)


def _balanced_batches(
    labels: np.ndarray,
    environments: np.ndarray,
    *,
    batch_size: int,
    steps: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    cells = [
        np.flatnonzero((environments == environment) & (labels == label))
        for environment in np.unique(environments)
        for label in (0, 1)
    ]
    if any(len(cell) == 0 for cell in cells):
        raise ValueError("Each retained environment requires both outcome classes")
    per_cell = max(1, math.ceil(batch_size / len(cells)))
    return [
        np.concatenate([rng.choice(cell, size=per_cell, replace=True) for cell in cells])
        for _ in range(steps)
    ]


def _uniform_batches(
    n_rows: int,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    order = rng.permutation(n_rows)
    return [order[start : start + batch_size] for start in range(0, n_rows, batch_size)]


def _training_policies(
    variant: str,
    evaluation_policies: tuple[FeatureMaskPolicy, ...],
    parameters: dict[str, Any],
) -> tuple[FeatureMaskPolicy, ...]:
    natural = FeatureMaskPolicy("natural", "natural")
    if variant in {"pooled", "prior_separated"}:
        return (natural,)
    if variant == "mirrams":
        return (
            natural,
            FeatureMaskPolicy(
                "mirrams_mcar",
                "mcar",
                rate=float(parameters.get("mirrams_mask_rate", 0.2)),
            ),
        )
    if variant in {"ps_maskdro", "ps_maskdro_ane"}:
        return evaluation_policies
    raise KeyError(f"Unknown generic variant: {variant}")


def _predict_logits_batched(
    model: ObservedFeatureSetEncoder,
    values: np.ndarray,
    observed: np.ndarray,
    reference_values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    reference = torch.tensor(reference_values, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            outputs.append(
                model(
                    torch.tensor(values[start:stop], dtype=torch.float32, device=device),
                    torch.tensor(observed[start:stop], dtype=torch.bool, device=device),
                    reference,
                )
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(outputs)


def predict_generic_policy_bank(
    result: GenericFitResult,
    data: pd.DataFrame,
    policies: tuple[FeatureMaskPolicy, ...],
    *,
    device: str,
    base_seed: int,
    replicates: int,
    batch_size: int,
) -> pd.DataFrame:
    values, natural = result.preprocessor.transform(data)
    records = []
    for policy in policies:
        policy_replicates = 1 if policy.kind in {"natural", "panel"} else replicates
        for replicate in range(policy_replicates):
            observed = apply_feature_mask_policy(
                natural,
                result.preprocessor.feature_columns,
                policy,
                base_seed=base_seed,
                replicate=replicate,
            )
            logits = _predict_logits_batched(
                result.model,
                values,
                observed,
                np.asarray(result.preprocessor.reference_values, dtype=np.float32),
                device=torch.device(device),
                batch_size=batch_size,
            )
            provenance = ["sample_id", "target", "split", "environment"]
            if "source_line_sha256" in data:
                provenance.append("source_line_sha256")
            if "patient_nbr" in data:
                provenance.append("patient_nbr")
            frame = data.loc[:, provenance].copy()
            frame["policy"] = policy.name
            frame["mask_replicate"] = replicate
            frame["evidence_logit"] = logits
            frame["y_score"] = expit(logits)
            frame["observed_fraction"] = observed.mean(axis=1)
            records.append(frame)
    return pd.concat(records, ignore_index=True)


def _validation_score(predictions: pd.DataFrame) -> float:
    from heartshift.metrics import balanced_log_loss

    losses = [
        balanced_log_loss(group["target"], group["y_score"])
        for _, group in predictions.groupby(["policy", "mask_replicate"], sort=False)
    ]
    return 0.5 * (float(np.mean(losses)) + float(np.max(losses)))


def fit_generic_observed_model(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    continuous_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    evaluation_policies: tuple[FeatureMaskPolicy, ...],
    variant: str,
    parameters: dict[str, Any],
    seed: int,
    device: str,
) -> GenericFitResult:
    """Train a larger clinical-table model using deterministic mini-batches."""
    _set_reproducibility(seed)
    torch_device = torch.device(device)
    preprocessor = NeuralPreprocessor.fit(
        training,
        feature_columns=feature_columns,
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    values, natural = preprocessor.transform(training)
    labels_array = training["target"].to_numpy(dtype=np.int64)
    environments_array = _environment_codes(
        training, int(parameters.get("minimum_environment_size", 500))
    )
    model = ObservedFeatureSetEncoder(
        n_features=len(feature_columns),
        continuous_indices=preprocessor.continuous_indices,
        categorical_cardinalities=preprocessor.categorical_cardinalities,
        d_model=int(parameters.get("d_model", 32)),
        n_heads=int(parameters.get("n_heads", 4)),
        n_layers=int(parameters.get("n_layers", 2)),
        dropout=float(parameters.get("dropout", 0.1)),
        ane=variant == "ps_maskdro_ane",
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters.get("learning_rate", 0.0005)),
        weight_decay=float(parameters.get("weight_decay", 0.001)),
    )
    policies = _training_policies(variant, evaluation_policies, parameters)
    max_epochs = int(parameters.get("max_epochs", 60))
    batch_size = int(parameters.get("batch_size", 384))
    steps = math.ceil(len(training) / batch_size)
    evaluation_interval = int(parameters.get("evaluation_interval", 2))
    patience = int(parameters.get("patience_evaluations", 6))
    inference_batch_size = int(parameters.get("inference_batch_size", 1024))
    reference = torch.tensor(
        preprocessor.reference_values, dtype=torch.float32, device=torch_device
    )
    history = []
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    for epoch in range(1, max_epochs + 1):
        rng = np.random.default_rng(policy_seed(seed, "minibatches", epoch))
        if variant in {"pooled", "mirrams"}:
            batches = _uniform_batches(len(training), batch_size=batch_size, rng=rng)
        else:
            batches = _balanced_batches(
                labels_array,
                environments_array,
                batch_size=batch_size,
                steps=steps,
                rng=rng,
            )
        epoch_objectives = []
        for step, indices in enumerate(batches):
            batch_values = torch.tensor(values[indices], dtype=torch.float32, device=torch_device)
            batch_labels = torch.tensor(
                labels_array[indices], dtype=torch.long, device=torch_device
            )
            batch_environments = torch.tensor(
                environments_array[indices], dtype=torch.long, device=torch_device
            )
            masks = np.stack(
                [
                    apply_feature_mask_policy(
                        natural[indices],
                        feature_columns,
                        policy,
                        base_seed=seed,
                        replicate=epoch * steps + step,
                    )
                    for policy in policies
                ]
            )
            n_policies, n_rows, _ = masks.shape
            logits = model(
                batch_values.repeat(n_policies, 1),
                torch.tensor(
                    masks.reshape(n_policies * n_rows, -1),
                    dtype=torch.bool,
                    device=torch_device,
                ),
                reference,
            ).reshape(n_policies, n_rows)
            if variant == "mirrams":
                objective, _ = mirrams_objective(
                    logits,
                    batch_labels,
                    lambda_supervised_mask=float(parameters.get("mirrams_lambda_1", 15.0)),
                    lambda_consistency=float(parameters.get("mirrams_lambda_2", 15.0)),
                    confidence_threshold=float(parameters.get("mirrams_confidence", 0.95)),
                )
            else:
                objective, _ = ps_maskdro_objective(
                    logits,
                    batch_labels,
                    batch_environments,
                    variant={
                        "pooled": "v0",
                        "prior_separated": "v2",
                        "ps_maskdro": "v5",
                        "ps_maskdro_ane": "v7",
                    }[variant],
                    dro_lambda=float(parameters.get("dro_lambda", 0.5)),
                    dro_tau=float(parameters.get("dro_tau", 0.2)),
                    brier_beta=float(parameters.get("brier_beta", 0.0)),
                )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_objectives.append(float(objective.detach()))

        if epoch % evaluation_interval and epoch != max_epochs:
            continue
        provisional = GenericFitResult(model, preprocessor, epoch, float("nan"), [], 0)
        validation_predictions = predict_generic_policy_bank(
            provisional,
            validation,
            evaluation_policies,
            device=device,
            base_seed=seed + 100_000,
            replicates=int(parameters.get("validation_mask_replicates", 2)),
            batch_size=inference_batch_size,
        )
        score = _validation_score(validation_predictions)
        history.append(
            {
                "epoch": float(epoch),
                "training_objective": float(np.mean(epoch_objectives)),
                "validation_score": score,
            }
        )
        if score < best_score - 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Generic observed-set training produced no checkpoint")
    model.load_state_dict(best_state)
    return GenericFitResult(
        model=model,
        preprocessor=preprocessor,
        best_epoch=best_epoch,
        validation_score=best_score,
        history=history,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def fit_generic_observed_fixed_epochs(
    training: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    continuous_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    evaluation_policies: tuple[FeatureMaskPolicy, ...],
    variant: str,
    parameters: dict[str, Any],
    seed: int,
    device: str,
    epochs: int,
) -> GenericFitResult:
    """Refit a source-selected generic model without consulting test labels."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    _set_reproducibility(seed)
    torch_device = torch.device(device)
    preprocessor = NeuralPreprocessor.fit(
        training,
        feature_columns=feature_columns,
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    values, natural = preprocessor.transform(training)
    labels_array = training["target"].to_numpy(dtype=np.int64)
    environments_array = _environment_codes(
        training, int(parameters.get("minimum_environment_size", 500))
    )
    model = ObservedFeatureSetEncoder(
        n_features=len(feature_columns),
        continuous_indices=preprocessor.continuous_indices,
        categorical_cardinalities=preprocessor.categorical_cardinalities,
        d_model=int(parameters.get("d_model", 32)),
        n_heads=int(parameters.get("n_heads", 4)),
        n_layers=int(parameters.get("n_layers", 2)),
        dropout=float(parameters.get("dropout", 0.1)),
        ane=variant == "ps_maskdro_ane",
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters.get("learning_rate", 0.0005)),
        weight_decay=float(parameters.get("weight_decay", 0.001)),
    )
    policies = _training_policies(variant, evaluation_policies, parameters)
    batch_size = int(parameters.get("batch_size", 384))
    steps = math.ceil(len(training) / batch_size)
    evaluation_interval = int(parameters.get("evaluation_interval", 2))
    reference = torch.tensor(
        preprocessor.reference_values, dtype=torch.float32, device=torch_device
    )
    history = []

    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(policy_seed(seed, "minibatches", epoch))
        if variant in {"pooled", "mirrams"}:
            batches = _uniform_batches(len(training), batch_size=batch_size, rng=rng)
        else:
            batches = _balanced_batches(
                labels_array,
                environments_array,
                batch_size=batch_size,
                steps=steps,
                rng=rng,
            )
        epoch_objectives = []
        for step, indices in enumerate(batches):
            batch_values = torch.tensor(values[indices], dtype=torch.float32, device=torch_device)
            batch_labels = torch.tensor(
                labels_array[indices], dtype=torch.long, device=torch_device
            )
            batch_environments = torch.tensor(
                environments_array[indices], dtype=torch.long, device=torch_device
            )
            masks = np.stack(
                [
                    apply_feature_mask_policy(
                        natural[indices],
                        feature_columns,
                        policy,
                        base_seed=seed,
                        replicate=epoch * steps + step,
                    )
                    for policy in policies
                ]
            )
            n_policies, n_rows, _ = masks.shape
            logits = model(
                batch_values.repeat(n_policies, 1),
                torch.tensor(
                    masks.reshape(n_policies * n_rows, -1),
                    dtype=torch.bool,
                    device=torch_device,
                ),
                reference,
            ).reshape(n_policies, n_rows)
            if variant == "mirrams":
                objective, _ = mirrams_objective(
                    logits,
                    batch_labels,
                    lambda_supervised_mask=float(parameters.get("mirrams_lambda_1", 15.0)),
                    lambda_consistency=float(parameters.get("mirrams_lambda_2", 15.0)),
                    confidence_threshold=float(parameters.get("mirrams_confidence", 0.95)),
                )
            else:
                objective, _ = ps_maskdro_objective(
                    logits,
                    batch_labels,
                    batch_environments,
                    variant={
                        "pooled": "v0",
                        "prior_separated": "v2",
                        "ps_maskdro": "v5",
                        "ps_maskdro_ane": "v7",
                    }[variant],
                    dro_lambda=float(parameters.get("dro_lambda", 0.5)),
                    dro_tau=float(parameters.get("dro_tau", 0.2)),
                    brier_beta=float(parameters.get("brier_beta", 0.0)),
                )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_objectives.append(float(objective.detach()))
        if epoch % evaluation_interval == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "training_objective": float(np.mean(epoch_objectives)),
                }
            )

    return GenericFitResult(
        model=model,
        preprocessor=preprocessor,
        best_epoch=epochs,
        validation_score=float("nan"),
        history=history,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
