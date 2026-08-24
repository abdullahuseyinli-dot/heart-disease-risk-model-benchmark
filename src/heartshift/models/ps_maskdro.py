"""Training and inference for prior-separated site-by-mask robust learning."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

from heartshift.data.uci import CORE_COLUMNS, FEATURE_COLUMNS
from heartshift.masks import MaskPolicy, apply_mask_policy, default_policy_bank
from heartshift.metrics import binary_metrics
from heartshift.models.observed_set import NeuralPreprocessor, ObservedFeatureSetEncoder

VALID_VARIANTS = (
    "v0",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5_site",
    "v5_mask",
    "v5",
    "v7",
    "mirrams",
)

CORE_PREDICTION_COLUMNS = tuple(f"core__{feature}" for feature in CORE_COLUMNS)
MASK_PREDICTION_COLUMNS = tuple(f"observed__{feature}" for feature in FEATURE_COLUMNS)


@dataclass
class FitResult:
    model: ObservedFeatureSetEncoder
    preprocessor: NeuralPreprocessor
    best_epoch: int
    validation_score: float
    history: list[dict[str, float]]
    variant: str
    parameter_count: int
    device: str


def policies_for_variant(variant: str) -> tuple[MaskPolicy, ...]:
    if variant not in VALID_VARIANTS:
        raise KeyError(f"Unknown PS-MaskDRO variant: {variant}")
    if variant == "mirrams":
        return (
            MaskPolicy("natural", "natural"),
            MaskPolicy("mirrams_mcar", "mcar", rate=0.2),
        )
    bank = {policy.name: policy for policy in default_policy_bank()}
    names: tuple[str, ...]
    if variant in {"v0", "v1", "v2"}:
        names = ("natural",)
    elif variant == "v3":
        names = ("natural", "mcar_10", "mcar_30", "mcar_50")
    else:
        names = tuple(bank)
    return tuple(bank[name] for name in names)


def _balanced_group_mean(
    loss: torch.Tensor,
    labels: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    class_losses = []
    for label in (0, 1):
        cell = selected & labels.eq(label)
        if cell.any():
            class_losses.append(loss[cell].mean())
    if len(class_losses) != 2:
        raise ValueError("Every training hospital must contain both outcome classes")
    return torch.stack(class_losses).mean()


def ps_maskdro_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    site_codes: torch.Tensor,
    *,
    variant: str,
    dro_lambda: float,
    dro_tau: float,
    brier_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute pooled, site-balanced, prior-separated, or smooth-DRO risk."""
    if logits.ndim != 2 or logits.shape[1] != len(labels):
        raise ValueError("logits must have shape [policies, patients]")
    losses = functional.binary_cross_entropy_with_logits(
        logits,
        labels.float().unsqueeze(0).expand_as(logits),
        reduction="none",
    )
    probabilities = torch.sigmoid(logits)
    brier = torch.square(probabilities - labels.float().unsqueeze(0))
    if variant == "v0":
        risk = losses[0].mean()
        brier_risk = brier[0].mean()
        objective = risk + brier_beta * brier_risk
        return objective, {
            "mean_group_risk": float(risk.detach()),
            "robust_group_risk": float(risk.detach()),
            "mean_group_brier": float(brier_risk.detach()),
        }

    risk_rows = []
    brier_rows = []
    sites = torch.unique(site_codes, sorted=True)
    for policy_index in range(logits.shape[0]):
        policy_risks = []
        policy_briers = []
        for site in sites:
            selected = site_codes.eq(site)
            if variant == "v1":
                policy_risks.append(losses[policy_index, selected].mean())
                policy_briers.append(brier[policy_index, selected].mean())
            else:
                policy_risks.append(_balanced_group_mean(losses[policy_index], labels, selected))
                policy_briers.append(_balanced_group_mean(brier[policy_index], labels, selected))
        risk_rows.append(torch.stack(policy_risks))
        brier_rows.append(torch.stack(policy_briers))
    risk_matrix = torch.stack(risk_rows)
    brier_matrix = torch.stack(brier_rows)
    risks = risk_matrix.reshape(-1)
    briers = brier_matrix.reshape(-1)
    mean_risk = risks.mean()
    if variant in {"v5_site", "v5_mask", "v5", "v7"}:
        if variant == "v5_site":
            robust_inputs = risk_matrix.mean(dim=0)
        elif variant == "v5_mask":
            robust_inputs = risk_matrix.mean(dim=1)
        else:
            robust_inputs = risks
        robust_risk = dro_tau * (
            torch.logsumexp(robust_inputs / dro_tau, dim=0) - math.log(len(robust_inputs))
        )
        risk = (1.0 - dro_lambda) * mean_risk + dro_lambda * robust_risk
    else:
        robust_risk = risks.max()
        risk = mean_risk
    objective = risk + brier_beta * briers.mean()
    return objective, {
        "mean_group_risk": float(mean_risk.detach()),
        "robust_group_risk": float(robust_risk.detach()),
        "mean_group_brier": float(briers.mean().detach()),
    }


def mirrams_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda_supervised_mask: float = 15.0,
    lambda_consistency: float = 15.0,
    confidence_threshold: float = 0.95,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Implement MIRRAMS Equation 9 for binary outcomes on a shared backbone."""
    if logits.shape != (2, len(labels)):
        raise ValueError("MIRRAMS requires natural and additionally masked logits")
    natural_logits, masked_logits = logits
    labels_float = labels.float()
    natural_loss = functional.binary_cross_entropy_with_logits(natural_logits, labels_float)
    masked_loss = functional.binary_cross_entropy_with_logits(masked_logits, labels_float)
    with torch.no_grad():
        natural_probability = torch.sigmoid(natural_logits)
        pseudo_labels = natural_probability.ge(0.5).float()
        confidence = torch.maximum(natural_probability, 1.0 - natural_probability)
        confident = confidence.ge(confidence_threshold)
    if confident.any():
        consistency = functional.binary_cross_entropy_with_logits(
            masked_logits[confident], pseudo_labels[confident]
        )
    else:
        consistency = masked_logits.sum() * 0.0
    objective = (
        natural_loss + lambda_supervised_mask * masked_loss + lambda_consistency * consistency
    )
    return objective, {
        "natural_supervised_risk": float(natural_loss.detach()),
        "masked_supervised_risk": float(masked_loss.detach()),
        "consistency_risk": float(consistency.detach()),
        "confident_fraction": float(confident.float().mean().detach()),
    }


def _policy_masks(
    data: pd.DataFrame,
    policies: tuple[MaskPolicy, ...],
    *,
    base_seed: int,
    replicate: int,
    empirical_mask_pool: np.ndarray,
) -> np.ndarray:
    masks = []
    for policy in policies:
        kwargs = {"empirical_mask_pool": empirical_mask_pool} if policy.kind == "empirical" else {}
        masks.append(
            apply_mask_policy(
                data,
                policy,
                base_seed=base_seed,
                replicate=replicate,
                **kwargs,
            )
        )
    return np.stack(masks, axis=0)


def predict_policy_bank(
    model: ObservedFeatureSetEncoder,
    preprocessor: NeuralPreprocessor,
    data: pd.DataFrame,
    policies: tuple[MaskPolicy, ...],
    *,
    device: torch.device,
    base_seed: int,
    replicates: int,
    empirical_mask_pool: np.ndarray,
) -> pd.DataFrame:
    values, _ = preprocessor.transform(data)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    reference = torch.tensor(preprocessor.reference_values, dtype=torch.float32, device=device)
    records: list[pd.DataFrame] = []
    model.eval()
    for replicate in range(replicates):
        for policy in policies:
            if replicate > 0 and policy.kind in {"natural", "panel"}:
                continue
            kwargs = (
                {"empirical_mask_pool": empirical_mask_pool} if policy.kind == "empirical" else {}
            )
            observed = apply_mask_policy(
                data,
                policy,
                base_seed=base_seed,
                replicate=replicate,
                **kwargs,
            )
            with torch.no_grad():
                logits = model(
                    values_tensor,
                    torch.tensor(observed, dtype=torch.bool, device=device),
                    reference,
                )
            frame = data.loc[:, ["sample_id", "site", "target", "record_sha256"]].copy()
            frame["policy"] = policy.name
            frame["mask_replicate"] = replicate
            frame["evidence_logit"] = logits.detach().cpu().numpy()
            frame["y_score"] = torch.sigmoid(logits).detach().cpu().numpy()
            frame["observed_fraction"] = observed.mean(axis=1)
            for feature, output_column in zip(CORE_COLUMNS, CORE_PREDICTION_COLUMNS, strict=True):
                frame[output_column] = data[feature].to_numpy(dtype=np.float64)
            for feature_index, output_column in enumerate(MASK_PREDICTION_COLUMNS):
                frame[output_column] = observed[:, feature_index]
            bit_weights = np.left_shift(
                np.int64(1), np.arange(len(FEATURE_COLUMNS), dtype=np.int64)
            )
            frame["observed_mask_code"] = observed.astype(np.int64) @ bit_weights
            records.append(frame)
    return pd.concat(records, ignore_index=True)


def policy_selection_score(predictions: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    metric_records = []
    for (policy, replicate), group in predictions.groupby(["policy", "mask_replicate"]):
        metric_records.append(
            {
                "policy": policy,
                "mask_replicate": replicate,
                **binary_metrics(group["target"], group["y_score"]),
            }
        )
    metrics = pd.DataFrame(metric_records)
    score = 0.5 * (
        float(metrics["balanced_log_loss"].mean()) + float(metrics["balanced_log_loss"].max())
    )
    return score, metrics


def _set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def fit_ps_maskdro(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    variant: str,
    parameters: dict[str, Any],
    seed: int,
    device: str,
) -> FitResult:
    """Fit one source-only fold and early-stop on a disjoint source hospital."""
    _set_reproducibility(seed)
    torch_device = torch.device(device)
    preprocessor = NeuralPreprocessor.fit(training)
    train_values, train_natural = preprocessor.transform(training)
    model = ObservedFeatureSetEncoder(
        n_features=len(FEATURE_COLUMNS),
        continuous_indices=preprocessor.continuous_indices,
        categorical_cardinalities=preprocessor.categorical_cardinalities,
        d_model=int(parameters.get("d_model", 64)),
        n_heads=int(parameters.get("n_heads", 4)),
        n_layers=int(parameters.get("n_layers", 2)),
        dropout=float(parameters.get("dropout", 0.1)),
        ane=variant == "v7",
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters.get("learning_rate", 0.001)),
        weight_decay=float(parameters.get("weight_decay", 0.001)),
    )
    labels = torch.tensor(training["target"].to_numpy(), dtype=torch.long, device=torch_device)
    site_values = pd.Categorical(training["site"]).codes
    site_codes = torch.tensor(site_values, dtype=torch.long, device=torch_device)
    values = torch.tensor(train_values, dtype=torch.float32, device=torch_device)
    reference = torch.tensor(
        preprocessor.reference_values,
        dtype=torch.float32,
        device=torch_device,
    )
    policies = policies_for_variant(variant)
    if variant == "mirrams":
        policies = (
            policies[0],
            MaskPolicy(
                "mirrams_mcar",
                "mcar",
                rate=float(parameters.get("mirrams_mask_rate", 0.2)),
            ),
        )
    validation_policies = default_policy_bank()
    max_epochs = int(parameters.get("max_epochs", 200))
    evaluation_interval = int(parameters.get("evaluation_interval", 5))
    patience_evaluations = int(parameters.get("patience_evaluations", 12))
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    evaluations_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        masks = _policy_masks(
            training,
            policies,
            base_seed=seed,
            replicate=epoch,
            empirical_mask_pool=train_natural,
        )
        n_policies, n_patients, _ = masks.shape
        expanded_values = values.repeat(n_policies, 1)
        expanded_masks = torch.tensor(
            masks.reshape(n_policies * n_patients, -1),
            dtype=torch.bool,
            device=torch_device,
        )
        logits = model(expanded_values, expanded_masks, reference).reshape(n_policies, n_patients)
        if variant == "mirrams":
            objective, components = mirrams_objective(
                logits,
                labels,
                lambda_supervised_mask=float(parameters.get("mirrams_lambda_1", 15.0)),
                lambda_consistency=float(parameters.get("mirrams_lambda_2", 15.0)),
                confidence_threshold=float(parameters.get("mirrams_confidence", 0.95)),
            )
        else:
            objective, components = ps_maskdro_objective(
                logits,
                labels,
                site_codes,
                variant=variant,
                dro_lambda=float(parameters.get("dro_lambda", 0.5)),
                dro_tau=float(parameters.get("dro_tau", 0.2)),
                brier_beta=float(parameters.get("brier_beta", 0.1)),
            )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % evaluation_interval != 0 and epoch != max_epochs:
            continue
        validation_predictions = predict_policy_bank(
            model,
            preprocessor,
            validation,
            validation_policies,
            device=torch_device,
            base_seed=seed + 100_000,
            replicates=int(parameters.get("validation_mask_replicates", 2)),
            empirical_mask_pool=train_natural,
        )
        validation_score, _ = policy_selection_score(validation_predictions)
        history.append(
            {
                "epoch": float(epoch),
                "objective": float(objective.detach()),
                "validation_score": validation_score,
                **components,
            }
        )
        if validation_score < best_score - 1e-5:
            best_score = validation_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
        if evaluations_without_improvement >= patience_evaluations:
            break

    if best_state is None:
        raise RuntimeError("PS-MaskDRO training produced no valid checkpoint")
    model.load_state_dict(best_state)
    return FitResult(
        model=model,
        preprocessor=preprocessor,
        best_epoch=best_epoch,
        validation_score=best_score,
        history=history,
        variant=variant,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        device=str(torch_device),
    )


def fit_ps_maskdro_fixed_epochs(
    training: pd.DataFrame,
    *,
    variant: str,
    parameters: dict[str, Any],
    seed: int,
    device: str,
    epochs: int,
) -> FitResult:
    """Refit a frozen candidate without consulting any validation or target labels."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if variant not in VALID_VARIANTS:
        raise KeyError(f"Unknown PS-MaskDRO variant: {variant}")
    _set_reproducibility(seed)
    torch_device = torch.device(device)
    preprocessor = NeuralPreprocessor.fit(training)
    train_values, train_natural = preprocessor.transform(training)
    model = ObservedFeatureSetEncoder(
        n_features=len(FEATURE_COLUMNS),
        continuous_indices=preprocessor.continuous_indices,
        categorical_cardinalities=preprocessor.categorical_cardinalities,
        d_model=int(parameters.get("d_model", 64)),
        n_heads=int(parameters.get("n_heads", 4)),
        n_layers=int(parameters.get("n_layers", 2)),
        dropout=float(parameters.get("dropout", 0.1)),
        ane=variant == "v7",
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters.get("learning_rate", 0.001)),
        weight_decay=float(parameters.get("weight_decay", 0.001)),
    )
    labels = torch.tensor(training["target"].to_numpy(), dtype=torch.long, device=torch_device)
    site_codes = torch.tensor(
        pd.Categorical(training["site"]).codes,
        dtype=torch.long,
        device=torch_device,
    )
    values = torch.tensor(train_values, dtype=torch.float32, device=torch_device)
    reference = torch.tensor(
        preprocessor.reference_values,
        dtype=torch.float32,
        device=torch_device,
    )
    policies = policies_for_variant(variant)
    if variant == "mirrams":
        policies = (
            policies[0],
            MaskPolicy(
                "mirrams_mcar",
                "mcar",
                rate=float(parameters.get("mirrams_mask_rate", 0.2)),
            ),
        )
    history: list[dict[str, float]] = []
    evaluation_interval = int(parameters.get("evaluation_interval", 5))

    for epoch in range(1, epochs + 1):
        model.train()
        masks = _policy_masks(
            training,
            policies,
            base_seed=seed,
            replicate=epoch,
            empirical_mask_pool=train_natural,
        )
        n_policies, n_patients, _ = masks.shape
        logits = model(
            values.repeat(n_policies, 1),
            torch.tensor(
                masks.reshape(n_policies * n_patients, -1),
                dtype=torch.bool,
                device=torch_device,
            ),
            reference,
        ).reshape(n_policies, n_patients)
        if variant == "mirrams":
            objective, components = mirrams_objective(
                logits,
                labels,
                lambda_supervised_mask=float(parameters.get("mirrams_lambda_1", 15.0)),
                lambda_consistency=float(parameters.get("mirrams_lambda_2", 15.0)),
                confidence_threshold=float(parameters.get("mirrams_confidence", 0.95)),
            )
        else:
            objective, components = ps_maskdro_objective(
                logits,
                labels,
                site_codes,
                variant=variant,
                dro_lambda=float(parameters.get("dro_lambda", 0.5)),
                dro_tau=float(parameters.get("dro_tau", 0.2)),
                brier_beta=float(parameters.get("brier_beta", 0.1)),
            )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch % evaluation_interval == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "objective": float(objective.detach()),
                    **components,
                }
            )

    return FitResult(
        model=model,
        preprocessor=preprocessor,
        best_epoch=epochs,
        validation_score=float("nan"),
        history=history,
        variant=variant,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        device=str(torch_device),
    )
