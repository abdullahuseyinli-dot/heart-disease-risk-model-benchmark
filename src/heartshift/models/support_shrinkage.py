"""Source-only support-aware shrinkage across fixed probability experts.

The router is intentionally prediction-level: it cannot access hospital identity
or target labels at inference. It learns, from source-held-out episodes, when to
shrink a high-variance evidence expert toward stable classical/robust anchors.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from heartshift.models.ps_maskdro import _set_reproducibility

PROBABILITY_EPSILON = 1e-6


@dataclass(frozen=True)
class SupportReference:
    """Frozen source support and feature standardization state."""

    unique_masks: np.ndarray
    mask_counts: np.ndarray
    location: np.ndarray
    scale: np.ndarray


@dataclass
class SupportAwareFitResult:
    model: SupportAwareRouter
    support_reference: SupportReference
    expert_names: tuple[str, ...]
    best_epoch: int
    validation_score: float
    history: list[dict[str, float]]
    parameter_count: int
    device: str


class SupportAwareRouter(nn.Module):
    """Predict simplex weights used to mix fixed expert logits."""

    def __init__(self, n_support_features: int, n_experts: int, hidden_dim: int, dropout: float):
        super().__init__()
        if n_support_features < 1 or n_experts < 2:
            raise ValueError("The support router requires features and at least two experts")
        self.network = nn.Sequential(
            nn.Linear(n_support_features, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(
        self,
        support_features: torch.Tensor,
        expert_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if expert_logits.ndim != 2 or support_features.ndim != 2:
            raise ValueError("Router inputs must be two-dimensional")
        if len(expert_logits) != len(support_features):
            raise ValueError("Expert and support rows must align")
        weights = torch.softmax(self.network(support_features), dim=1)
        mixed_logit = torch.sum(weights * expert_logits, dim=1)
        return mixed_logit, weights


def _binary_masks(values: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isin(matrix, (0, 1, False, True)).all():
        raise ValueError(f"{name} must contain only binary values")
    return matrix.astype(bool, copy=False)


def _probability_matrix(values: Any) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
        raise ValueError("Expert probabilities require at least two experts")
    if (
        not np.isfinite(probabilities).all()
        or not ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    ):
        raise ValueError("Expert probabilities must be finite and lie in [0, 1]")
    return np.clip(probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def fit_support_reference(
    probabilities: Any,
    observed_masks: Any,
    extra_features: Any | None = None,
) -> tuple[SupportReference, np.ndarray]:
    """Fit source-only mask support and standardization, returning training features."""
    expert_probability = _probability_matrix(probabilities)
    masks = _binary_masks(observed_masks, name="observed_masks")
    if len(expert_probability) != len(masks):
        raise ValueError("Expert probabilities and observed masks must align")
    unique_masks, inverse, mask_counts = np.unique(
        masks, axis=0, return_inverse=True, return_counts=True
    )
    features = _raw_support_features(
        expert_probability,
        masks,
        unique_masks,
        mask_counts,
        exact_indices=inverse,
        extra_features=extra_features,
    )
    location = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    reference = SupportReference(
        unique_masks=unique_masks,
        mask_counts=mask_counts.astype(np.int64),
        location=location,
        scale=scale,
    )
    return reference, (features - location) / scale


def transform_support_features(
    reference: SupportReference,
    probabilities: Any,
    observed_masks: Any,
    extra_features: Any | None = None,
) -> np.ndarray:
    """Create inference-time features using only frozen source support."""
    expert_probability = _probability_matrix(probabilities)
    masks = _binary_masks(observed_masks, name="observed_masks")
    if len(expert_probability) != len(masks):
        raise ValueError("Expert probabilities and observed masks must align")
    exact_lookup = {row.tobytes(): index for index, row in enumerate(reference.unique_masks)}
    exact_indices = np.asarray(
        [exact_lookup.get(row.tobytes(), -1) for row in masks], dtype=np.int64
    )
    features = _raw_support_features(
        expert_probability,
        masks,
        reference.unique_masks,
        reference.mask_counts,
        exact_indices=exact_indices,
        extra_features=extra_features,
    )
    if features.shape[1] != len(reference.location):
        raise ValueError("Inference support-feature schema differs from training")
    return np.asarray((features - reference.location) / reference.scale, dtype=np.float64)


def _raw_support_features(
    probabilities: np.ndarray,
    masks: np.ndarray,
    unique_masks: np.ndarray,
    mask_counts: np.ndarray,
    *,
    exact_indices: np.ndarray,
    extra_features: Any | None,
) -> np.ndarray:
    nearest_parts = []
    for start in range(0, len(masks), 512):
        chunk = masks[start : start + 512]
        distance = np.count_nonzero(chunk[:, None, :] != unique_masks[None, :, :], axis=2)
        nearest_parts.append(distance.min(axis=1) / masks.shape[1])
    nearest = np.concatenate(nearest_parts)
    total = int(mask_counts.sum())
    pattern_count = np.asarray(
        [mask_counts[index] if index >= 0 else 0 for index in exact_indices],
        dtype=np.float64,
    )
    rarity = -np.log((pattern_count + 1.0) / (total + len(mask_counts)))
    exact = exact_indices >= 0
    expert_logits = np.log(probabilities / (1.0 - probabilities))
    mean_probability = probabilities.mean(axis=1)
    entropy = -(
        mean_probability * np.log(np.clip(mean_probability, PROBABILITY_EPSILON, 1.0))
        + (1.0 - mean_probability)
        * np.log(np.clip(1.0 - mean_probability, PROBABILITY_EPSILON, 1.0))
    )
    columns = [
        masks.astype(np.float64),
        masks.mean(axis=1, keepdims=True),
        nearest[:, None],
        exact.astype(np.float64)[:, None],
        rarity[:, None],
        expert_logits.std(axis=1, keepdims=True),
        np.ptp(expert_logits, axis=1, keepdims=True),
        entropy[:, None],
    ]
    if extra_features is not None:
        extra = np.asarray(extra_features, dtype=np.float64)
        if extra.ndim == 1:
            extra = extra[:, None]
        if extra.ndim != 2 or len(extra) != len(masks) or not np.isfinite(extra).all():
            raise ValueError("Extra support features must be a finite aligned matrix")
        columns.append(extra)
    return np.concatenate(columns, axis=1)


def _group_risks(
    losses: torch.Tensor,
    group_codes: torch.Tensor,
) -> torch.Tensor:
    risks = []
    for group in torch.unique(group_codes, sorted=True):
        selected = group_codes.eq(group)
        if selected.any():
            risks.append(losses[selected].mean())
    if not risks:
        raise ValueError("At least one robust group is required")
    return torch.stack(risks)


def support_router_objective(
    mixed_logits: torch.Tensor,
    expert_logits: torch.Tensor,
    labels: torch.Tensor,
    group_codes: torch.Tensor,
    natural: torch.Tensor,
    weights: torch.Tensor,
    *,
    dro_lambda: float,
    dro_tau: float,
    regret_beta: float,
    natural_margin: float,
    entropy_bonus: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Robust meta-objective with oracle-regret and natural-risk constraints."""
    if dro_tau <= 0.0 or not 0.0 <= dro_lambda <= 1.0:
        raise ValueError("DRO parameters are invalid")
    losses = functional.binary_cross_entropy_with_logits(
        mixed_logits, labels.float(), reduction="none"
    )
    risks = _group_risks(losses, group_codes)
    mean_risk = risks.mean()
    robust_risk = dro_tau * (torch.logsumexp(risks / dro_tau, dim=0) - math.log(len(risks)))
    expert_losses = functional.binary_cross_entropy_with_logits(
        expert_logits,
        labels.float().unsqueeze(1).expand_as(expert_logits),
        reduction="none",
    )
    expert_robust = []
    for expert_index in range(expert_logits.shape[1]):
        expert_risks = _group_risks(expert_losses[:, expert_index], group_codes)
        expert_robust.append(
            dro_tau * (torch.logsumexp(expert_risks / dro_tau, dim=0) - math.log(len(expert_risks)))
        )
    best_expert_robust = torch.stack(expert_robust).min().detach()
    regret = torch.relu(robust_risk - best_expert_robust)
    if natural.any():
        natural_loss = losses[natural].mean()
        best_natural = expert_losses[natural].mean(dim=0).min().detach()
        natural_penalty = torch.relu(natural_loss - best_natural - natural_margin)
    else:
        natural_loss = mixed_logits.sum() * 0.0
        natural_penalty = mixed_logits.sum() * 0.0
    routing_entropy = -torch.sum(weights * torch.log(torch.clamp(weights, min=1e-8)), dim=1).mean()
    objective = (
        (1.0 - dro_lambda) * mean_risk
        + dro_lambda * robust_risk
        + regret_beta * regret
        + regret_beta * natural_penalty
        - entropy_bonus * routing_entropy
    )
    return objective, {
        "mean_group_risk": float(mean_risk.detach()),
        "robust_group_risk": float(robust_risk.detach()),
        "best_expert_robust_risk": float(best_expert_robust),
        "robust_regret": float(regret.detach()),
        "natural_risk": float(natural_loss.detach()),
        "natural_noninferiority_penalty": float(natural_penalty.detach()),
        "routing_entropy": float(routing_entropy.detach()),
    }


def _evaluation_score(
    model: SupportAwareRouter,
    support_features: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    group_codes: np.ndarray,
    *,
    device: torch.device,
    dro_tau: float,
) -> float:
    model.eval()
    logits = np.log(probabilities / (1.0 - probabilities))
    with torch.no_grad():
        mixed, _ = model(
            torch.tensor(support_features, dtype=torch.float32, device=device),
            torch.tensor(logits, dtype=torch.float32, device=device),
        )
        losses = functional.binary_cross_entropy_with_logits(
            mixed,
            torch.tensor(labels, dtype=torch.float32, device=device),
            reduction="none",
        )
        risks = _group_risks(losses, torch.tensor(group_codes, dtype=torch.long, device=device))
        robust = dro_tau * (torch.logsumexp(risks / dro_tau, dim=0) - math.log(len(risks)))
    return float(robust.cpu())


def fit_support_aware_router(
    training_probabilities: Any,
    training_masks: Any,
    training_labels: Any,
    training_group_codes: Any,
    training_natural: Any,
    validation_probabilities: Any,
    validation_masks: Any,
    validation_labels: Any,
    validation_group_codes: Any,
    *,
    expert_names: tuple[str, ...],
    parameters: dict[str, Any],
    seed: int,
    device: str,
    training_extra_features: Any | None = None,
    validation_extra_features: Any | None = None,
) -> SupportAwareFitResult:
    """Fit the router on source OOF predictions and source support only."""
    _set_reproducibility(seed)
    training_probability = _probability_matrix(training_probabilities)
    validation_probability = _probability_matrix(validation_probabilities)
    if training_probability.shape[1] != len(expert_names):
        raise ValueError("Expert names do not match training probability columns")
    if validation_probability.shape[1] != len(expert_names):
        raise ValueError("Training and validation expert schemas differ")
    training_label = np.asarray(training_labels, dtype=np.int64)
    validation_label = np.asarray(validation_labels, dtype=np.int64)
    training_group = np.asarray(training_group_codes, dtype=np.int64)
    validation_group = np.asarray(validation_group_codes, dtype=np.int64)
    training_natural_array = np.asarray(training_natural, dtype=bool)
    if not (
        len(training_probability)
        == len(training_label)
        == len(training_group)
        == len(training_natural_array)
    ):
        raise ValueError("Training router arrays do not align")
    if not (len(validation_probability) == len(validation_label) == len(validation_group)):
        raise ValueError("Validation router arrays do not align")
    if set(np.unique(training_label)) != {0, 1} or set(np.unique(validation_label)) != {0, 1}:
        raise ValueError("Training and validation require both outcome classes")
    reference, training_support = fit_support_reference(
        training_probability, training_masks, training_extra_features
    )
    validation_support = transform_support_features(
        reference,
        validation_probability,
        validation_masks,
        validation_extra_features,
    )
    torch_device = torch.device(device)
    model = SupportAwareRouter(
        n_support_features=training_support.shape[1],
        n_experts=len(expert_names),
        hidden_dim=int(parameters.get("hidden_dim", 32)),
        dropout=float(parameters.get("dropout", 0.05)),
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters.get("learning_rate", 0.001)),
        weight_decay=float(parameters.get("weight_decay", 0.001)),
    )
    max_epochs = int(parameters.get("max_epochs", 500))
    patience = int(parameters.get("patience", 50))
    dro_tau = float(parameters.get("dro_tau", 0.2))
    train_support_tensor = torch.tensor(training_support, dtype=torch.float32, device=torch_device)
    train_expert_tensor = torch.tensor(
        np.log(training_probability / (1.0 - training_probability)),
        dtype=torch.float32,
        device=torch_device,
    )
    train_labels_tensor = torch.tensor(training_label, dtype=torch.long, device=torch_device)
    train_groups_tensor = torch.tensor(training_group, dtype=torch.long, device=torch_device)
    train_natural_tensor = torch.tensor(
        training_natural_array, dtype=torch.bool, device=torch_device
    )
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        mixed, weights = model(train_support_tensor, train_expert_tensor)
        objective, components = support_router_objective(
            mixed,
            train_expert_tensor,
            train_labels_tensor,
            train_groups_tensor,
            train_natural_tensor,
            weights,
            dro_lambda=float(parameters.get("dro_lambda", 0.5)),
            dro_tau=dro_tau,
            regret_beta=float(parameters.get("regret_beta", 0.5)),
            natural_margin=float(parameters.get("natural_margin", 0.01)),
            entropy_bonus=float(parameters.get("entropy_bonus", 0.001)),
        )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        score = _evaluation_score(
            model,
            validation_support,
            validation_probability,
            validation_label,
            validation_group,
            device=torch_device,
            dro_tau=dro_tau,
        )
        if epoch == 1 or epoch % 10 == 0:
            history.append(
                {
                    "epoch": float(epoch),
                    "objective": float(objective.detach()),
                    "validation_robust_risk": score,
                    **components,
                }
            )
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Support-aware router produced no finite checkpoint")
    model.load_state_dict(best_state)
    return SupportAwareFitResult(
        model=model,
        support_reference=reference,
        expert_names=expert_names,
        best_epoch=best_epoch,
        validation_score=best_score,
        history=history,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        device=str(torch_device),
    )


def predict_support_aware_router(
    result: SupportAwareFitResult,
    probabilities: Any,
    observed_masks: Any,
    *,
    extra_features: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mixed probabilities and auditable per-expert routing weights."""
    expert_probability = _probability_matrix(probabilities)
    if expert_probability.shape[1] != len(result.expert_names):
        raise ValueError("Inference expert schema differs from the fitted router")
    support = transform_support_features(
        result.support_reference,
        expert_probability,
        observed_masks,
        extra_features,
    )
    logits = np.log(expert_probability / (1.0 - expert_probability))
    device = torch.device(result.device)
    result.model.eval()
    with torch.no_grad():
        mixed, weights = result.model(
            torch.tensor(support, dtype=torch.float32, device=device),
            torch.tensor(logits, dtype=torch.float32, device=device),
        )
    probability = torch.sigmoid(mixed).cpu().numpy()
    return probability, weights.cpu().numpy()
