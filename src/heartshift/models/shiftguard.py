"""Testability-trained, set-valued prior adaptation under observable shift.

ShiftGuard does not claim to detect arbitrary concept shift. It learns a compact
diagnostic representation that has power against observable class-conditional and
acquisition changes while retaining an explicit no-adaptation outcome.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.spatial.distance import pdist
from scipy.special import expit, logit
from torch import nn

from heartshift.models.ps_maskdro import _set_reproducibility


@dataclass(frozen=True)
class ShiftEpisode:
    target_features: np.ndarray
    true_prevalence: float
    valid_label_shift: bool
    mechanism: str


@dataclass(frozen=True)
class ShiftGuardCalibration:
    prior_grid: np.ndarray
    critical_values: np.ndarray
    alpha: float
    target_size: int
    repetitions_per_prior: int
    discrepancy_reducer: str = "mean_square"
    discrepancy_regularization: float = 0.05
    precision_matrices: np.ndarray | None = None
    null_discrepancies: np.ndarray | None = None


@dataclass(frozen=True)
class JointShiftGuardCalibration:
    """Joint null calibration for correlated diagnostic representations."""

    view_names: tuple[str, ...]
    prior_grid: np.ndarray
    view_scales: np.ndarray
    critical_values: np.ndarray
    alpha: float
    target_size: int
    repetitions_per_prior: int


@dataclass(frozen=True)
class PrevalenceSet:
    accepted: bool
    accepted_priors: tuple[float, ...]
    lower: float | None
    upper: float | None
    best_prior: float
    best_discrepancy: float
    discrepancies: np.ndarray


@dataclass
class ShiftGuardFitResult:
    model: ShiftDiagnosticEncoder
    location: np.ndarray
    scale: np.ndarray
    best_epoch: int
    validation_score: float
    history: list[dict[str, float]]
    parameter_count: int
    device: str


@dataclass(frozen=True)
class RandomFourierDiagnostic:
    """Frozen Gaussian-kernel mean embedding fitted from source features only."""

    location: np.ndarray
    scale: np.ndarray
    frequencies: np.ndarray
    phases: np.ndarray
    bandwidth: float
    include_linear: bool

    def transform(self, features: Any) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.location):
            raise ValueError("RFF diagnostic feature schema differs from fitting")
        if not np.isfinite(matrix).all():
            raise ValueError("RFF diagnostic features must be finite")
        standardized = (matrix - self.location) / self.scale
        projection = standardized @ self.frequencies + self.phases
        rff = np.sqrt(2.0 / len(self.phases)) * np.cos(projection)
        if self.include_linear:
            return np.concatenate(
                [standardized / np.sqrt(standardized.shape[1]), rff], axis=1
            )
        return np.asarray(rff, dtype=np.float64)


@dataclass(frozen=True)
class PolynomialMomentDiagnostic:
    """Source-standardized linear and second-order diagnostic moments."""

    location: np.ndarray
    scale: np.ndarray
    moment_location: np.ndarray
    moment_scale: np.ndarray
    pair_rows: np.ndarray
    pair_columns: np.ndarray

    def transform(self, features: Any) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.location):
            raise ValueError("Moment diagnostic feature schema differs from fitting")
        if not np.isfinite(matrix).all():
            raise ValueError("Moment diagnostic features must be finite")
        standardized = (matrix - self.location) / self.scale
        moments = standardized[:, self.pair_rows] * standardized[:, self.pair_columns]
        moments = (moments - self.moment_location) / self.moment_scale
        return np.concatenate(
            [
                standardized / np.sqrt(standardized.shape[1]),
                moments / np.sqrt(moments.shape[1]),
            ],
            axis=1,
        )


@dataclass(frozen=True)
class QuantileCopulaDiagnostic:
    """Source-fitted marginal CDF and pairwise copula indicator features."""

    continuous_feature_count: int
    univariate_thresholds: np.ndarray
    pair_rows: np.ndarray
    pair_columns: np.ndarray
    pair_left_thresholds: np.ndarray
    pair_right_thresholds: np.ndarray
    bank_location: np.ndarray
    bank_scale: np.ndarray

    def _raw_bank(self, matrix: np.ndarray) -> np.ndarray:
        continuous = matrix[:, : self.continuous_feature_count]
        univariate = (
            continuous[:, :, None] <= self.univariate_thresholds[None, :, :]
        ).reshape(len(matrix), -1)
        left = (
            continuous[:, self.pair_rows, None, None]
            <= self.pair_left_thresholds[None, :, :, None]
        )
        right = (
            continuous[:, self.pair_columns, None, None]
            <= self.pair_right_thresholds[None, :, None, :]
        )
        pairwise = (left & right).reshape(len(matrix), -1)
        remainder = matrix[:, self.continuous_feature_count :]
        return np.concatenate([univariate, pairwise, remainder], axis=1).astype(
            np.float64
        )

    def transform(self, features: Any) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        expected_features = self.continuous_feature_count + (
            len(self.bank_location)
            - self.continuous_feature_count * self.univariate_thresholds.shape[1]
            - len(self.pair_rows)
            * self.pair_left_thresholds.shape[1]
            * self.pair_right_thresholds.shape[1]
        )
        if matrix.ndim != 2 or matrix.shape[1] != expected_features:
            raise ValueError("Quantile-copula diagnostic feature schema differs from fitting")
        if not np.isfinite(matrix).all():
            raise ValueError("Quantile-copula diagnostic features must be finite")
        bank = self._raw_bank(matrix)
        return np.asarray(
            (bank - self.bank_location) / self.bank_scale,
            dtype=np.float64,
        )


class ShiftDiagnosticEncoder(nn.Module):
    """Low-dimensional diagnostic representation with a predictive anchor."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_features < 1 or embedding_dim < 2:
            raise ValueError("ShiftGuard requires input features and a multivariate embedding")
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(features)
        return embedding, self.classifier(embedding).squeeze(1)


def compose_diagnostic_features(
    evidence: Any,
    core: Any,
    observed_mask: Any,
    representation: Any | None = None,
) -> np.ndarray:
    """Compose aligned evidence, stable-core, mask, and optional learned views."""
    views = []
    expected_rows: int | None = None
    for name, values in (
        ("evidence", evidence),
        ("core", core),
        ("observed_mask", observed_mask),
        ("representation", representation),
    ):
        if values is None:
            continue
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty matrix")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} must contain only finite values")
        if expected_rows is None:
            expected_rows = len(matrix)
        elif len(matrix) != expected_rows:
            raise ValueError("Diagnostic views do not align")
        views.append(matrix)
    if len(views) < 3:
        raise ValueError("Evidence, core, and mask views are required")
    return np.concatenate(views, axis=1)


def fit_random_fourier_diagnostic(
    features: Any,
    *,
    n_components: int,
    seed: int,
    include_linear: bool = True,
    bandwidth_sample_size: int = 512,
) -> RandomFourierDiagnostic:
    """Fit a deterministic source-only RBF feature map with median bandwidth.

    Mean differences in this feature map approximate maximum mean discrepancy,
    so class-conditional scale and multimodal changes can be visible even when
    the original feature mean lies on the label-shift mixture line.
    """
    if n_components < 2 or bandwidth_sample_size < 2:
        raise ValueError("RFF component and bandwidth sample counts must exceed one")
    location, scale, standardized = _standardizer(np.asarray(features, dtype=np.float64))
    rng = np.random.default_rng(seed)
    sample_size = min(len(standardized), bandwidth_sample_size)
    sample = standardized[rng.choice(len(standardized), size=sample_size, replace=False)]
    distances = pdist(sample, metric="euclidean")
    positive_distances = distances[distances > 1e-8]
    bandwidth = float(np.median(positive_distances)) if len(positive_distances) else 1.0
    frequencies = rng.normal(
        0.0,
        1.0 / bandwidth,
        size=(standardized.shape[1], n_components),
    )
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    return RandomFourierDiagnostic(
        location=location,
        scale=scale,
        frequencies=frequencies,
        phases=phases,
        bandwidth=bandwidth,
        include_linear=include_linear,
    )


def fit_polynomial_moment_diagnostic(features: Any) -> PolynomialMomentDiagnostic:
    """Fit a finite omnibus view with explicit variances and covariances."""
    location, scale, standardized = _standardizer(np.asarray(features, dtype=np.float64))
    pair_rows, pair_columns = np.triu_indices(standardized.shape[1])
    moments = standardized[:, pair_rows] * standardized[:, pair_columns]
    moment_location = moments.mean(axis=0)
    moment_scale = moments.std(axis=0)
    moment_scale = np.where(moment_scale > 1e-8, moment_scale, 1.0)
    return PolynomialMomentDiagnostic(
        location=location,
        scale=scale,
        moment_location=moment_location,
        moment_scale=moment_scale,
        pair_rows=pair_rows,
        pair_columns=pair_columns,
    )


def fit_quantile_copula_diagnostic(
    features: Any,
    *,
    continuous_feature_count: int,
    univariate_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    pair_levels: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> QuantileCopulaDiagnostic:
    """Fit a deterministic low-order empirical-CDF feature bank on source data."""
    matrix = np.asarray(features, dtype=np.float64)
    if (
        matrix.ndim != 2
        or len(matrix) == 0
        or not np.isfinite(matrix).all()
        or not 2 <= continuous_feature_count <= matrix.shape[1]
    ):
        raise ValueError("Quantile-copula fitting inputs are invalid")
    if (
        not univariate_levels
        or not pair_levels
        or any(not 0.0 < value < 1.0 for value in (*univariate_levels, *pair_levels))
    ):
        raise ValueError("Quantile-copula levels must lie inside (0, 1)")
    continuous = matrix[:, :continuous_feature_count]
    univariate_thresholds = np.quantile(
        continuous, np.asarray(univariate_levels), axis=0
    ).T
    pair_rows, pair_columns = np.triu_indices(continuous_feature_count, k=1)
    pair_quantiles = np.quantile(continuous, np.asarray(pair_levels), axis=0).T
    diagnostic = QuantileCopulaDiagnostic(
        continuous_feature_count=continuous_feature_count,
        univariate_thresholds=np.asarray(univariate_thresholds, dtype=np.float64),
        pair_rows=pair_rows,
        pair_columns=pair_columns,
        pair_left_thresholds=np.asarray(pair_quantiles[pair_rows], dtype=np.float64),
        pair_right_thresholds=np.asarray(pair_quantiles[pair_columns], dtype=np.float64),
        bank_location=np.empty(0, dtype=np.float64),
        bank_scale=np.empty(0, dtype=np.float64),
    )
    bank = diagnostic._raw_bank(matrix)
    location = bank.mean(axis=0)
    scale = bank.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return QuantileCopulaDiagnostic(
        continuous_feature_count=continuous_feature_count,
        univariate_thresholds=diagnostic.univariate_thresholds,
        pair_rows=pair_rows,
        pair_columns=pair_columns,
        pair_left_thresholds=diagnostic.pair_left_thresholds,
        pair_right_thresholds=diagnostic.pair_right_thresholds,
        bank_location=location,
        bank_scale=scale,
    )


def _standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) == 0 or not np.isfinite(matrix).all():
        raise ValueError("Diagnostic features must be a finite non-empty matrix")
    location = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return location, scale, (matrix - location) / scale


def _draw_label_shift_target(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    *,
    prevalence: float,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    target_labels = rng.random(size) < prevalence
    target = np.empty((size, source_features.shape[1]), dtype=np.float64)
    for label in (0, 1):
        selected = target_labels == label
        pool = source_features[source_labels == label]
        if len(pool) < 2:
            raise ValueError("Each source class needs at least two diagnostic examples")
        target[selected] = pool[rng.integers(0, len(pool), size=int(selected.sum()))]
    return target, target_labels.astype(np.int64)


def generate_shift_episodes(
    source_features: Any,
    source_labels: Any,
    *,
    episode_count: int,
    target_size: int,
    valid_fraction: float,
    seed: int,
    invalid_mechanisms: tuple[str, ...] = (
        "conditional_translation",
        "outcome_dependent_dropout",
        "support_translation",
        "conditional_scale",
    ),
    mask_feature_indices: tuple[int, ...] = (),
) -> tuple[ShiftEpisode, ...]:
    """Generate source-only valid and observable-invalid pseudo-target episodes."""
    features = np.asarray(source_features, dtype=np.float64)
    labels = np.asarray(source_labels, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(labels) or not np.isfinite(features).all():
        raise ValueError("Source diagnostic features and labels must align and be finite")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Episode generation requires both source classes")
    if episode_count < 2 or target_size < 16 or not 0.0 < valid_fraction < 1.0:
        raise ValueError("Episode count, target size, or valid fraction is invalid")
    if not invalid_mechanisms:
        raise ValueError("At least one observable-invalid mechanism is required")
    rng = np.random.default_rng(seed)
    episodes = []
    valid_count = round(episode_count * valid_fraction)
    valid_count = min(max(valid_count, 1), episode_count - 1)
    feature_scale = np.std(features, axis=0)
    feature_scale = np.where(feature_scale > 1e-6, feature_scale, 1.0)
    for index in range(episode_count):
        prevalence = float(rng.uniform(0.1, 0.9))
        target, target_labels = _draw_label_shift_target(
            features,
            labels,
            prevalence=prevalence,
            size=target_size,
            rng=rng,
        )
        valid = index < valid_count
        mechanism = "pure_label_shift"
        if not valid:
            mechanism = invalid_mechanisms[(index - valid_count) % len(invalid_mechanisms)]
            column = int(rng.integers(0, features.shape[1]))
            strength = float(rng.uniform(0.75, 1.5))
            if mechanism == "conditional_translation":
                target[target_labels == 1, column] += strength * feature_scale[column]
            elif mechanism == "outcome_dependent_dropout":
                candidate_columns = mask_feature_indices or tuple(range(features.shape[1]))
                selected_column = int(candidate_columns[index % len(candidate_columns)])
                affected = (target_labels == 1) & (rng.random(target_size) < 0.75)
                target[affected, selected_column] = 0.0
            elif mechanism == "support_translation":
                target[:, column] += strength * 1.5 * feature_scale[column]
            elif mechanism == "conditional_scale":
                center = float(features[labels == 0, column].mean())
                selected = target_labels == 0
                target[selected, column] = center + (target[selected, column] - center) * (
                    1.0 + strength
                )
            else:
                raise KeyError(f"Unknown invalid shift mechanism: {mechanism}")
        episodes.append(
            ShiftEpisode(
                target_features=target,
                true_prevalence=float(target_labels.mean()),
                valid_label_shift=valid,
                mechanism=mechanism,
            )
        )
    rng.shuffle(episodes)
    return tuple(episodes)


def _class_means(
    embedding: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if set(labels.detach().cpu().unique().tolist()) != {0, 1}:
        raise ValueError("Diagnostic source embedding requires both classes")
    return embedding[labels.eq(0)].mean(dim=0), embedding[labels.eq(1)].mean(dim=0)


def _minimum_mixture_discrepancy(
    negative_mean: torch.Tensor,
    positive_mean: torch.Tensor,
    target_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    direction = positive_mean - negative_mean
    prevalence = torch.clamp(
        torch.dot(target_mean - negative_mean, direction)
        / (torch.dot(direction, direction) + 1e-8),
        0.01,
        0.99,
    )
    residual = target_mean - (negative_mean + prevalence * direction)
    return prevalence, torch.mean(torch.square(residual))


def _whitening_loss(embedding: torch.Tensor) -> torch.Tensor:
    centered = embedding - embedding.mean(dim=0)
    covariance = centered.T @ centered / max(len(embedding) - 1, 1)
    identity = torch.eye(covariance.shape[0], device=embedding.device)
    return torch.mean(torch.square(covariance - identity))


def _balanced_anchor_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    return torch.stack([losses[labels.eq(label)].mean() for label in (0, 1)]).mean()


def shiftguard_objective(
    model: ShiftDiagnosticEncoder,
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    episodes: tuple[ShiftEpisode, ...],
    *,
    valid_weight: float,
    prior_weight: float,
    invalid_weight: float,
    invalid_margin: float,
    whitening_weight: float,
    anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    source_embedding, source_logits = model(source_features)
    negative_mean, positive_mean = _class_means(source_embedding, source_labels)
    valid_losses = []
    prior_losses = []
    invalid_losses = []
    valid_discrepancies = []
    invalid_discrepancies = []
    for episode in episodes:
        target = torch.tensor(
            episode.target_features, dtype=source_features.dtype, device=source_features.device
        )
        target_embedding, _ = model(target)
        estimated_prior, discrepancy = _minimum_mixture_discrepancy(
            negative_mean, positive_mean, target_embedding.mean(dim=0)
        )
        if episode.valid_label_shift:
            valid_losses.append(discrepancy)
            prior_losses.append(
                torch.square(
                    estimated_prior
                    - torch.tensor(
                        episode.true_prevalence,
                        dtype=source_features.dtype,
                        device=source_features.device,
                    )
                )
            )
            valid_discrepancies.append(discrepancy)
        else:
            invalid_losses.append(torch.relu(invalid_margin - discrepancy))
            invalid_discrepancies.append(discrepancy)
    if not valid_losses or not invalid_losses:
        raise ValueError("ShiftGuard training requires valid and invalid episodes")
    valid_loss = torch.stack(valid_losses).mean()
    prior_loss = torch.stack(prior_losses).mean()
    invalid_loss = torch.stack(invalid_losses).mean()
    whitening = _whitening_loss(source_embedding)
    anchor = _balanced_anchor_loss(source_logits, source_labels)
    objective = (
        valid_weight * valid_loss
        + prior_weight * prior_loss
        + invalid_weight * invalid_loss
        + whitening_weight * whitening
        + anchor_weight * anchor
    )
    return objective, {
        "valid_discrepancy": float(torch.stack(valid_discrepancies).mean().detach()),
        "prior_mse": float(prior_loss.detach()),
        "invalid_hinge": float(invalid_loss.detach()),
        "invalid_discrepancy": float(torch.stack(invalid_discrepancies).mean().detach()),
        "whitening_loss": float(whitening.detach()),
        "anchor_loss": float(anchor.detach()),
    }


def _validation_score(
    model: ShiftDiagnosticEncoder,
    source: torch.Tensor,
    labels: torch.Tensor,
    episodes: tuple[ShiftEpisode, ...],
    *,
    invalid_margin: float,
) -> float:
    model.eval()
    with torch.no_grad():
        _, components = shiftguard_objective(
            model,
            source,
            labels,
            episodes,
            valid_weight=1.0,
            prior_weight=1.0,
            invalid_weight=1.0,
            invalid_margin=invalid_margin,
            whitening_weight=0.0,
            anchor_weight=0.1,
        )
    return float(
        components["valid_discrepancy"]
        + components["prior_mse"]
        + components["invalid_hinge"]
        + 0.1 * components["anchor_loss"]
    )


def fit_shiftguard(
    training_features: Any,
    training_labels: Any,
    validation_features: Any,
    validation_labels: Any,
    *,
    parameters: dict[str, Any],
    seed: int,
    device: str,
    training_mask_feature_indices: tuple[int, ...] = (),
    validation_mask_feature_indices: tuple[int, ...] = (),
) -> ShiftGuardFitResult:
    """Fit on source development folds; target outcomes are neither accepted nor used."""
    _set_reproducibility(seed)
    training = np.asarray(training_features, dtype=np.float64)
    validation = np.asarray(validation_features, dtype=np.float64)
    training_label = np.asarray(training_labels, dtype=np.int64)
    validation_label = np.asarray(validation_labels, dtype=np.int64)
    if training.ndim != 2 or validation.ndim != 2 or training.shape[1] != validation.shape[1]:
        raise ValueError("Training and validation diagnostic feature schemas differ")
    if len(training) != len(training_label) or len(validation) != len(validation_label):
        raise ValueError("Diagnostic features and labels do not align")
    if set(np.unique(training_label)) != {0, 1} or set(np.unique(validation_label)) != {0, 1}:
        raise ValueError("Both source folds require both outcome classes")
    location, scale, training_scaled = _standardizer(training)
    validation_scaled = (validation - location) / scale
    episode_count = int(parameters.get("episode_count", 24))
    target_size = int(parameters.get("target_size", min(128, len(training))))
    train_episodes = generate_shift_episodes(
        training_scaled,
        training_label,
        episode_count=episode_count,
        target_size=target_size,
        valid_fraction=float(parameters.get("valid_fraction", 0.5)),
        seed=seed + 1000,
        invalid_mechanisms=tuple(parameters.get("training_invalid_mechanisms", (
            "conditional_translation",
            "outcome_dependent_dropout",
            "support_translation",
        ))),
        mask_feature_indices=training_mask_feature_indices,
    )
    validation_episodes = generate_shift_episodes(
        validation_scaled,
        validation_label,
        episode_count=max(12, episode_count // 2),
        target_size=min(target_size, len(validation)),
        valid_fraction=0.5,
        seed=seed + 2000,
        invalid_mechanisms=tuple(parameters.get("validation_invalid_mechanisms", (
            "conditional_scale",
            "outcome_dependent_dropout",
        ))),
        mask_feature_indices=validation_mask_feature_indices,
    )
    torch_device = torch.device(device)
    model = ShiftDiagnosticEncoder(
        training.shape[1],
        hidden_dim=int(parameters.get("hidden_dim", 64)),
        embedding_dim=int(parameters.get("embedding_dim", 16)),
        dropout=float(parameters.get("dropout", 0.05)),
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters.get("learning_rate", 0.001)),
        weight_decay=float(parameters.get("weight_decay", 0.001)),
    )
    training_tensor = torch.tensor(training_scaled, dtype=torch.float32, device=torch_device)
    training_label_tensor = torch.tensor(training_label, dtype=torch.long, device=torch_device)
    validation_tensor = torch.tensor(
        validation_scaled, dtype=torch.float32, device=torch_device
    )
    validation_label_tensor = torch.tensor(
        validation_label, dtype=torch.long, device=torch_device
    )
    max_epochs = int(parameters.get("max_epochs", 300))
    patience = int(parameters.get("patience", 40))
    invalid_margin = float(parameters.get("invalid_margin", 0.05))
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        objective, components = shiftguard_objective(
            model,
            training_tensor,
            training_label_tensor,
            train_episodes,
            valid_weight=float(parameters.get("valid_weight", 1.0)),
            prior_weight=float(parameters.get("prior_weight", 1.0)),
            invalid_weight=float(parameters.get("invalid_weight", 1.0)),
            invalid_margin=invalid_margin,
            whitening_weight=float(parameters.get("whitening_weight", 0.1)),
            anchor_weight=float(parameters.get("anchor_weight", 0.5)),
        )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        score = _validation_score(
            model,
            validation_tensor,
            validation_label_tensor,
            validation_episodes,
            invalid_margin=invalid_margin,
        )
        if epoch == 1 or epoch % 10 == 0:
            history.append(
                {
                    "epoch": float(epoch),
                    "objective": float(objective.detach()),
                    "validation_score": score,
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
        raise RuntimeError("ShiftGuard produced no finite checkpoint")
    model.load_state_dict(best_state)
    return ShiftGuardFitResult(
        model=model,
        location=location,
        scale=scale,
        best_epoch=best_epoch,
        validation_score=best_score,
        history=history,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        device=str(torch_device),
    )


def encode_shiftguard(result: ShiftGuardFitResult, features: Any) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(result.location):
        raise ValueError("ShiftGuard inference feature schema differs from training")
    if not np.isfinite(matrix).all():
        raise ValueError("ShiftGuard inference features must be finite")
    scaled = (matrix - result.location) / result.scale
    result.model.eval()
    with torch.no_grad():
        embedding, _ = result.model(
            torch.tensor(scaled, dtype=torch.float32, device=torch.device(result.device))
        )
    return np.asarray(embedding.cpu().numpy(), dtype=np.float64)


def _embedding_discrepancies(
    source_embedding: np.ndarray,
    source_labels: np.ndarray,
    target_embedding: np.ndarray,
    prior_grid: np.ndarray,
    reducer: str = "mean_square",
    *,
    regularization: float = 0.05,
    precision_matrices: np.ndarray | None = None,
) -> np.ndarray:
    negative = source_embedding[source_labels == 0].mean(axis=0)
    positive = source_embedding[source_labels == 1].mean(axis=0)
    target = target_embedding.mean(axis=0)
    mixtures = negative[None, :] + prior_grid[:, None] * (positive - negative)[None, :]
    squared = np.square(mixtures - target[None, :])
    if reducer == "mean_square":
        return np.asarray(np.mean(squared, axis=1), dtype=np.float64)
    if reducer == "max_square":
        return np.asarray(np.max(squared, axis=1), dtype=np.float64)
    if reducer == "spectral_ridge":
        precisions = (
            fit_mixture_precision_matrices(
                source_embedding,
                source_labels,
                prior_grid,
                regularization=regularization,
            )
            if precision_matrices is None
            else np.asarray(precision_matrices, dtype=np.float64)
        )
        expected_shape = (len(prior_grid), source_embedding.shape[1], source_embedding.shape[1])
        if precisions.shape != expected_shape or not np.isfinite(precisions).all():
            raise ValueError("ShiftGuard precision matrices do not align")
        residual = target[None, :] - mixtures
        quadratic = np.einsum(
            "pi,pij,pj->p", residual, precisions, residual, optimize=True
        )
        return np.asarray(
            len(target_embedding) * quadratic / source_embedding.shape[1],
            dtype=np.float64,
        )
    raise KeyError(f"Unknown ShiftGuard discrepancy reducer: {reducer}")


def fit_mixture_precision_matrices(
    source_embedding: Any,
    source_labels: Any,
    prior_grid: Any,
    *,
    regularization: float,
) -> np.ndarray:
    """Fit source-only ridge precision operators for label-shift mixtures.

    This is a finite-feature analogue of a covariance-regularized kernel mean
    test.  The ridge is scaled by each mixture covariance's average variance,
    so the configured value is dimensionless and auditable across views.
    """
    embedding = np.asarray(source_embedding, dtype=np.float64)
    labels = np.asarray(source_labels, dtype=np.int64)
    priors = np.asarray(prior_grid, dtype=np.float64)
    if (
        embedding.ndim != 2
        or len(embedding) != len(labels)
        or set(np.unique(labels)) != {0, 1}
    ):
        raise ValueError("Precision fitting requires aligned binary source embeddings")
    if priors.ndim != 1 or not ((priors > 0.0) & (priors < 1.0)).all():
        raise ValueError("Precision fitting requires interior prevalence values")
    if not np.isfinite(embedding).all() or not np.isfinite(regularization):
        raise ValueError("Precision inputs must be finite")
    if regularization <= 0.0:
        raise ValueError("Spectral-ridge regularization must be positive")

    class_values = [embedding[labels == label] for label in (0, 1)]
    class_means = [values.mean(axis=0) for values in class_values]
    class_covariances = [
        np.atleast_2d(np.cov(values, rowvar=False, ddof=1)) for values in class_values
    ]
    dimension = embedding.shape[1]
    identity = np.eye(dimension, dtype=np.float64)
    precisions = np.empty((len(priors), dimension, dimension), dtype=np.float64)
    mean_difference = class_means[1] - class_means[0]
    for index, prior in enumerate(priors):
        covariance = (
            (1.0 - prior) * class_covariances[0]
            + prior * class_covariances[1]
            + prior * (1.0 - prior) * np.outer(mean_difference, mean_difference)
        )
        average_variance = max(float(np.trace(covariance) / dimension), 1e-12)
        stabilized = covariance + regularization * average_variance * identity
        precisions[index] = np.linalg.pinv(stabilized, hermitian=True)
    return precisions


def calibrate_shiftguard_prevalence_set(
    source_embedding: Any,
    source_labels: Any,
    *,
    calibration_pool_embedding: Any | None = None,
    calibration_pool_labels: Any | None = None,
    prior_grid: Any,
    target_size: int,
    repetitions_per_prior: int,
    alpha: float,
    seed: int,
    discrepancy_reducer: str = "mean_square",
    discrepancy_regularization: float = 0.05,
) -> ShiftGuardCalibration:
    """Calibrate prior-wise critical values from held-out pure-label-shift episodes."""
    embedding = np.asarray(source_embedding, dtype=np.float64)
    labels = np.asarray(source_labels, dtype=np.int64)
    pool_embedding = (
        embedding
        if calibration_pool_embedding is None
        else np.asarray(calibration_pool_embedding, dtype=np.float64)
    )
    pool_labels = (
        labels
        if calibration_pool_labels is None
        else np.asarray(calibration_pool_labels, dtype=np.int64)
    )
    priors = np.asarray(prior_grid, dtype=np.float64)
    if set(np.unique(labels)) != {0, 1} or len(embedding) != len(labels):
        raise ValueError("Calibration embeddings require aligned binary labels")
    if (
        pool_embedding.ndim != 2
        or pool_embedding.shape[1] != embedding.shape[1]
        or len(pool_embedding) != len(pool_labels)
        or set(np.unique(pool_labels)) != {0, 1}
    ):
        raise ValueError("Held-out calibration pool must align with the source embedding schema")
    if priors.ndim != 1 or len(priors) < 2 or np.any(np.diff(priors) <= 0):
        raise ValueError("Prior grid must be increasing")
    if not ((priors > 0.0) & (priors < 1.0)).all():
        raise ValueError("Prior grid values must lie inside (0, 1)")
    if repetitions_per_prior < 20 or not 0.0 < alpha < 1.0:
        raise ValueError("Calibration repetitions or alpha are invalid")
    rng = np.random.default_rng(seed)
    precision_matrices = (
        fit_mixture_precision_matrices(
            embedding,
            labels,
            priors,
            regularization=discrepancy_regularization,
        )
        if discrepancy_reducer == "spectral_ridge"
        else None
    )
    values = np.empty((len(priors), repetitions_per_prior), dtype=np.float64)
    for prior_index, prior in enumerate(priors):
        for repetition in range(repetitions_per_prior):
            target, _ = _draw_label_shift_target(
                pool_embedding,
                pool_labels,
                prevalence=float(prior),
                size=target_size,
                rng=rng,
            )
            values[prior_index, repetition] = _embedding_discrepancies(
                embedding,
                labels,
                target,
                np.asarray([prior]),
                reducer=discrepancy_reducer,
                regularization=discrepancy_regularization,
                precision_matrices=(
                    precision_matrices[prior_index : prior_index + 1]
                    if precision_matrices is not None
                    else None
                ),
            )[0]
    critical = np.quantile(values, 1.0 - alpha, axis=1, method="higher")
    return ShiftGuardCalibration(
        prior_grid=priors,
        critical_values=np.asarray(critical, dtype=np.float64),
        alpha=float(alpha),
        target_size=int(target_size),
        repetitions_per_prior=int(repetitions_per_prior),
        discrepancy_reducer=discrepancy_reducer,
        discrepancy_regularization=float(discrepancy_regularization),
        precision_matrices=precision_matrices,
        null_discrepancies=values,
    )


def shiftguard_prevalence_set(
    source_embedding: Any,
    source_labels: Any,
    target_embedding: Any,
    calibration: ShiftGuardCalibration,
) -> PrevalenceSet:
    """Invert the calibrated compatibility test into a discrete prevalence set."""
    source = np.asarray(source_embedding, dtype=np.float64)
    target = np.asarray(target_embedding, dtype=np.float64)
    labels = np.asarray(source_labels, dtype=np.int64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Source and target embedding schemas differ")
    if len(target) != calibration.target_size:
        raise ValueError("Target batch size differs from ShiftGuard calibration")
    discrepancies = _embedding_discrepancies(
        source,
        labels,
        target,
        calibration.prior_grid,
        reducer=calibration.discrepancy_reducer,
        regularization=calibration.discrepancy_regularization,
        precision_matrices=calibration.precision_matrices,
    )
    accepted = calibration.prior_grid[discrepancies <= calibration.critical_values]
    best_index = int(np.argmin(discrepancies))
    return PrevalenceSet(
        accepted=bool(len(accepted)),
        accepted_priors=tuple(float(value) for value in accepted),
        lower=float(accepted.min()) if len(accepted) else None,
        upper=float(accepted.max()) if len(accepted) else None,
        best_prior=float(calibration.prior_grid[best_index]),
        best_discrepancy=float(discrepancies[best_index]),
        discrepancies=discrepancies,
    )


def posterior_interval_from_prevalence_set(
    evidence_logit: Any,
    prevalence_set: PrevalenceSet,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a non-empty prevalence set to patient-level posterior bounds."""
    if not prevalence_set.accepted or prevalence_set.lower is None or prevalence_set.upper is None:
        raise ValueError("Posterior intervals require an accepted prevalence set")
    evidence = np.asarray(evidence_logit, dtype=np.float64)
    if not np.isfinite(evidence).all():
        raise ValueError("Evidence logits must be finite")
    lower = expit(evidence + logit(prevalence_set.lower))
    upper = expit(evidence + logit(prevalence_set.upper))
    return np.minimum(lower, upper), np.maximum(lower, upper)


def selective_decisions(
    lower_probability: Any,
    upper_probability: Any,
    *,
    threshold: float,
) -> np.ndarray:
    """Return 0/1 decisions only when every compatible prior agrees; else -1."""
    lower = np.asarray(lower_probability, dtype=np.float64)
    upper = np.asarray(upper_probability, dtype=np.float64)
    if lower.shape != upper.shape or np.any(lower > upper):
        raise ValueError("Posterior interval bounds are invalid")
    if not 0.0 < threshold < 1.0:
        raise ValueError("Decision threshold must lie inside (0, 1)")
    decisions = np.full(lower.shape, -1, dtype=np.int8)
    decisions[upper < threshold] = 0
    decisions[lower > threshold] = 1
    return decisions
