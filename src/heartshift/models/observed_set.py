"""Compact observed-feature-set encoder and fold-local numerical schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from heartshift.data.uci import CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS, FEATURE_COLUMNS


def _category_key(value: Any) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"number:{float(value):.12g}"
    return f"string:{value}"


@dataclass
class NeuralPreprocessor:
    """A serializable preprocessing state fitted only on a training fold."""

    feature_columns: tuple[str, ...]
    continuous_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    continuous_mean: dict[str, float]
    continuous_scale: dict[str, float]
    category_maps: dict[str, dict[str, int]]
    reference_values: list[float]

    @classmethod
    def fit(
        cls,
        training: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
        continuous_columns: tuple[str, ...] = CONTINUOUS_COLUMNS,
        categorical_columns: tuple[str, ...] = CATEGORICAL_COLUMNS,
    ) -> NeuralPreprocessor:
        if set(continuous_columns) | set(categorical_columns) != set(feature_columns):
            raise ValueError("Continuous and categorical columns must partition all features")
        if set(continuous_columns) & set(categorical_columns):
            raise ValueError("A feature cannot be both continuous and categorical")
        continuous_mean: dict[str, float] = {}
        continuous_scale: dict[str, float] = {}
        category_maps: dict[str, dict[str, int]] = {}
        reference_raw: dict[str, Any] = {}

        for feature in continuous_columns:
            observed = training[feature].dropna().astype(float)
            mean = float(observed.mean()) if len(observed) else 0.0
            scale = float(observed.std(ddof=0)) if len(observed) else 1.0
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            continuous_mean[feature] = mean
            continuous_scale[feature] = scale
            class_medians = [
                training.loc[training["target"].eq(label), feature].dropna().median()
                for label in (0, 1)
            ]
            finite_medians = [float(value) for value in class_medians if pd.notna(value)]
            reference_raw[feature] = float(np.mean(finite_medians)) if finite_medians else mean

        for feature in categorical_columns:
            keyed_values = {
                _category_key(value): value for value in training[feature].dropna().unique()
            }
            keys = sorted(keyed_values)
            mapping = {key: index for index, key in enumerate(keys, start=1)}
            category_maps[feature] = mapping
            if not keys:
                reference_raw[feature] = float("nan")
                continue
            balanced_frequencies: dict[str, float] = {}
            for candidate in keys:
                frequencies = []
                for label in (0, 1):
                    class_values = training.loc[training["target"].eq(label), feature].dropna()
                    frequency = (
                        float(class_values.map(_category_key).eq(candidate).mean())
                        if len(class_values)
                        else 0.0
                    )
                    frequencies.append(frequency)
                balanced_frequencies[candidate] = float(np.mean(frequencies))
            chosen = max(keys, key=lambda key: (balanced_frequencies[key], key))
            reference_raw[feature] = keyed_values[chosen]

        provisional = cls(
            feature_columns,
            continuous_columns,
            categorical_columns,
            continuous_mean,
            continuous_scale,
            category_maps,
            [],
        )
        reference_frame = pd.DataFrame([reference_raw], columns=feature_columns)
        reference_values, _ = provisional.transform(reference_frame)
        provisional.reference_values = reference_values[0].tolist()
        return provisional

    @property
    def categorical_cardinalities(self) -> dict[int, int]:
        return {
            self.feature_columns.index(feature): len(mapping)
            for feature, mapping in self.category_maps.items()
        }

    @property
    def continuous_indices(self) -> tuple[int, ...]:
        return tuple(self.feature_columns.index(feature) for feature in self.continuous_columns)

    def transform(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        observed = data.loc[:, self.feature_columns].notna().to_numpy(dtype=bool)
        values = np.zeros((len(data), len(self.feature_columns)), dtype=np.float32)
        for feature in self.continuous_columns:
            index = self.feature_columns.index(feature)
            raw = data[feature].to_numpy(dtype=np.float64, na_value=np.nan)
            normalized = (raw - self.continuous_mean[feature]) / self.continuous_scale[feature]
            values[:, index] = np.nan_to_num(normalized, nan=0.0).astype(np.float32)
        for feature in self.categorical_columns:
            index = self.feature_columns.index(feature)
            mapping = self.category_maps[feature]
            encoded = []
            for value in data[feature]:
                encoded.append(0 if pd.isna(value) else mapping.get(_category_key(value), 0))
            values[:, index] = np.asarray(encoded, dtype=np.float32)
        return values, observed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObservedFeatureSetEncoder(nn.Module):
    """Encode only observed feature-value tokens; hospital identity is never an input."""

    def __init__(
        self,
        *,
        n_features: int,
        continuous_indices: tuple[int, ...],
        categorical_cardinalities: dict[int, int],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        ane: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_features = n_features
        self.continuous_indices = frozenset(continuous_indices)
        self.ane = ane
        self.feature_embedding = nn.Embedding(n_features, d_model)
        self.continuous_weight = nn.Parameter(torch.empty(n_features, d_model))
        self.continuous_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.categorical_embeddings = nn.ModuleDict(
            {
                str(index): nn.Embedding(cardinality + 1, d_model, padding_idx=0)
                for index, cardinality in categorical_cardinalities.items()
            }
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1, bias=False)
        self.global_bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.feature_embedding.weight, std=0.02)
        nn.init.normal_(self.continuous_weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def _raw_score(self, values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError("values must have shape [batch, n_features]")
        if observed.shape != values.shape or observed.dtype != torch.bool:
            raise ValueError("observed must be a boolean tensor aligned with values")
        feature_ids = torch.arange(self.n_features, device=values.device)
        base = self.feature_embedding(feature_ids).unsqueeze(0).expand(len(values), -1, -1)
        tokens = base.clone()
        for index in range(self.n_features):
            if index in self.continuous_indices:
                continuous_value = torch.where(
                    observed[:, index], values[:, index], torch.zeros_like(values[:, index])
                )
                tokens[:, index] = (
                    tokens[:, index]
                    + continuous_value[:, None] * self.continuous_weight[index]
                    + self.continuous_bias[index]
                )
            else:
                category = torch.where(
                    observed[:, index],
                    values[:, index].long(),
                    torch.zeros_like(values[:, index].long()),
                )
                tokens[:, index] = tokens[:, index] + self.categorical_embeddings[str(index)](
                    category
                )
        cls = self.cls_token.expand(len(values), -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        padding = torch.cat(
            [torch.zeros((len(values), 1), device=values.device, dtype=torch.bool), ~observed],
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        return cast(torch.Tensor, self.output(self.output_norm(encoded[:, 0])).squeeze(-1))

    def forward(
        self,
        values: torch.Tensor,
        observed: torch.Tensor,
        reference_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = self._raw_score(values, observed)
        if not self.ane:
            return raw + self.global_bias
        if reference_values is None:
            raise ValueError("ANE mode requires fold-local reference values")
        if reference_values.ndim == 1:
            reference_values = reference_values.unsqueeze(0).expand(len(values), -1)
        reference_score = self._raw_score(reference_values, observed)
        return raw - reference_score + self.global_bias
