"""Deterministic natural and counterfactual measurement-policy interventions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from heartshift.data.uci import CORE_COLUMNS, FEATURE_COLUMNS, FEATURE_PANELS


@dataclass(frozen=True)
class MaskPolicy:
    """A named intervention that may only hide already observed values."""

    name: str
    kind: str
    rate: float = 0.0
    panel: str | None = None
    strength: float = 1.0
    feature_rates: dict[str, float] = field(default_factory=dict)
    protect_core: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> MaskPolicy:
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            rate=float(payload.get("rate", 0.0)),
            panel=payload.get("panel"),
            strength=float(payload.get("strength", 1.0)),
            feature_rates={
                str(key): float(value) for key, value in payload.get("feature_rates", {}).items()
            },
            protect_core=bool(payload.get("protect_core", False)),
        )


def policy_seed(base_seed: int, policy_name: str, replicate: int) -> int:
    payload = f"{base_seed}|{policy_name}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def _eligible_columns(policy: MaskPolicy) -> np.ndarray:
    eligible = np.ones(len(FEATURE_COLUMNS), dtype=bool)
    if policy.protect_core:
        for column in CORE_COLUMNS:
            eligible[FEATURE_COLUMNS.index(column)] = False
    return eligible


def apply_mask_policy(
    data: pd.DataFrame,
    policy: MaskPolicy,
    *,
    base_seed: int,
    replicate: int = 0,
    empirical_mask_pool: np.ndarray | None = None,
) -> np.ndarray:
    """Return the counterfactual observed mask; never reveal a missing value."""
    natural = data.loc[:, FEATURE_COLUMNS].notna().to_numpy(dtype=bool)
    if policy.kind == "natural":
        return np.asarray(natural.copy(), dtype=bool)

    rng = np.random.default_rng(policy_seed(base_seed, policy.name, replicate))
    keep = natural.copy()
    eligible = _eligible_columns(policy)

    if policy.kind == "mcar":
        deleted = rng.random(natural.shape) < policy.rate
        keep &= ~(deleted & eligible[None, :])
    elif policy.kind == "featurewise":
        for feature, rate in policy.feature_rates.items():
            column = FEATURE_COLUMNS.index(feature)
            if eligible[column]:
                keep[:, column] &= rng.random(len(data)) >= rate
    elif policy.kind == "panel":
        if policy.panel not in FEATURE_PANELS:
            raise KeyError(f"Unknown clinical feature panel: {policy.panel}")
        for feature in FEATURE_PANELS[str(policy.panel)]:
            column = FEATURE_COLUMNS.index(feature)
            if eligible[column]:
                keep[:, column] = False
    elif policy.kind == "mar":
        age = data["age"].fillna(55.0).to_numpy(dtype=np.float64)
        sex = data["sex"].fillna(0.5).to_numpy(dtype=np.float64)
        linear = logit(np.clip(policy.rate, 1e-4, 1 - 1e-4))
        linear = linear + policy.strength * ((age - 55.0) / 10.0 + 0.5 * (sex - 0.5))
        row_probability = expit(linear)
        deleted = rng.random(natural.shape) < row_probability[:, None]
        keep &= ~(deleted & eligible[None, :])
    elif policy.kind == "mnar_outcome":
        if "target" not in data:
            raise ValueError("Outcome-dependent stress policy requires target labels")
        target = data["target"].to_numpy(dtype=np.float64)
        linear = logit(np.clip(policy.rate, 1e-4, 1 - 1e-4))
        linear = linear + policy.strength * (2.0 * target - 1.0)
        row_probability = expit(linear)
        deleted = rng.random(natural.shape) < row_probability[:, None]
        keep &= ~(deleted & eligible[None, :])
    elif policy.kind == "empirical":
        if empirical_mask_pool is None or len(empirical_mask_pool) == 0:
            raise ValueError("Empirical mask policy requires a non-empty source mask pool")
        sampled = empirical_mask_pool[rng.integers(0, len(empirical_mask_pool), size=len(data))]
        if sampled.shape != natural.shape:
            raise ValueError("Empirical masks do not match the canonical feature count")
        keep &= sampled
    else:
        raise KeyError(f"Unknown mask-policy kind: {policy.kind}")

    if np.any(keep & ~natural):
        raise AssertionError("A mask intervention revealed naturally missing information")
    empty = ~keep.any(axis=1)
    if empty.any():
        # Preserve the first naturally observed feature to keep the prediction task defined.
        for row in np.flatnonzero(empty):
            observed_columns = np.flatnonzero(natural[row])
            if len(observed_columns):
                keep[row, observed_columns[0]] = True
    return np.asarray(keep, dtype=bool)


def mask_frame(data: pd.DataFrame, observed_mask: np.ndarray) -> pd.DataFrame:
    """Return a modelling-frame copy with counterfactually hidden values set to NaN."""
    if observed_mask.shape != (len(data), len(FEATURE_COLUMNS)):
        raise ValueError("Observed mask shape does not match data")
    result = data.copy()
    values = result.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    values[~observed_mask] = np.nan
    result.loc[:, FEATURE_COLUMNS] = values
    return result


def default_policy_bank(include_failure_stress: bool = False) -> tuple[MaskPolicy, ...]:
    policies = (
        MaskPolicy("natural", "natural"),
        MaskPolicy("mcar_10", "mcar", rate=0.10),
        MaskPolicy("mcar_30", "mcar", rate=0.30),
        MaskPolicy("mcar_50", "mcar", rate=0.50),
        MaskPolicy("drop_routine", "panel", panel="routine"),
        MaskPolicy("drop_exercise", "panel", panel="exercise"),
        MaskPolicy("drop_advanced", "panel", panel="advanced"),
        MaskPolicy("mar_age_sex_30", "mar", rate=0.30, strength=0.8, protect_core=True),
        MaskPolicy("empirical_source", "empirical"),
    )
    if include_failure_stress:
        return (
            *policies,
            MaskPolicy("mnar_outcome_30", "mnar_outcome", rate=0.30, strength=1.5),
        )
    return policies
