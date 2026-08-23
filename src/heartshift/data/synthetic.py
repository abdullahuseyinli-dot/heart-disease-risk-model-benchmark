"""Controlled clinical-table environments for mechanism and failure-mode tests."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.special import expit

from heartshift.data.uci import FEATURE_COLUMNS, SCHEMA_VERSION, ZERO_SENTINEL_COLUMNS

SYNTHETIC_SCENARIOS = ("label_mar", "conditional_shift", "mnar_outcome")


def _categorical_from_uniform(uniform: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    return np.asarray(1 + (uniform[:, None] > cumulative).sum(axis=1))


def _environment_frame(
    *,
    environment: int,
    n: int,
    prevalence: float,
    scenario: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    y = (rng.random(n) < prevalence).astype(np.int8)
    signed = 2.0 * y - 1.0
    conditional_modifier = -1.0 if scenario == "conditional_shift" and environment == 3 else 1.0

    age = rng.normal(53.0 + 4.0 * signed, 8.0, n)
    trestbps = rng.normal(130.0 + 7.0 * signed, 14.0, n)
    chol = rng.normal(230.0 + 12.0 * signed, 35.0, n)
    thalach = rng.normal(150.0 - 15.0 * conditional_modifier * signed, 18.0, n)
    oldpeak = np.maximum(0.0, rng.normal(0.8 + 0.8 * signed, 0.8, n))
    sex = (rng.random(n) < expit(0.1 + 0.5 * signed)).astype(float)
    cp_positive = np.array([0.08, 0.12, 0.20, 0.60])
    cp_negative = np.array([0.30, 0.30, 0.25, 0.15])
    cp_probabilities = np.where(y[:, None].astype(bool), cp_positive, cp_negative)
    if scenario == "conditional_shift" and environment == 3:
        cp_probabilities = np.where(y[:, None].astype(bool), cp_negative, cp_positive)
    cp = _categorical_from_uniform(rng.random(n), cp_probabilities).astype(float)
    fbs = (rng.random(n) < expit(-1.8 + 0.5 * signed)).astype(float)
    restecg = (
        _categorical_from_uniform(
            rng.random(n),
            np.where(
                y[:, None].astype(bool),
                np.array([0.40, 0.20, 0.40]),
                np.array([0.70, 0.20, 0.10]),
            ),
        ).astype(float)
        - 1.0
    )
    exang = (rng.random(n) < expit(-1.0 + 1.2 * signed)).astype(float)
    slope = _categorical_from_uniform(
        rng.random(n),
        np.where(
            y[:, None].astype(bool),
            np.array([0.20, 0.60, 0.20]),
            np.array([0.65, 0.30, 0.05]),
        ),
    ).astype(float)
    ca = np.clip(rng.poisson(np.where(y == 1, 1.2, 0.2)), 0, 3).astype(float)
    thal_index = _categorical_from_uniform(
        rng.random(n),
        np.where(
            y[:, None].astype(bool),
            np.array([0.20, 0.25, 0.55]),
            np.array([0.75, 0.15, 0.10]),
        ),
    )
    thal = np.asarray([3.0, 6.0, 7.0])[thal_index - 1]

    frame = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
    )
    missing_base = (0.05, 0.15, 0.30, 0.45)[environment]
    mar_driver = (age - 53.0) / 10.0 + 0.4 * (sex - 0.5)
    for feature_index, feature in enumerate(FEATURE_COLUMNS[3:], start=3):
        feature_offset = 0.12 * ((feature_index % 4) - 1.5)
        missing_logit = np.log(missing_base / (1 - missing_base)) + mar_driver + feature_offset
        if scenario == "mnar_outcome":
            missing_logit = missing_logit + 1.5 * signed
        missing = rng.random(n) < expit(missing_logit)
        frame.loc[missing, feature] = np.nan

    site = f"synthetic_e{environment}"
    sample_ids = [f"synthetic:{scenario}:{environment}:{index:05d}" for index in range(n)]
    frame.insert(0, "site", site)
    frame.insert(0, "sample_id", sample_ids)
    frame["target"] = y
    frame["target_ordinal"] = y
    frame["source_row"] = np.arange(1, n + 1)
    frame["source_file"] = f"generated:{scenario}"
    frame["source_file_sha256"] = "synthetic"
    frame["source_line_sha256"] = [
        hashlib.sha256(value.encode()).hexdigest() for value in sample_ids
    ]
    record_payload = frame.loc[:, FEATURE_COLUMNS].fillna(-9999).astype(str).agg("|".join, axis=1)
    frame["record_sha256"] = [
        hashlib.sha256(value.encode()).hexdigest() for value in record_payload
    ]
    frame["schema_version"] = f"synthetic-{SCHEMA_VERSION}"
    frame["duplicate_cluster_size"] = 1
    frame["is_exact_duplicate"] = False
    for column in ZERO_SENTINEL_COLUMNS:
        frame[f"{column}_zero_sentinel"] = False
    return frame


def generate_synthetic_environments(
    scenario: str,
    *,
    n_per_environment: int = 500,
    seed: int = 5062,
    prevalences: tuple[float, float, float, float] = (0.20, 0.40, 0.60, 0.80),
) -> pd.DataFrame:
    """Generate four environments with independently controlled shift mechanisms."""
    if scenario not in SYNTHETIC_SCENARIOS:
        raise KeyError(f"Unknown synthetic scenario: {scenario}")
    if len(prevalences) != 4:
        raise ValueError("Exactly four environment prevalences are required")
    frames = []
    for environment, prevalence in enumerate(prevalences):
        rng = np.random.default_rng(seed + 10_007 * environment)
        frames.append(
            _environment_frame(
                environment=environment,
                n=n_per_environment,
                prevalence=prevalence,
                scenario=scenario,
                rng=rng,
            )
        )
    return pd.concat(frames, ignore_index=True)
