"""Leakage-safe classical baselines with fold-fitted preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartshift.data.uci import (
    CATEGORICAL_COLUMNS,
    CONTINUOUS_COLUMNS,
    CORE_COLUMNS,
    FEATURE_COLUMNS,
)


def selected_features(view: str) -> tuple[str, ...]:
    if view == "all":
        return FEATURE_COLUMNS
    if view == "core":
        return CORE_COLUMNS
    if view == "mask":
        return FEATURE_COLUMNS
    raise KeyError(f"Unknown feature view: {view}")


def build_preprocessor(view: str = "all") -> ColumnTransformer:
    features = selected_features(view)
    if view == "mask":
        return ColumnTransformer(
            [("mask", MissingIndicator(features="all"), list(features))],
            remainder="drop",
            verbose_feature_names_out=True,
        )
    numerical = [column for column in CONTINUOUS_COLUMNS if column in features]
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
            ("mask", MissingIndicator(features="all"), list(features)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_generic_preprocessor(
    feature_columns: tuple[str, ...],
    continuous_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
) -> ColumnTransformer:
    """Build a fold-fitted transformer for an arbitrary heterogeneous table."""
    if set(continuous_columns) & set(categorical_columns):
        raise ValueError("Continuous and categorical feature sets overlap")
    if set(continuous_columns) | set(categorical_columns) != set(feature_columns):
        raise ValueError("Continuous and categorical features must partition all features")
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(continuous_columns),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical_columns),
            ),
            ("mask", MissingIndicator(features="all"), list(feature_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _estimator(name: str, parameters: dict[str, Any], seed: int) -> Any:
    if name in {"logistic", "core_logistic", "mask_logistic"}:
        l1_ratio = float(parameters.get("l1_ratio", 0.0))
        return LogisticRegression(
            C=float(parameters.get("C", 1.0)),
            l1_ratio=l1_ratio,
            solver="lbfgs" if l1_ratio == 0.0 else "saga",
            max_iter=100_000,
            tol=1e-5,
            random_state=seed,
        )
    if name == "elastic_net_logistic":
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=float(parameters.get("alpha", 0.001)),
            l1_ratio=float(parameters.get("l1_ratio", 0.5)),
            max_iter=100_000,
            tol=1e-6,
            average=True,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(parameters.get("n_estimators", 500)),
            max_depth=parameters.get("max_depth"),
            min_samples_leaf=int(parameters.get("min_samples_leaf", 5)),
            max_features=parameters.get("max_features", "sqrt"),
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(parameters.get("learning_rate", 0.05)),
            max_iter=int(parameters.get("max_iter", 300)),
            max_leaf_nodes=int(parameters.get("max_leaf_nodes", 15)),
            min_samples_leaf=int(parameters.get("min_samples_leaf", 20)),
            l2_regularization=float(parameters.get("l2_regularization", 1.0)),
            random_state=seed,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=int(parameters.get("n_estimators", 300)),
            max_depth=int(parameters.get("max_depth", 3)),
            learning_rate=float(parameters.get("learning_rate", 0.05)),
            min_child_weight=float(parameters.get("min_child_weight", 5.0)),
            subsample=float(parameters.get("subsample", 0.8)),
            colsample_bytree=float(parameters.get("colsample_bytree", 0.8)),
            reg_lambda=float(parameters.get("reg_lambda", 5.0)),
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(parameters.get("n_estimators", 300)),
            learning_rate=float(parameters.get("learning_rate", 0.03)),
            num_leaves=int(parameters.get("num_leaves", 15)),
            min_child_samples=int(parameters.get("min_child_samples", 30)),
            reg_lambda=float(parameters.get("reg_lambda", 5.0)),
            subsample=float(parameters.get("subsample", 0.8)),
            colsample_bytree=float(parameters.get("colsample_bytree", 0.8)),
            verbosity=-1,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=int(parameters.get("iterations", 300)),
            depth=int(parameters.get("depth", 5)),
            learning_rate=float(parameters.get("learning_rate", 0.03)),
            l2_leaf_reg=float(parameters.get("l2_leaf_reg", 5.0)),
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=seed,
        )
    if name == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier

        return ExplainableBoostingClassifier(
            interactions=int(parameters.get("interactions", 3)),
            max_bins=int(parameters.get("max_bins", 256)),
            max_rounds=int(parameters.get("max_rounds", 5000)),
            outer_bags=int(parameters.get("outer_bags", 8)),
            min_samples_leaf=int(parameters.get("min_samples_leaf", 4)),
            validation_size=0.15,
            early_stopping_rounds=100,
            n_jobs=-2,
            random_state=seed,
        )
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        overrides = {
            "n_estimators": int(parameters.get("n_estimators", 8)),
            "device": str(parameters.get("device", "cuda")),
            "fit_mode": "fit_preprocessors",
            "memory_saving_mode": "auto",
            "random_state": seed,
            "show_progress_bar": False,
        }
        version = str(parameters.get("version", "v2"))
        if version == "v2":
            return TabPFNClassifier.create_default_for_version(ModelVersion.V2, **overrides)
        if version == "v3":
            return TabPFNClassifier(**overrides)
        raise ValueError(f"Unsupported TabPFN version: {version}")
    if name == "tabicl":
        from tabicl import TabICLClassifier

        return TabICLClassifier(
            n_estimators=int(parameters.get("n_estimators", 8)),
            checkpoint_version=str(
                parameters.get("checkpoint_version", "tabicl-classifier-v2-20260212.ckpt")
            ),
            device=str(parameters.get("device", "cuda")),
            batch_size=int(parameters.get("batch_size", 8)),
            allow_auto_download=False,
            random_state=seed,
            verbose=False,
        )
    if name in {"realmlp", "ft_transformer", "tabm"}:
        from pytabkit.models.sklearn.sklearn_interfaces import (
            FTT_D_Classifier,
            RealMLP_TD_S_Classifier,
            TabM_D_Classifier,
        )

        common = {
            "device": str(parameters.get("device", "cuda")),
            "random_state": seed,
            "n_cv": 1,
            "n_refit": 0,
            "n_repeats": 1,
            "val_fraction": float(parameters.get("val_fraction", 0.2)),
            "n_threads": int(parameters.get("n_threads", 8)),
            "verbosity": 0,
            "val_metric_name": "cross_entropy",
        }
        if name == "realmlp":
            return RealMLP_TD_S_Classifier(
                **common,
                n_epochs=int(parameters.get("n_epochs", 256)),
                batch_size=int(parameters.get("batch_size", 128)),
            )
        if name == "ft_transformer":
            return FTT_D_Classifier(
                **common,
                max_epochs=int(parameters.get("max_epochs", 200)),
                batch_size=int(parameters.get("batch_size", 128)),
                es_patience=int(parameters.get("patience", 20)),
            )
        return TabM_D_Classifier(
            **common,
            n_epochs=int(parameters.get("n_epochs", 256)),
            batch_size=int(parameters.get("batch_size", 128)),
            patience=int(parameters.get("patience", 20)),
            tabm_k=int(parameters.get("tabm_k", 16)),
        )
    raise KeyError(f"Unknown model: {name}")


def build_classical_pipeline(name: str, parameters: dict[str, Any], seed: int) -> Pipeline:
    view = "all"
    if name == "core_logistic":
        view = "core"
    elif name == "mask_logistic":
        view = "mask"
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(view)),
            ("model", _estimator(name, parameters, seed)),
        ]
    )


def build_generic_classical_pipeline(
    name: str,
    parameters: dict[str, Any],
    seed: int,
    *,
    feature_columns: tuple[str, ...],
    continuous_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
) -> Pipeline:
    """Build a leakage-safe classical pipeline for a non-UCI-Heart table."""
    return Pipeline(
        [
            (
                "preprocessor",
                build_generic_preprocessor(
                    feature_columns,
                    continuous_columns,
                    categorical_columns,
                ),
            ),
            ("model", _estimator(name, parameters, seed)),
        ]
    )


def group_sample_weights(
    data: pd.DataFrame,
    strategy: str,
    *,
    group_column: str,
) -> np.ndarray:
    """Return normalized class/group weights without exposing the group as a feature."""
    if strategy == "pooled":
        return np.ones(len(data), dtype=np.float64)
    group_balanced = strategy in {"site_balanced", "environment_balanced"}
    group_class_balanced = strategy in {
        "site_class_balanced",
        "environment_class_balanced",
    }
    if strategy != "class_balanced" and not group_balanced and not group_class_balanced:
        raise KeyError(f"Unknown weighting strategy: {strategy}")
    weights = np.ones(len(data), dtype=np.float64)
    if group_balanced or group_class_balanced:
        groups_array = data[group_column].to_numpy()
        for group in np.unique(groups_array):
            selected = groups_array == group
            weights[selected] *= len(data) / (len(np.unique(groups_array)) * selected.sum())
    if strategy == "class_balanced" or group_class_balanced:
        if strategy == "class_balanced":
            groups = [(np.ones(len(data), dtype=bool), data["target"].to_numpy())]
        else:
            groups = [
                (
                    data[group_column].eq(group).to_numpy(),
                    data.loc[data[group_column].eq(group), "target"].to_numpy(),
                )
                for group in data[group_column].unique()
            ]
        for selected, labels in groups:
            for label in (0, 1):
                local = np.zeros(len(data), dtype=bool)
                local[np.flatnonzero(selected)[labels == label]] = True
                if local.any():
                    weights[local] *= selected.sum() / (2.0 * local.sum())
    return np.asarray(weights / weights.mean(), dtype=np.float64)


def sample_weights(data: pd.DataFrame, strategy: str) -> np.ndarray:
    """Backward-compatible heart-benchmark weighting by source hospital."""
    return group_sample_weights(data, strategy, group_column="site")
