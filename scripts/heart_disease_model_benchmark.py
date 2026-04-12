#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    GridSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.base import clone  # <--- NEW: robust cloning for CV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

import shap
from tabulate import tabulate


try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
except ImportError:
    TabNetClassifier = None
    torch = None

try:
    import optuna
except ImportError:
    optuna = None

from scipy.stats import wilcoxon, spearmanr

import warnings
warnings.filterwarnings("ignore")

plt.style.use("default")
plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["axes.grid"] = True

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# In[ ]:


PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
ASCII_DIR = RESULTS_DIR / "tables_ascii"
MODELS_DIR = PROJECT_ROOT / "models"

for d in [DATA_DIR, RESULTS_DIR, FIG_DIR, METRICS_DIR, ASCII_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

def save_fig(name, subfolder=""):
    folder = FIG_DIR / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(folder / f"{name}.png", dpi=200)
    plt.close()

def save_csv(df, name, subfolder=""):
    folder = METRICS_DIR / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{name}.csv", index=False)

def save_ascii_table(df, name, subfolder=""):
    folder = ASCII_DIR / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    txt = tabulate(df, headers="keys", tablefmt="psql", floatfmt=".3f")
    with open(folder / f"{name}.txt", "w") as f:
        f.write(txt + "\n")

def compute_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
    }

def plot_and_save_confusion(y_true, y_proba, model_name, subfolder="model_performance"):
    """Plot and save confusion matrix (counts + normalised)."""
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    labels = ["No disease (0)", "Disease (1)"]

    # Raw counts
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} – Confusion matrix (counts)")
    fig.colorbar(im)
    save_fig(f"{model_name}_confusion_counts", subfolder)

    # Normalised
    fig, ax = plt.subplots()
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} – Confusion matrix (normalised)")
    fig.colorbar(im)
    save_fig(f"{model_name}_confusion_norm", subfolder)

    # Classification report table
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "class"})
    save_csv(report_df, f"{model_name}_classification_report", "test")
    save_ascii_table(report_df, f"{model_name}_classification_report", "test")


# In[ ]:


DATA_PATH = PROJECT_ROOT / "heart_disease_uci.csv"  

df_raw = pd.read_csv(DATA_PATH)
print("Raw shape:", df_raw.shape)
display(df_raw.head())

df = df_raw.copy()


df["num"] = (df["num"] > 0).astype(int)


groups_dataset = df["dataset"].astype(str) if "dataset" in df.columns else pd.Series(
    ["unknown"] * len(df), name="dataset"
)


df["sex"] = df["sex"].astype(str).str.lower().map({"male": 1, "female": 0})
df["sex"] = df["sex"].fillna(df["sex"].mode()[0]).astype(int)

for col in ["fbs", "exang"]:
    df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
    df[col] = df[col].fillna(df[col].mode()[0]).astype(int)


num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["ca_missing"] = df["ca"].isna().astype(int)
df["thal_missing"] = df["thal"].isna().astype(int)

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)


multi_cat_cols = ["cp", "restecg", "slope", "thal"]
for col in multi_cat_cols:
    df[col] = df[col].astype(str).fillna(df[col].mode()[0])

drop_cols = ["id"] if "id" in df.columns else []
if "dataset" in df.columns:
    drop_cols.append("dataset")

df_model = df.drop(columns=drop_cols)
df_model = pd.get_dummies(df_model, columns=multi_cat_cols, drop_first=False)

X = df_model.drop(columns=["num"])
y = df_model["num"]

print("Processed shape:", df_model.shape)
print("Target distribution:\n", y.value_counts(normalize=True))
print("Positive class proportion:", y.mean())


df_model.to_parquet(DATA_DIR / "heart_disease_processed.parquet")


# In[ ]:


def eda_plots(df_model):
    
    plt.figure()
    df_model["num"].value_counts().sort_index().plot(kind="bar")
    plt.xticks([0, 1], ["No disease (0)", "Disease (1)"], rotation=0)
    plt.ylabel("Count")
    plt.title("Target distribution")
    save_fig("target_distribution", "eda")

    plt.figure()
    df_model[df_model["num"] == 0]["age"].hist(alpha=0.6, label="num=0")
    df_model[df_model["num"] == 1]["age"].hist(alpha=0.6, label="num=1")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Age distribution by class")
    plt.legend()
    save_fig("age_by_class", "eda")

    plt.figure()
    df_model[df_model["num"] == 0]["oldpeak"].hist(alpha=0.6, label="num=0")
    df_model[df_model["num"] == 1]["oldpeak"].hist(alpha=0.6, label="num=1")
    plt.xlabel("oldpeak (ST depression)")
    plt.ylabel("Count")
    plt.title("Oldpeak distribution by class")
    plt.legend()
    save_fig("oldpeak_by_class", "eda")

    plt.figure()
    df_model[df_model["num"] == 0]["thalch"].hist(alpha=0.6, label="num=0")
    df_model[df_model["num"] == 1]["thalch"].hist(alpha=0.6, label="num=1")
    plt.xlabel("thalch (max HR)")
    plt.ylabel("Count")
    plt.title("Max heart rate by class")
    plt.legend()
    save_fig("thalch_by_class", "eda")

    plt.figure()
    pd.crosstab(df_model["ca"], df_model["num"]).plot(kind="bar")
    plt.title("ca vs Heart Disease")
    plt.xlabel("ca")
    plt.ylabel("Count")
    save_fig("ca_vs_class", "eda")

    plt.figure()
    pd.crosstab(df_model["sex"], df_model["num"]).plot(kind="bar")
    plt.title("Sex vs Heart Disease (0=female,1=male)")
    plt.xlabel("sex")
    plt.ylabel("Count")
    save_fig("sex_vs_class", "eda")

    thal_cols = [c for c in df_model.columns if c.startswith("thal_")]
    thal_df = df_model[thal_cols + ["num"]]
    thal_long = thal_df.melt(id_vars="num", var_name="thal_type", value_name="present")
    thal_long = thal_long[thal_long["present"] == 1]
    plt.figure(figsize=(8, 5))
    (
        thal_long.groupby(["thal_type", "num"])["present"]
        .count()
        .unstack()
        .plot(kind="bar")
    )
    plt.title("Thal result vs Heart Disease")
    plt.ylabel("Count")
    save_fig("thal_vs_class", "eda")

    plt.figure(figsize=(10, 8))
    corr = df_model.corr()
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=90)
    plt.yticks(range(corr.shape[1]), corr.columns)
    plt.title("Correlation heatmap")
    save_fig("corr_heatmap", "eda")

eda_plots(df_model)


# In[ ]:


def cv_evaluate(model, X, y, model_name="model", n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        
        m = clone(model)
        t0 = time.time()
        m.fit(X_tr, y_tr)
        train_time = time.time() - t0

        t0 = time.time()
        y_proba = m.predict_proba(X_val)[:, 1]
        test_time = time.time() - t0

        metrics = compute_metrics(y_val, y_proba)
        metrics.update({"fold": fold, "train_time": train_time, "test_time": test_time})
        rows.append(metrics)

        print(
            f"{model_name} Fold {fold}: "
            f"Acc={metrics['accuracy']:.3f}, Prec={metrics['precision']:.3f}, "
            f"Rec={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, "
            f"AUC={metrics['auc']:.3f}, train={train_time:.3f}s, test={test_time:.4f}s"
        )

    df_cv = pd.DataFrame(rows)
    save_csv(df_cv, f"{model_name}_10fold", "cv")
    save_ascii_table(df_cv.describe().T, f"{model_name}_10fold_summary", "cv")
    return df_cv


def nested_cv(model, param_grid, X, y,
              outer_splits=10, inner_splits=10,
              scoring="roc_auc", model_name="model"):
    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE
    )
    rows = []

    for k, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=RANDOM_STATE + k
        )

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1,
        )

        t0 = time.time()
        grid.fit(X_tr, y_tr)
        search_time = time.time() - t0
        best_model = grid.best_estimator_

        y_proba = best_model.predict_proba(X_te)[:, 1]
        metrics = compute_metrics(y_te, y_proba)
        metrics.update({
            "outer_fold": k,
            "search_time": search_time,
            "best_params": json.dumps(grid.best_params_)
        })
        rows.append(metrics)

        print(
            f"[Nested CV] {model_name} outer fold {k}: "
            f"AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}, "
            f"best_params={grid.best_params_}"
        )

    df_nested = pd.DataFrame(rows)
    save_csv(df_nested, f"{model_name}_nestedCV", "nested_cv")
    save_ascii_table(df_nested.drop(columns=["best_params"]).describe().T,
                     f"{model_name}_nestedCV_summary", "nested_cv")
    return df_nested


# In[ ]:


xgb_base = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
xgb_cv_results = cv_evaluate(xgb_base, X, y, "XGBoost")


lgb_base = LGBMClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    feature_fraction=0.8,
    num_leaves=31,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
lgb_cv_results = cv_evaluate(lgb_base, X, y, "LightGBM")

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        penalty="l2",
        solver="liblinear",
        random_state=RANDOM_STATE
    )),
])

class LRWrapper(LogisticRegression):
    pass  


lr_cv_results = cv_evaluate(lr_pipe, X, y, "LogisticRegression")


if CatBoostClassifier is not None:
    cat_base = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=4,
        learning_rate=0.1,
        iterations=200,
        random_seed=RANDOM_STATE,
        verbose=False,
    )

    def cv_evaluate_cat(model, X, y, model_name="CatBoost", n_splits=10):
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )
        rows = []
        X_np = X.values
        y_np = y.values
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_np, y_np), 1):
            X_tr, X_val = X_np[tr_idx], X_np[val_idx]
            y_tr, y_val = y_np[tr_idx], y_np[val_idx]

            m = clone(model)
            t0 = time.time()
            m.fit(X_tr, y_tr)
            train_time = time.time() - t0

            t0 = time.time()
            y_proba = m.predict_proba(X_val)[:, 1]
            test_time = time.time() - t0

            metrics = compute_metrics(y_val, y_proba)
            metrics.update({"fold": fold, "train_time": train_time, "test_time": test_time})
            rows.append(metrics)

            print(
                f"{model_name} Fold {fold}: "
                f"Acc={metrics['accuracy']:.3f}, Prec={metrics['precision']:.3f}, "
                f"Rec={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, "
                f"AUC={metrics['auc']:.3f}, train={train_time:.3f}s, test={test_time:.4f}s"
            )
        df_cv = pd.DataFrame(rows)
        save_csv(df_cv, f"{model_name}_10fold", "cv")
        save_ascii_table(df_cv.describe().T, f"{model_name}_10fold_summary", "cv")
        return df_cv

    cat_cv_results = cv_evaluate_cat(cat_base, X, y)
else:
    print("CatBoost not installed – skipping CatBoost CV.")


# In[ ]:


lgb_nested_model = LGBMClassifier(
    objective="binary",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
lgb_param_grid = {
    "n_estimators": [200, 400],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4],
    "num_leaves": [15, 31],
    "subsample": [0.8, 1.0],
    "feature_fraction": [0.8, 1.0],
}
lgb_nested_results = nested_cv(
    lgb_nested_model, lgb_param_grid, X, y, model_name="LightGBM"
)

lr_nested_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )),
])
lr_param_grid = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l2"],
}
lr_nested_results = nested_cv(
    lr_nested_model, lr_param_grid, X, y, model_name="LogisticRegression"
)


def extract_best_params(nested_df, model_name):
    params_df = nested_df["best_params"].apply(json.loads).apply(pd.Series)
    params_df["outer_fold"] = nested_df["outer_fold"].values
    save_csv(params_df, f"{model_name}_nested_best_params", "nested_cv")
    save_ascii_table(params_df.describe(include="all").T,
                     f"{model_name}_nested_best_params_summary", "nested_cv")
    return params_df

lgb_best_params_df = extract_best_params(lgb_nested_results, "LightGBM")
lr_best_params_df = extract_best_params(lr_nested_results, "LogisticRegression")


for param in lgb_best_params_df.columns:
    if param == "outer_fold":
        continue
    counts = lgb_best_params_df[param].value_counts().sort_index()
    plt.figure()
    counts.plot(kind="bar")
    plt.ylabel("Count (outer folds)")
    plt.title(f"LGBM nested CV – selected values for {param}")
    save_fig(f"lgbm_nested_param_dist_{param}", "nested_cv")


if optuna is not None:
    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0)
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        aucs = []
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            m = LGBMClassifier(
                objective="binary",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                **params
            )
            m.fit(X_tr, y_tr)
            y_proba = m.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, y_proba))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_lgb, n_trials=30, show_progress_bar=False)
    print("Optuna best params for LightGBM:", study.best_params)
    with open(METRICS_DIR / "optuna_lgbm_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
else:
    print("Optuna not installed – skipping Bayesian optimisation.")


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)


lgb_final = LGBMClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    feature_fraction=0.8,
    min_child_samples=20,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
t0 = time.time()
lgb_final.fit(X_train, y_train)
train_time_lgb = time.time() - t0
y_proba_lgb = lgb_final.predict_proba(X_test)[:, 1]
metrics_lgb = compute_metrics(y_test, y_proba_lgb)
metrics_lgb["train_time"] = train_time_lgb
metrics_lgb["brier"] = brier_score_loss(y_test, y_proba_lgb)
print("LightGBM hold-out metrics:", metrics_lgb)


lr_final = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE
    )),
])
t0 = time.time()
lr_final.fit(X_train, y_train)
train_time_lr = time.time() - t0
y_proba_lr = lr_final.predict_proba(X_test)[:, 1]
metrics_lr = compute_metrics(y_test, y_proba_lr)
metrics_lr["train_time"] = train_time_lr
metrics_lr["brier"] = brier_score_loss(y_test, y_proba_lr)
print("Logistic Regression hold-out metrics:", metrics_lr)


xgb_final = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
t0 = time.time()
xgb_final.fit(X_train, y_train)
train_time_xgb = time.time() - t0
y_proba_xgb = xgb_final.predict_proba(X_test)[:, 1]
metrics_xgb = compute_metrics(y_test, y_proba_xgb)
metrics_xgb["train_time"] = train_time_xgb
metrics_xgb["brier"] = brier_score_loss(y_test, y_proba_xgb)
print("XGBoost hold-out metrics:", metrics_xgb)


if TabNetClassifier is not None and torch is not None:
    tabnet_final = TabNetClassifier(
        seed=RANDOM_STATE,
        verbose=0
    )
   
    X_tr_np = X_train.to_numpy(dtype=np.float32)
    y_tr_np = y_train.to_numpy(dtype=np.int64)
    X_te_np = X_test.to_numpy(dtype=np.float32)
    y_te_np = y_test.to_numpy(dtype=np.int64)

    t0 = time.time()
    tabnet_final.fit(
        X_tr_np, y_tr_np,
        eval_set=[(X_te_np, y_te_np)],
        eval_metric=["auc"],
        max_epochs=200,
        patience=30,
        batch_size=128,
        virtual_batch_size=64,
    )
    train_time_tabnet = time.time() - t0
    y_proba_tabnet = tabnet_final.predict_proba(X_te_np)[:, 1]
    metrics_tabnet = compute_metrics(y_te_np, y_proba_tabnet)
    metrics_tabnet["train_time"] = train_time_tabnet
    metrics_tabnet["brier"] = brier_score_loss(y_te_np, y_proba_tabnet)
    print("TabNet hold-out metrics:", metrics_tabnet)
else:
    metrics_tabnet = None
    X_tr_np = X_te_np = y_tr_np = y_te_np = None
    print("TabNet not installed – skipping TabNet model.")

rows = [
    {"model": "LightGBM", **metrics_lgb},
    {"model": "LogisticRegression", **metrics_lr},
    {"model": "XGBoost", **metrics_xgb},
]
if metrics_tabnet is not None:
    rows.append({"model": "TabNet", **metrics_tabnet})

df_holdout = pd.DataFrame(rows)
save_csv(df_holdout, "holdout_models", "test")
save_ascii_table(df_holdout, "holdout_models", "test")



def bootstrap_auc_ci(y_true, y_proba, n_boot=1000, alpha=0.95, random_state=RANDOM_STATE):
    """Non-parametric bootstrap CI for ROC-AUC."""
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    aucs = np.array(aucs)
    lower = np.percentile(aucs, (1 - alpha) / 2 * 100)
    upper = np.percentile(aucs, (1 + alpha) / 2 * 100)
    return aucs.mean(), lower, upper

bootstrap_records = []
for name, y_true, y_proba in [
    ("LightGBM", y_test.values, y_proba_lgb),
    ("LogisticRegression", y_test.values, y_proba_lr),
    ("XGBoost", y_test.values, y_proba_xgb),
]:
    mean_auc, ci_lo, ci_hi = bootstrap_auc_ci(y_true, y_proba)
    print(f"{name} bootstrap AUC: mean={mean_auc:.3f}, "
          f"{int(95)}% CI=({ci_lo:.3f}, {ci_hi:.3f})")
    bootstrap_records.append({
        "model": name,
        "auc_boot_mean": mean_auc,
        "auc_ci_low": ci_lo,
        "auc_ci_high": ci_hi,
    })

if metrics_tabnet is not None:
    mean_auc, ci_lo, ci_hi = bootstrap_auc_ci(y_te_np, y_proba_tabnet)
    print(f"TabNet bootstrap AUC: mean={mean_auc:.3f}, "
          f"{int(95)}% CI=({ci_lo:.3f}, {ci_hi:.3f})")
    bootstrap_records.append({
        "model": "TabNet",
        "auc_boot_mean": mean_auc,
        "auc_ci_low": ci_lo,
        "auc_ci_high": ci_hi,
    })

df_auc_ci = pd.DataFrame(bootstrap_records)
save_csv(df_auc_ci, "holdout_auc_bootstrap_ci", "test")
save_ascii_table(df_auc_ci, "holdout_auc_bootstrap_ci", "test")


# In[ ]:


def _bootstrap_metric_ci(y_true, y_proba, metric_fn, n_boot=1000, 
                         alpha=0.95, random_state=RANDOM_STATE):
    """
    Simple non-parametric bootstrap CI for a metric(theta) = metric_fn(y_true, y_proba).
    Returns (mean_bootstrap, lower, upper).
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    idx = np.arange(n)

    stats = []
    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=n, replace=True)
        m = metric_fn(y_true[sample_idx], y_proba[sample_idx])
        stats.append(m)

    stats = np.asarray(stats)
    lower = np.quantile(stats, (1 - alpha) / 2)
    upper = np.quantile(stats, 1 - (1 - alpha) / 2)
    return float(stats.mean()), float(lower), float(upper)


def _f1_from_proba(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return f1_score(y_true, y_pred)


boot_rows = []
model_probas = [
    ("LightGBM", y_proba_lgb),
    ("LogReg",   y_proba_lr),
    ("XGBoost",  y_proba_xgb),
]
if metrics_tabnet is not None:
    model_probas.append(("TabNet", y_proba_tabnet))

for name, yp in model_probas:
    y_true = y_test.values

    auc_mean, auc_lo, auc_hi = _bootstrap_metric_ci(
        y_true, yp, roc_auc_score, n_boot=1000
    )
    f1_mean, f1_lo, f1_hi = _bootstrap_metric_ci(
        y_true, yp, _f1_from_proba, n_boot=1000
    )

    boot_rows.append({
        "model": name,
        "auc_mean": auc_mean,
        "auc_ci_low": auc_lo,
        "auc_ci_high": auc_hi,
        "f1_mean": f1_mean,
        "f1_ci_low": f1_lo,
        "f1_ci_high": f1_hi,
    })

df_boot = pd.DataFrame(boot_rows)
print("Bootstrap 95% CIs for AUC and F1 on hold-out:")
display(df_boot)

save_csv(df_boot, "holdout_bootstrap_ci", "test")
save_ascii_table(df_boot, "holdout_bootstrap_ci", "test")


# In[ ]:


def lgb_parallelism_comparison():
    configs = [("single_thread", 1), ("multi_thread", -1)]
    records = []
    for label, n_jobs in configs:
        model = LGBMClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            feature_fraction=0.8,
            min_child_samples=20,
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
        )
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t1 = time.time()
        _ = model.predict_proba(X_test)[:, 1]
        infer_time = time.time() - t1

        records.append({
            "mode": label,
            "n_jobs": n_jobs,
            "train_time_s": train_time,
            "infer_time_s": infer_time
        })

    df = pd.DataFrame(records)
    print("\nLightGBM parallelism comparison:")
    display(df)

    save_csv(df, "lgbm_parallelism", "test")
    save_ascii_table(df, "lgbm_parallelism", "test")

    plt.figure()
    plt.bar(df["mode"], df["train_time_s"])
    plt.ylabel("Training time (s)")
    plt.title("LightGBM: single-thread vs multi-thread training time")
    save_fig("lgbm_parallel_train_time", "model_performance")

    plt.figure()
    plt.bar(df["mode"], df["infer_time_s"])
    plt.ylabel("Inference time (s) for full test set")
    plt.title("LightGBM: single-thread vs multi-thread inference time")
    save_fig("lgbm_parallel_infer_time", "model_performance")

lgb_parallelism_comparison()


# In[ ]:


from dask.distributed import Client
import dask.dataframe as dd
from lightgbm import dask as lgb_dask  


client = Client()
print(client)


dX_train = dd.from_pandas(X_train.astype("float32"), npartitions=4)
dy_train = dd.from_pandas(y_train.astype("int64"), npartitions=4)
dX_test = dd.from_pandas(X_test.astype("float32"), npartitions=4)

dask_results = []


try:
    dask_clf = lgb_dask.DaskLGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=400,
        max_depth=4,
        subsample=0.8,
        feature_fraction=0.8,
        random_state=RANDOM_STATE,
    )

    t0 = time.time()
    dask_clf = dask_clf.fit(dX_train, dy_train)
    train_time_dask_clf = time.time() - t0

    y_proba_dask_clf = dask_clf.predict_proba(dX_test)[:, 1].compute()

    metrics_dask_clf = compute_metrics(y_test.values, y_proba_dask_clf)
    metrics_dask_clf["train_time"] = train_time_dask_clf

    print("Distributed LightGBM (DaskLGBMClassifier) hold-out metrics:",
          metrics_dask_clf)

    df_dask_clf = pd.DataFrame(
        [{"model": "LightGBM_DaskClassifier", **metrics_dask_clf}]
    )
    save_csv(df_dask_clf, "holdout_lightgbm_dask_classifier", subfolder="test")
    save_ascii_table(df_dask_clf,
                     "holdout_lightgbm_dask_classifier",
                     subfolder="test")

    dask_results.append(df_dask_clf)

except Exception as e:
    print("\n[WARN] DaskLGBMClassifier training failed:", repr(e))
    print("       This is usually due to a LightGBM MPI build / machine-list issue.")
    dask_clf = None



try:
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 4,
        "subsample": 0.8,
        "feature_fraction": 0.8,
        "verbosity": -1,
        "seed": RANDOM_STATE,
    }

    t0 = time.time()
    out = lgb_dask.train(
        client,
        params,
        dX_train,
        dy_train,
        num_boost_round=400,
    )
    train_time_dask_low = time.time() - t0

    booster = out["booster"]
    booster_local = booster.to_local()

    y_proba_dask_low = booster_local.predict(X_test.to_numpy(dtype="float32"))

    metrics_dask_low = compute_metrics(y_test.values, y_proba_dask_low)
    metrics_dask_low["train_time"] = train_time_dask_low

    print("Distributed LightGBM (dask.train) hold-out metrics:",
          metrics_dask_low)

    df_dask_low = pd.DataFrame(
        [{"model": "LightGBM_DaskTrain", **metrics_dask_low}]
    )
    save_csv(df_dask_low, "holdout_lightgbm_dask_train", subfolder="test")
    save_ascii_table(df_dask_low,
                     "holdout_lightgbm_dask_train",
                     subfolder="test")

    dask_results.append(df_dask_low)

except Exception as e:
    print("\n[WARN] lgb.dask.train() failed:", repr(e))
    print("       Again, likely due to LightGBM's MPI/network configuration.")
    booster = None

if dask_results:
    df_all_dask = pd.concat(dask_results, ignore_index=True)
    save_csv(df_all_dask, "holdout_lightgbm_dask_all", subfolder="test")
    save_ascii_table(df_all_dask, "holdout_lightgbm_dask_all", subfolder="test")
else:
    print("\nNo Dask LightGBM results available – falling back on single-node "
          "parallel vs single-thread experiments for the report.")


if dask_results:
    df_all_dask = pd.concat(dask_results, ignore_index=True)
    save_csv(df_all_dask, "holdout_lightgbm_dask_all", subfolder="test")
    save_ascii_table(df_all_dask, "holdout_lightgbm_dask_all", subfolder="test")

    
    single_node_row = {
        "model": "LightGBM_single_node",
        "accuracy": metrics_lgb["accuracy"],
        "precision": metrics_lgb["precision"],
        "recall": metrics_lgb["recall"],
        "f1": metrics_lgb["f1"],
        "auc": metrics_lgb["auc"],
        "train_time": metrics_lgb["train_time"],
    }

    df_dask_summary = pd.concat(
        [pd.DataFrame([single_node_row]), df_all_dask],
        ignore_index=True
    )

    save_csv(df_dask_summary, "lightgbm_distributed_summary", subfolder="test")
    save_ascii_table(df_dask_summary, "lightgbm_distributed_summary", subfolder="test")
    print("Distributed vs single-node LightGBM summary:")
    display(df_dask_summary)
else:
    print("\nNo Dask LightGBM results available – falling back on single-node "
          "parallel vs single-thread experiments for the report.")



# In[ ]:


def plot_training_curves_xgb():
    """
    XGBoost: track training/validation logloss, AUC and accuracy vs iteration.
    """
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
      
        eval_metric=["logloss", "auc", "error"],
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    evals_result = model.evals_result()
    train_key, valid_key = list(evals_result.keys())

    train_res = evals_result[train_key]
    valid_res = evals_result[valid_key]

    iters = np.arange(len(train_res["logloss"]))

   
    train_acc = 1.0 - np.array(train_res["error"])
    valid_acc = 1.0 - np.array(valid_res["error"])

    df_curves = pd.DataFrame({
        "iter": iters,
        "train_logloss": train_res["logloss"],
        "valid_logloss": valid_res["logloss"],
        "train_auc": train_res["auc"],
        "valid_auc": valid_res["auc"],
        "train_acc": train_acc,
        "valid_acc": valid_acc,
    })
    save_csv(df_curves, "xgb_training_curves", "model_performance")

    
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_logloss"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_logloss"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("Logloss")
    plt.title("XGBoost – training vs validation logloss")
    plt.legend()
    save_fig("xgb_train_valid_logloss", "model_performance")

    
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_auc"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_auc"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title("XGBoost – training vs validation AUC")
    plt.legend()
    save_fig("xgb_train_valid_auc", "model_performance")

    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_acc"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_acc"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("XGBoost – training vs validation accuracy")
    plt.legend()
    save_fig("xgb_train_valid_accuracy", "model_performance")


def plot_training_curves_lgbm():
    """
    LightGBM: track training/validation binary_logloss, AUC and accuracy vs iteration.
    """
    model = LGBMClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        feature_fraction=0.8,
        min_child_samples=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

 
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric=["binary_logloss", "auc", "binary_error"],
            verbose=False,
        )
    except TypeError:
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric=["binary_logloss", "auc", "binary_error"],
        )

    evals_result = model.evals_result_
    train_key = "training"
    valid_key = [k for k in evals_result.keys() if k != train_key][0]

    train_res = evals_result[train_key]
    valid_res = evals_result[valid_key]

    iters = np.arange(len(train_res["binary_logloss"]))

    
    train_acc = 1.0 - np.array(train_res["binary_error"])
    valid_acc = 1.0 - np.array(valid_res["binary_error"])

    df_curves = pd.DataFrame({
        "iter": iters,
        "train_logloss": train_res["binary_logloss"],
        "valid_logloss": valid_res["binary_logloss"],
        "train_auc": train_res["auc"],
        "valid_auc": valid_res["auc"],
        "train_acc": train_acc,
        "valid_acc": valid_acc,
    })
    save_csv(df_curves, "lgbm_training_curves", "model_performance")

    
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_logloss"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_logloss"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("Binary logloss")
    plt.title("LightGBM – training vs validation logloss")
    plt.legend()
    save_fig("lgbm_train_valid_logloss", "model_performance")

    
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_auc"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_auc"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title("LightGBM – training vs validation AUC")
    plt.legend()
    save_fig("lgbm_train_valid_auc", "model_performance")

 
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_acc"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_acc"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("LightGBM – training vs validation accuracy")
    plt.legend()
    save_fig("lgbm_train_valid_accuracy", "model_performance")



plot_training_curves_xgb()
plot_training_curves_lgbm()



# In[ ]:


def estimate_latency(model, X_input, n_runs=30, name="model"):
    
    start = time.time()
    for _ in range(n_runs):
        _ = model.predict_proba(X_input)
    total = time.time() - start
    avg = total / (n_runs * X_input.shape[0])
    print(f"{name}: avg latency per sample ~ {avg*1e6:.2f} µs")
    return avg

complexity_records = []


n_features = X_train.shape[1]
logreg_params = n_features + 1  
lat_lr = estimate_latency(lr_final, X_test, name="LogisticRegression")
complexity_records.append({
    "model": "LogisticRegression",
    "approx_params": logreg_params,
    "approx_ops_per_sample": 2 * n_features,
    "latency_s_per_sample": lat_lr,
    "auc": metrics_lr["auc"],
    "brier": metrics_lr["brier"],
})


n_trees_lgb = lgb_final.n_estimators
max_depth_lgb = lgb_final.max_depth if lgb_final.max_depth is not None else 6
lat_lgb = estimate_latency(lgb_final, X_test, name="LightGBM")
complexity_records.append({
    "model": "LightGBM",
    "approx_params": n_trees_lgb,
    "approx_ops_per_sample": n_trees_lgb * max_depth_lgb,
    "latency_s_per_sample": lat_lgb,
    "auc": metrics_lgb["auc"],
    "brier": metrics_lgb["brier"],
})


n_trees_xgb = xgb_final.n_estimators
max_depth_xgb = xgb_final.max_depth if xgb_final.max_depth is not None else 6
lat_xgb = estimate_latency(xgb_final, X_test, name="XGBoost")
complexity_records.append({
    "model": "XGBoost",
    "approx_params": n_trees_xgb,
    "approx_ops_per_sample": n_trees_xgb * max_depth_xgb,
    "latency_s_per_sample": lat_xgb,
    "auc": metrics_xgb["auc"],
    "brier": metrics_xgb["brier"],
})


if metrics_tabnet is not None and TabNetClassifier is not None:
    try:
        param_count = sum(p.numel() for p in tabnet_final.network.parameters())
        lat_tab = estimate_latency(tabnet_final, X_te_np, name="TabNet")
        complexity_records.append({
            "model": "TabNet",
            "approx_params": param_count,
            "approx_ops_per_sample": 2 * param_count,
            "latency_s_per_sample": lat_tab,
            "auc": metrics_tabnet["auc"],
            "brier": metrics_tabnet["brier"],
        })
    except Exception as e:
        print("Could not compute TabNet param count:", e)

complexity_df = pd.DataFrame(complexity_records)
print("\nApproximate complexity, latency and quality:")
display(complexity_df)

save_csv(complexity_df, "model_complexity_latency", "test")
save_ascii_table(complexity_df, "model_complexity_latency", "test")

plt.figure()
plt.scatter(complexity_df["latency_s_per_sample"] * 1e6, complexity_df["auc"])
for _, row in complexity_df.iterrows():
    plt.text(row["latency_s_per_sample"] * 1e6, row["auc"], row["model"])
plt.xlabel("Latency per sample (µs)")
plt.ylabel("AUC")
plt.title("Latency vs AUC across models")
save_fig("models_latency_vs_auc", "model_performance")


# In[ ]:


def plot_roc_pr_calibration_and_confusion():

    plt.figure()
    fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_proba_lgb)
    plt.plot(fpr_lgb, tpr_lgb, label=f"LGBM (AUC={metrics_lgb['auc']:.2f})")
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={metrics_lr['auc']:.2f})")
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGB (AUC={metrics_xgb['auc']:.2f})")
    if metrics_tabnet is not None:
        fpr_tab, tpr_tab, _ = roc_curve(y_test, y_proba_tabnet)
        plt.plot(fpr_tab, tpr_tab, label=f"TabNet (AUC={metrics_tabnet['auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curves")
    plt.legend()
    save_fig("roc_comparison", "model_performance")

    
    plt.figure()
    prec_lgb, rec_lgb, _ = precision_recall_curve(y_test, y_proba_lgb)
    plt.plot(rec_lgb, prec_lgb, label="LGBM")
    prec_lr_, rec_lr_, _ = precision_recall_curve(y_test, y_proba_lr)
    plt.plot(rec_lr_, prec_lr_, label="LR")
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_proba_xgb)
    plt.plot(rec_xgb, prec_xgb, label="XGB")
    if metrics_tabnet is not None:
        prec_tab, rec_tab, _ = precision_recall_curve(y_test, y_proba_tabnet)
        plt.plot(rec_tab, prec_tab, label="TabNet")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.legend()
    save_fig("pr_comparison", "model_performance")

    
    thresholds = np.linspace(0.1, 0.9, 17)
    f1_scores = [
        f1_score(y_test, (y_proba_lgb >= t).astype(int)) for t in thresholds
    ]
    plt.figure()
    plt.plot(thresholds, f1_scores, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title("LGBM – F1 vs threshold")
    save_fig("lgbm_f1_vs_threshold", "model_performance")

    
    plt.figure()
    for name, y_proba in [("LGBM", y_proba_lgb),
                          ("LR", y_proba_lr),
                          ("XGB", y_proba_xgb)]:
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curves")
    plt.legend()
    save_fig("calibration_curves", "model_performance")

    
    plot_and_save_confusion(y_test, y_proba_lgb, "LightGBM")
    plot_and_save_confusion(y_test, y_proba_lr, "LogisticRegression")
    plot_and_save_confusion(y_test, y_proba_xgb, "XGBoost")
    if metrics_tabnet is not None:
        plot_and_save_confusion(y_te_np, y_proba_tabnet, "TabNet")

plot_roc_pr_calibration_and_confusion()


# In[ ]:


from sklearn.linear_model import LogisticRegression  

def brier_decomposition(y_true, y_proba, n_bins=10):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    N = len(y_true)

    brier = np.mean((y_proba - y_true) ** 2)
    base_rate = y_true.mean()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_proba, bins) - 1  

    reliability = 0.0
    resolution = 0.0

    for b in range(n_bins):
        mask = bin_ids == b
        n_k = mask.sum()
        if n_k == 0:
            continue
        p_k = y_proba[mask].mean()
        o_k = y_true[mask].mean()
        w_k = n_k / N
        reliability += w_k * (p_k - o_k) ** 2
        resolution += w_k * (o_k - base_rate) ** 2

    uncertainty = base_rate * (1.0 - base_rate)
    return {
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
    }


def calibration_slope_intercept(y_true, y_proba):
    """
    Fit a calibration model:
        logit(y_true) ~ a + b * logit(p_hat)
    Slope b ~ 1 and intercept a ~ 0 => good calibration.
    """
    eps = 1e-6
    p = np.clip(np.asarray(y_proba), eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    y_true = np.asarray(y_true)

   
    lr_cal = LogisticRegression(
        fit_intercept=True,
        penalty="l2",
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    )
    lr_cal.fit(logit_p, y_true)
    slope = float(lr_cal.coef_[0, 0])
    intercept = float(lr_cal.intercept_[0])
    return slope, intercept


brier_rows = []
model_probas = [
    ("LightGBM", y_proba_lgb),
    ("LogReg",   y_proba_lr),
    ("XGBoost",  y_proba_xgb),
]
if metrics_tabnet is not None:
    model_probas.append(("TabNet", y_proba_tabnet))

for name, yp in model_probas:
    dec = brier_decomposition(y_test.values, yp, n_bins=10)
    slope, intercept = calibration_slope_intercept(y_test.values, yp)
    row = {
        "model": name,
        **dec,
        "cal_slope": slope,
        "cal_intercept": intercept,
    }
    brier_rows.append(row)

df_brier = pd.DataFrame(brier_rows)
print("Brier decomposition and calibration slope/intercept:")
display(df_brier)

save_csv(df_brier, "brier_decomposition_and_calibration", "test")
save_ascii_table(df_brier, "brier_decomposition_and_calibration", "test")


plt.figure()
x = np.arange(len(df_brier))
width = 0.25
plt.bar(x - width, df_brier["reliability"], width, label="Reliability")
plt.bar(x,         df_brier["resolution"], width, label="Resolution")
plt.bar(x + width, df_brier["uncertainty"], width, label="Uncertainty")
plt.xticks(x, df_brier["model"])
plt.ylabel("Component value")
plt.title("Brier score decomposition components")
plt.legend()
save_fig("brier_decomposition_components", "model_performance")



# In[ ]:


def threshold_grid_analysis(y_true, y_proba, cost_fn=5.0, cost_fp=1.0):
    """
    Compute TPR, FPR, F1 and expected cost over a grid of thresholds.
    cost_fn: cost of a false negative (missed disease)
    cost_fp: cost of a false positive (unnecessary alarm/test)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    thresholds = np.linspace(0.05, 0.95, 19)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = f1_score(y_true, y_pred)
        exp_cost = cost_fn * fn + cost_fp * fp

        rows.append({
            "threshold": t,
            "TPR": tpr,
            "FPR": fpr,
            "F1": f1,
            "Youden_J": tpr - fpr,
            "expected_cost": exp_cost,
        })

    return pd.DataFrame(rows)



thresh_df_lgb = threshold_grid_analysis(y_test, y_proba_lgb, cost_fn=5.0, cost_fp=1.0)
print("Threshold analysis – LightGBM:")
display(thresh_df_lgb.head())

save_csv(thresh_df_lgb, "lgbm_threshold_analysis", "test")
save_ascii_table(thresh_df_lgb, "lgbm_threshold_analysis", "test")


plt.figure()
plt.plot(thresh_df_lgb["threshold"], thresh_df_lgb["Youden_J"], marker="o")
plt.xlabel("Threshold")
plt.ylabel("Youden J = TPR - FPR")
plt.title("LightGBM – Youden J vs threshold")
save_fig("lgbm_youdenJ_vs_threshold", "model_performance")

plt.figure()
plt.plot(thresh_df_lgb["threshold"], thresh_df_lgb["expected_cost"], marker="o")
plt.xlabel("Threshold")
plt.ylabel("Expected cost (FN cost=5, FP cost=1)")
plt.title("LightGBM – cost-sensitive risk vs threshold")
save_fig("lgbm_cost_vs_threshold", "model_performance")


# In[ ]:


def model_comparison_stats_and_fairness():
    
    auc_xgb = xgb_cv_results["auc"]
    auc_lgb = lgb_cv_results["auc"]
    stat, p_value = wilcoxon(auc_xgb, auc_lgb)
    print("Wilcoxon test XGBoost vs LightGBM (AUC): stat=", stat, "p=", p_value)
    df_wilcox = pd.DataFrame([{"statistic": stat, "p_value": p_value}])
    save_csv(df_wilcox, "wilcoxon_xgb_vs_lgb_auc", "test")
    save_ascii_table(df_wilcox, "wilcoxon_xgb_vs_lgb_auc", "test")

    
    if "sex" in X_test.columns:
        male_idx = X_test["sex"] == 1
        female_idx = X_test["sex"] == 0

        def subgroup_metrics(mask, y_proba):
            return compute_metrics(y_test[mask], y_proba[mask])

        def tpr_fpr(y_true, y_proba, threshold=0.5):
            y_pred = (y_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            return tpr, fpr

        metrics_rows = []

        for model_name, y_proba in [
            ("LightGBM", y_proba_lgb),
            ("LogReg", y_proba_lr),
            ("XGBoost", y_proba_xgb),
        ]:
            male_metrics = subgroup_metrics(male_idx, y_proba)
            female_metrics = subgroup_metrics(female_idx, y_proba)
            male_tpr, male_fpr = tpr_fpr(y_test[male_idx], y_proba[male_idx])
            female_tpr, female_fpr = tpr_fpr(y_test[female_idx], y_proba[female_idx])

            metrics_rows.append({
                "model": model_name,
                "group": "Male",
                **male_metrics,
                "tpr": male_tpr,
                "fpr": male_fpr,
            })
            metrics_rows.append({
                "model": model_name,
                "group": "Female",
                **female_metrics,
                "tpr": female_tpr,
                "fpr": female_fpr,
            })
            metrics_rows.append({
                "model": model_name,
                "group": "Male-Female_delta",
                "accuracy": male_metrics["accuracy"] - female_metrics["accuracy"],
                "precision": male_metrics["precision"] - female_metrics["precision"],
                "recall": male_metrics["recall"] - female_metrics["recall"],
                "f1": male_metrics["f1"] - female_metrics["f1"],
                "auc": male_metrics["auc"] - female_metrics["auc"],
                "tpr": male_tpr - female_tpr,
                "fpr": male_fpr - female_fpr,
            })

        df_sub = pd.DataFrame(metrics_rows)
        print("\nSex-based subgroup metrics and fairness deltas:")
        display(df_sub)
        save_csv(df_sub, "subgroup_sex_metrics_and_fairness", "test")
        save_ascii_table(df_sub, "subgroup_sex_metrics_and_fairness", "test")

    
        model_shortnames = {
            "LightGBM": "lgbm",
            "LogReg": "logreg",
            "XGBoost": "xgb",
        }
        for model_name, short in model_shortnames.items():
            df_delta = df_sub[(df_sub["model"] == model_name) &
                              (df_sub["group"] == "Male-Female_delta")]
            if not df_delta.empty:
                row = df_delta.iloc[0]
                plt.figure()
                plt.bar(["TPR_delta", "FPR_delta"], [row["tpr"], row["fpr"]])
                plt.axhline(0.0, color="black", linewidth=0.8)
                plt.ylabel("Male minus Female")
                plt.title(f"{model_name} sex fairness – Male minus Female")
                save_fig(f"{short}_sex_fairness_deltas", "model_performance")

        
        df_deltas = df_sub[df_sub["group"] == "Male-Female_delta"].copy()

        plt.figure(figsize=(8, 5))
        x = np.arange(len(df_deltas))
        width = 0.35
        plt.bar(x - width / 2, df_deltas["tpr"], width, label="ΔTPR (Male - Female)")
        plt.bar(x + width / 2, df_deltas["fpr"], width, label="ΔFPR (Male - Female)")
        plt.xticks(x, df_deltas["model"])
        plt.axhline(0.0, linestyle="--")
        plt.ylabel("Difference")
        plt.title("Sex fairness across models – Male minus Female")
        plt.legend()
        save_fig("models_sex_fairness_deltas", "model_performance")

model_comparison_stats_and_fairness()




# In[ ]:


shap.initjs()

explainer_lgb = shap.TreeExplainer(lgb_final)
shap_values_lgb_raw = explainer_lgb.shap_values(X_test)

if isinstance(shap_values_lgb_raw, list):
    shap_values_lgb = shap_values_lgb_raw[1]
    base_value_lgb = explainer_lgb.expected_value[1]
else:
    shap_values_lgb = shap_values_lgb_raw
    base_value_lgb = explainer_lgb.expected_value

print("LGBM SHAP shape:", shap_values_lgb.shape)

shap.summary_plot(shap_values_lgb, X_test, plot_type="bar", show=False)
save_fig("shap_lgbm_bar", "shap")
shap.summary_plot(shap_values_lgb, X_test, show=False)
save_fig("shap_lgbm_beeswarm", "shap")

for feat in ["oldpeak", "thalch", "ca"]:
    if feat in X_test.columns:
        shap.dependence_plot(feat, shap_values_lgb, X_test, show=False)
        save_fig(f"shap_lgbm_dependence_{feat}", "shap")

i_example = 0
force_plot_lgb = shap.force_plot(
    base_value_lgb,
    shap_values_lgb[i_example, :],
    X_test.iloc[i_example, :],
)


try:
    shap_interactions_raw = explainer_lgb.shap_interaction_values(X_test)
    if isinstance(shap_interactions_raw, list):
        shap_interactions = shap_interactions_raw[1]
    else:
        shap_interactions = shap_interactions_raw

    mean_abs_inter = np.mean(np.abs(shap_interactions), axis=(0, 1))
    inter_importance = pd.Series(mean_abs_inter, index=X_test.columns).sort_values(ascending=False)
    save_csv(
        inter_importance.reset_index().rename(columns={"index": "feature", 0: "mean_abs_inter"}),
        "lgbm_shap_interaction_importance", "test"
    )

    print("Top 10 interaction-heavy features (LGBM):")
    print(inter_importance.head(10))

    if "oldpeak" in X_test.columns and "exang" in X_test.columns:
        shap.dependence_plot(("oldpeak", "exang"), shap_interactions, X_test, show=False)
        save_fig("shap_lgbm_interaction_oldpeak_exang", "shap")
except Exception as e:
    print("SHAP interaction values skipped:", e)


n_boot = 20
topk = 10
feature_counts = pd.Series(0, index=X.columns)
for b in range(n_boot):
    X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=RANDOM_STATE + b)
    lgb_boot = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        feature_fraction=0.8,
        num_leaves=31,
        n_jobs=-1,
        random_state=RANDOM_STATE + b,
    )
    lgb_boot.fit(X_boot, y_boot)
    expl_boot = shap.TreeExplainer(lgb_boot)
    sv_boot_raw = expl_boot.shap_values(X_boot)
    if isinstance(sv_boot_raw, list):
        sv_boot = sv_boot_raw[1]
    else:
        sv_boot = sv_boot_raw
    mean_abs_boot = np.mean(np.abs(sv_boot), axis=0)
    imp_boot = pd.Series(mean_abs_boot, index=X.columns).sort_values(ascending=False)
    feature_counts[imp_boot.head(topk).index] += 1

save_csv(
    feature_counts.reset_index().rename(columns={"index": "feature", 0: "freq"}),
    "lgbm_shap_bootstrap_frequency", "test"
)



# In[ ]:


X_train_scaled = lr_final.named_steps["scaler"].transform(X_train)
X_test_scaled = lr_final.named_steps["scaler"].transform(X_test)
X_train_lr = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_lr = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

log_reg_final = lr_final.named_steps["clf"]
explainer_lr = shap.LinearExplainer(log_reg_final, X_train_lr)
sv_lr = explainer_lr.shap_values(X_test_lr)
if hasattr(sv_lr, "values"):
    sv_lr = sv_lr.values
sv_lr = np.array(sv_lr)
if sv_lr.ndim == 3:
    sv_lr = sv_lr[1]
shap_values_lr = sv_lr.astype(float)
print("Logistic SHAP shape:", shap_values_lr.shape)

shap.summary_plot(shap_values_lr, X_test_lr, plot_type="bar", show=False)
save_fig("shap_lr_bar", "shap")
shap.summary_plot(shap_values_lr, X_test_lr, show=False)
save_fig("shap_lr_beeswarm", "shap")

force_plot_lr = shap.force_plot(
    explainer_lr.expected_value,
    shap_values_lr[i_example, :],
    X_test_lr.iloc[i_example, :],
)


# In[ ]:


mean_abs_shap_lgb = np.mean(np.abs(shap_values_lgb), axis=0)
mean_abs_shap_lr = np.mean(np.abs(shap_values_lr), axis=0)

imp_lgb = pd.Series(mean_abs_shap_lgb, index=X_test.columns, name="mean_abs_shap_lgb")
imp_lr = pd.Series(mean_abs_shap_lr, index=X_test.columns, name="mean_abs_shap_lr")

shap_comp = pd.concat([imp_lgb, imp_lr], axis=1)
shap_comp_sorted = shap_comp.sort_values("mean_abs_shap_lgb", ascending=False)
save_csv(
    shap_comp_sorted.reset_index().rename(columns={"index": "feature"}),
    "shap_importance_lgb_vs_lr", "test"
)

print("Top 15 SHAP features by LGBM:")
display(shap_comp_sorted.head(15))

rho, p_val = spearmanr(imp_lgb, imp_lr)
print(f"Spearman rank correlation LGBM vs LR SHAP: rho={rho:.3f}, p={p_val:.3e}")
shap_corr_df = pd.DataFrame([{"rho": rho, "p_value": p_val}])
save_csv(shap_corr_df, "shap_lgb_vs_lr_spearman", "test")
save_ascii_table(shap_corr_df, "shap_lgb_vs_lr_spearman", "test")

plt.figure()
plt.scatter(imp_lgb, imp_lr)
for feat in shap_comp_sorted.head(10).index:
    plt.text(imp_lgb[feat], imp_lr[feat], feat)
plt.xlabel("mean |SHAP| – LGBM")
plt.ylabel("mean |SHAP| – Logistic Regression")
plt.title("SHAP global importance comparison: LGBM vs Logistic")
save_fig("shap_importance_scatter_lgb_vs_lr", "shap")

tree_surrogate = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
y_pred_lgb_labels = (y_proba_lgb >= 0.5).astype(int)
tree_surrogate.fit(X_test, y_pred_lgb_labels)
rules_text = export_text(tree_surrogate, feature_names=list(X_test.columns))
with open(RESULTS_DIR / "surrogate_tree_lgbm_rules.txt", "w") as f:
    f.write(rules_text)
print("Surrogate tree rules saved to results/surrogate_tree_lgbm_rules.txt")


# In[ ]:


def simple_counterfactual(model, x, feature_columns,
                          max_changes=3, step_size=0.5, target_prob=0.5):
    """
    Very naive counterfactual: for continuous risk-positive features (oldpeak, ca),
    try decreasing them; for thalch try increasing; return modified sample.
    """
    x_cf = x.copy()

    def _predict_proba_row(row_series):
        row_series = row_series.reindex(feature_columns)
        X_row = row_series.to_frame().T.astype(float)
        return model.predict_proba(X_row)[:, 1][0]

    proba = _predict_proba_row(x_cf)
    if proba < target_prob:
        return x_cf, proba

    features_order = ["oldpeak", "ca", "thalch"]
    changes = 0

    for feat in features_order:
        if feat not in x_cf.index:
            continue
        while changes < max_changes:
            if feat == "oldpeak":
                x_cf[feat] = max(0.0, float(x_cf[feat]) - step_size)
            elif feat == "ca":
                x_cf[feat] = max(0.0, float(x_cf[feat]) - 1.0)
            elif feat == "thalch":
                x_cf[feat] = float(x_cf[feat]) + 5.0

            changes += 1
            proba = _predict_proba_row(x_cf)
            if proba < target_prob:
                return x_cf, proba

    return x_cf, proba

counterfactual_examples = []
high_risk_indices = np.where(y_proba_lgb >= 0.8)[0][:5]

for idx in high_risk_indices:
    x_orig = X_test.iloc[idx]
    p_orig = y_proba_lgb[idx]
    x_cf, p_cf = simple_counterfactual(
        lgb_final,
        x_orig,
        feature_columns=X_train.columns,
        target_prob=0.5,
    )

    counterfactual_examples.append({
        "index": int(idx),
        "orig_prob": float(p_orig),
        "cf_prob": float(p_cf),
        "changes": {
            f: (float(x_orig[f]), float(x_cf[f]))
            for f in ["oldpeak", "ca", "thalch"] if f in X_test.columns
        },
    })

with open(RESULTS_DIR / "counterfactual_examples.json", "w") as f:
    json.dump(counterfactual_examples, f, indent=2)

print("Counterfactual examples saved to results/counterfactual_examples.json")


# In[ ]:


metric_cols = ["accuracy", "precision", "recall", "f1", "auc", "brier", "train_time"]
df_plot = df_holdout.copy()

for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
    plt.figure()
    plt.bar(df_plot["model"], df_plot[metric])
    plt.ylabel(metric)
    plt.title(f"Models – {metric} on hold-out test set")
    save_fig(f"models_bar_{metric}", "model_performance")

plt.figure()
plt.bar(df_plot["model"], df_plot["train_time"])
plt.ylabel("Train time (s)")
plt.title("Models – training time")
save_fig("models_bar_train_time", "model_performance")

plt.figure()
plt.bar(complexity_df["model"], complexity_df["latency_s_per_sample"] * 1e6)
plt.ylabel("Latency per sample (µs)")
plt.title("Models – inference latency per sample")
save_fig("models_bar_latency", "model_performance")

print("Pipeline completed. All metrics, tables, SHAP plots, and counterfactuals saved under 'results/'.")


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

fig_dir = Path("results/figures/nested_cv")


param_figs = [
    "lgbm_nested_param_dist_n_estimators.png",
    "lgbm_nested_param_dist_learning_rate.png",
    "lgbm_nested_param_dist_max_depth.png",
    "lgbm_nested_param_dist_num_leaves.png",
    "lgbm_nested_param_dist_subsample.png",
    "lgbm_nested_param_dist_feature_fraction.png",
]

images = [mpimg.imread(fig_dir / f) for f in param_figs]

# 2 x 3 grid
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, img, title in zip(axes.ravel(), images, param_figs):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title.replace("lgbm_nested_param_dist_", "").replace(".png", ""))

plt.tight_layout()
out_path = fig_dir / "lgbm_nested_param_dist_panel.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("Saved:", out_path)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

fig_dir = Path("results/figures/model_performance")

metric_figs = [
    "models_bar_accuracy.png",
    "models_bar_precision.png",
    "models_bar_recall.png",
    "models_bar_f1.png",
    "models_bar_auc.png",
    "models_bar_train_time.png",
    "models_bar_latency.png",
]

images = [mpimg.imread(fig_dir / f) for f in metric_figs]

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, img, title in zip(axes.ravel(), images, metric_figs):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title.replace("models_bar_", "").replace(".png", ""))


for ax in axes.ravel()[len(images):]:
    ax.axis("off")

plt.tight_layout()
out_path = fig_dir / "models_bar_metrics_panel.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("Saved:", out_path)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

fig_dir = Path("results/figures/model_performance")

thresh_figs = [
    "lgbm_f1_vs_threshold.png",
    "lgbm_youdenJ_vs_threshold.png",
    "lgbm_cost_vs_threshold.png",
]

images = [mpimg.imread(fig_dir / f) for f in thresh_figs]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, title in zip(axes, images, thresh_figs):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title.replace("lgbm_", "").replace(".png", ""))

plt.tight_layout()
out_path = fig_dir / "lgbm_threshold_panel.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("Saved:", out_path)


# In[ ]:


import pandas as pd
from pathlib import Path

METRICS_DIR = Path("results/metrics")

def summarise_cv(path, model_name, cv_type):
    df = pd.read_csv(path)
    return {
        "model": model_name,
        "cv_type": cv_type,
        "mean_auc": df["auc"].mean(),
        "std_auc": df["auc"].std(ddof=1),
        "mean_f1": df["f1"].mean(),
        "std_f1": df["f1"].std(ddof=1),
        "mean_accuracy": df["accuracy"].mean(),
        "std_accuracy": df["accuracy"].std(ddof=1),
    }

rows = []


rows.append(
    summarise_cv(
        METRICS_DIR / "nested_cv" / "LogisticRegression_nestedCV.csv",
        model_name="Logistic Regression",
        cv_type="10x10 nested"
    )
)


rows.append(
    summarise_cv(
        METRICS_DIR / "nested_cv" / "LightGBM_nestedCV.csv",
        model_name="LightGBM",
        cv_type="10x10 nested"
    )
)


rows.append(
    summarise_cv(
        METRICS_DIR / "cv" / "XGBoost_10fold.csv",
        model_name="XGBoost",
        cv_type="10-fold CV"
    )
)


rows.append(
    summarise_cv(
        METRICS_DIR / "cv" / "CatBoost_10fold.csv",
        model_name="CatBoost",
        cv_type="10-fold CV"
    )
)

df_cv_summary = pd.DataFrame(rows)


df_cv_summary_rounded = df_cv_summary.copy()
for col in ["mean_auc", "std_auc", "mean_f1", "std_f1", "mean_accuracy", "std_accuracy"]:
    df_cv_summary_rounded[col] = df_cv_summary_rounded[col].round(3)

display(df_cv_summary_rounded)


out_name = "cv_models_summary"
df_cv_summary_rounded.to_csv(METRICS_DIR / f"{out_name}.csv", index=False)


from tabulate import tabulate
ascii_txt = tabulate(df_cv_summary_rounded, headers="keys", tablefmt="psql", floatfmt=".3f")
ASCII_DIR = Path("results/tables_ascii")
ASCII_DIR.mkdir(parents=True, exist_ok=True)
with open(ASCII_DIR / f"{out_name}.txt", "w") as f:
    f.write(ascii_txt + "\n")


# In[ ]:


if TabNetClassifier is not None and torch is not None:
    def tabnet_cv_evaluate():
        X_np = X.to_numpy(dtype=np.float32)
        y_np = y.to_numpy(dtype=np.int64)

        skf = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=RANDOM_STATE
        )
        rows = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_np, y_np), 1):
            X_tr, X_val = X_np[tr_idx], X_np[val_idx]
            y_tr, y_val = y_np[tr_idx], y_np[val_idx]

            m = TabNetClassifier(seed=RANDOM_STATE, verbose=0)
            t0 = time.time()
            m.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  eval_metric=["auc"],
                  max_epochs=200,
                  patience=30,
                  batch_size=128,
                  virtual_batch_size=64)
            train_time = time.time() - t0

            y_proba = m.predict_proba(X_val)[:, 1]
            test_time = 0.0  
            metrics = compute_metrics(y_val, y_proba)
            metrics.update({"fold": fold,
                            "train_time": train_time,
                            "test_time": test_time})
            rows.append(metrics)

        df_cv = pd.DataFrame(rows)
        save_csv(df_cv, "TabNet_10fold", "cv")
        save_ascii_table(df_cv.describe().T, "TabNet_10fold_summary", "cv")
        return df_cv

    tabnet_cv_results = tabnet_cv_evaluate()



# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

fig_dir = Path("results/figures/model_performance")

cm_figs = [
    "LightGBM_confusion_counts.png",
    "LightGBM_confusion_norm.png",
    "LogisticRegression_confusion_counts.png",
    "LogisticRegression_confusion_norm.png",
    "XGBoost_confusion_counts.png",
    "XGBoost_confusion_norm.png",
    "TabNet_confusion_counts.png",
    "TabNet_confusion_norm.png",
]

images = [mpimg.imread(fig_dir / f) for f in cm_figs]

fig, axes = plt.subplots(4, 2, figsize=(10, 16))

titles = [
    "LGBM – counts", "LGBM – normalised",
    "LogReg – counts", "LogReg – normalised",
    "XGB – counts", "XGB – normalised",
    "TabNet – counts", "TabNet – normalised",
]

for ax, img, title in zip(axes.ravel(), images, titles):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)

plt.tight_layout()
out_path = fig_dir / "models_confusion_panel.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("Saved:", out_path)



# In[ ]:


def plot_training_curves_xgb():
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric=["logloss", "auc", "error"],
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    evals_result = model.evals_result()

    keys = list(evals_result.keys())
    train_key, valid_key = keys[0], keys[1]

    train_logloss = np.array(evals_result[train_key]["logloss"])
    valid_logloss = np.array(evals_result[valid_key]["logloss"])
    train_auc = np.array(evals_result[train_key]["auc"])
    valid_auc = np.array(evals_result[valid_key]["auc"])
    train_error = np.array(evals_result[train_key]["error"])
    valid_error = np.array(evals_result[valid_key]["error"])

   
    train_accuracy = 1.0 - train_error
    valid_accuracy = 1.0 - valid_error

    df_curves = pd.DataFrame({
        "iter": np.arange(len(train_logloss)),
        "train_logloss": train_logloss,
        "valid_logloss": valid_logloss,
        "train_auc": train_auc,
        "valid_auc": valid_auc,
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
    })
    save_csv(df_curves, "xgb_training_curves", "model_performance")

    
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_logloss"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_logloss"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("Logloss")
    plt.title("XGBoost – training vs validation logloss")
    plt.legend()
    save_fig("xgb_train_valid_logloss", "model_performance")

    
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_auc"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_auc"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title("XGBoost – training vs validation AUC")
    plt.legend()
    save_fig("xgb_train_valid_auc", "model_performance")


    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_accuracy"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_accuracy"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("XGBoost – training vs validation accuracy")
    plt.legend()
    save_fig("xgb_train_valid_accuracy", "model_performance")

def plot_training_curves_lgbm():
    model = LGBMClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        feature_fraction=0.8,
        min_child_samples=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric=["binary_logloss", "binary_error", "auc"],
            verbose=False,
        )
    except TypeError:
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric=["binary_logloss", "binary_error", "auc"],
        )

    evals_result = model.evals_result_
    
    train_key = "training"
    valid_key = [k for k in evals_result.keys() if k != train_key][0]

    train_logloss = np.array(evals_result[train_key]["binary_logloss"])
    valid_logloss = np.array(evals_result[valid_key]["binary_logloss"])
    train_auc = np.array(evals_result[train_key]["auc"])
    valid_auc = np.array(evals_result[valid_key]["auc"])
    train_error = np.array(evals_result[train_key]["binary_error"])
    valid_error = np.array(evals_result[valid_key]["binary_error"])

    train_accuracy = 1.0 - train_error
    valid_accuracy = 1.0 - valid_error

    df_curves = pd.DataFrame({
        "iter": np.arange(len(train_logloss)),
        "train_logloss": train_logloss,
        "valid_logloss": valid_logloss,
        "train_auc": train_auc,
        "valid_auc": valid_auc,
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
    })
    save_csv(df_curves, "lgbm_training_curves", "model_performance")

   
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_logloss"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_logloss"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("Binary logloss")
    plt.title("LightGBM – training vs validation logloss")
    plt.legend()
    save_fig("lgbm_train_valid_logloss", "model_performance")


    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_auc"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_auc"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title("LightGBM – training vs validation AUC")
    plt.legend()
    save_fig("lgbm_train_valid_auc", "model_performance")

    
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_accuracy"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_accuracy"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("LightGBM – training vs validation accuracy")
    plt.legend()
    save_fig("lgbm_train_valid_accuracy", "model_performance")



# In[ ]:


single_node_row_main = {
    "model": "LightGBM_single_node",
    "accuracy": metrics_lgb["accuracy"],
    "precision": metrics_lgb["precision"],
    "recall": metrics_lgb["recall"],
    "f1": metrics_lgb["f1"],
    "auc": metrics_lgb["auc"],
    "brier": metrics_lgb["brier"],
    "train_time": metrics_lgb["train_time"],
}

df_lgb_single_main = pd.DataFrame([single_node_row_main])

print("Single-node LightGBM summary from main notebook:")
display(df_lgb_single_main)


save_csv(df_lgb_single_main, "lightgbm_single_node_summary", subfolder="test")
save_ascii_table(df_lgb_single_main,
                 "lightgbm_single_node_summary",
                 subfolder="test")


# In[ ]:


def repeated_cv_evaluate(model, X, y, model_name="model",
                         n_splits=10, n_repeats=10):
    """
    Repeated stratified K-fold evaluation.
    - n_splits: folds per repeat (10 for '10-fold')
    - n_repeats: how many times to re-draw the folds (10 for '10 x 10-fold')
    Saves all folds to results/metrics/cv and returns the full DataFrame.
    """
    rows = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE + rep,
        )
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            m = clone(model)
            t0 = time.time()
            m.fit(X_tr, y_tr)
            train_time = time.time() - t0

            t0 = time.time()
            y_proba = m.predict_proba(X_val)[:, 1]
            test_time = time.time() - t0

            metrics = compute_metrics(y_val, y_proba)
            metrics.update({
                "repeat": rep + 1,
                "fold": fold,
                "train_time": train_time,
                "test_time": test_time,
            })
            rows.append(metrics)

            print(
                f"{model_name} rep {rep+1:02d}, fold {fold:02d}: "
                f"AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}"
            )

    df_rep = pd.DataFrame(rows)
    fname = f"{model_name}_10x10fold"
    save_csv(df_rep, fname, "cv")
    save_ascii_table(df_rep.describe().T, f"{fname}_summary", "cv")
    return df_rep



xgb_10x10 = repeated_cv_evaluate(xgb_base, X, y, "XGBoost")
lgb_10x10 = repeated_cv_evaluate(lgb_base, X, y, "LightGBM")
lr_10x10  = repeated_cv_evaluate(lr_pipe, X, y, "LogisticRegression")


# In[ ]:


def plot_training_curves_xgb():
    """
    XGBoost: track training/validation logloss, AUC and accuracy vs iteration.
    """
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
    
        eval_metric=["logloss", "auc", "error"],
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    evals_result = model.evals_result()
    train_key, valid_key = list(evals_result.keys())

    train_res = evals_result[train_key]
    valid_res = evals_result[valid_key]

    iters = np.arange(len(train_res["logloss"]))

    
    train_acc = 1.0 - np.array(train_res["error"])
    valid_acc = 1.0 - np.array(valid_res["error"])

    df_curves = pd.DataFrame({
        "iter": iters,
        "train_logloss": train_res["logloss"],
        "valid_logloss": valid_res["logloss"],
        "train_auc": train_res["auc"],
        "valid_auc": valid_res["auc"],
        "train_acc": train_acc,
        "valid_acc": valid_acc,
    })
    save_csv(df_curves, "xgb_training_curves", "model_performance")

    
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_logloss"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_logloss"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("Logloss")
    plt.title("XGBoost – training vs validation logloss")
    plt.legend()
    save_fig("xgb_train_valid_logloss", "model_performance")

    
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_auc"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_auc"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title("XGBoost – training vs validation AUC")
    plt.legend()
    save_fig("xgb_train_valid_auc", "model_performance")

    
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_acc"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_acc"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("XGBoost – training vs validation accuracy")
    plt.legend()
    save_fig("xgb_train_valid_accuracy", "model_performance")


def plot_training_curves_lgbm():
    """
    LightGBM: track training/validation binary_logloss, AUC and accuracy vs iteration.
    """
    model = LGBMClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        feature_fraction=0.8,
        min_child_samples=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric=["binary_logloss", "auc", "binary_error"],
            verbose=False,
        )
    except TypeError:
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric=["binary_logloss", "auc", "binary_error"],
        )

    evals_result = model.evals_result_
    train_key = "training"
    valid_key = [k for k in evals_result.keys() if k != train_key][0]

    train_res = evals_result[train_key]
    valid_res = evals_result[valid_key]

    iters = np.arange(len(train_res["binary_logloss"]))

    train_acc = 1.0 - np.array(train_res["binary_error"])
    valid_acc = 1.0 - np.array(valid_res["binary_error"])

    df_curves = pd.DataFrame({
        "iter": iters,
        "train_logloss": train_res["binary_logloss"],
        "valid_logloss": valid_res["binary_logloss"],
        "train_auc": train_res["auc"],
        "valid_auc": valid_res["auc"],
        "train_acc": train_acc,
        "valid_acc": valid_acc,
    })
    save_csv(df_curves, "lgbm_training_curves", "model_performance")

    
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_logloss"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_logloss"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("Binary logloss")
    plt.title("LightGBM – training vs validation logloss")
    plt.legend()
    save_fig("lgbm_train_valid_logloss", "model_performance")

    # AUC
    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_auc"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_auc"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title("LightGBM – training vs validation AUC")
    plt.legend()
    save_fig("lgbm_train_valid_auc", "model_performance")


    plt.figure()
    plt.plot(df_curves["iter"], df_curves["train_acc"], label="train")
    plt.plot(df_curves["iter"], df_curves["valid_acc"], label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("LightGBM – training vs validation accuracy")
    plt.legend()
    save_fig("lgbm_train_valid_accuracy", "model_performance")



plot_training_curves_xgb()
plot_training_curves_lgbm()


# In[ ]:


from pathlib import Path
import joblib

DEPLOY_DIR = Path("deployment_bundle")
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)


lgbm_pipeline_final = lgb_final   

feature_names = list(X_train.columns)

bundle = {
    
    "pipeline": lgbm_pipeline_final,
    "feature_names": feature_names,
}

out_path = DEPLOY_DIR / "lgbm_deployment_bundle.joblib"
joblib.dump(bundle, out_path)
print("Saved deployment bundle to:", out_path)
print("Number of features:", len(feature_names))



# In[ ]:


get_ipython().system('pip install psutil')


# In[ ]:


import os
import psutil


p = psutil.Process(os.getpid())
p.cpu_affinity([0]) 
p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt  


bundle = joblib.load(Path("deployment_bundle") / "lgbm_deployment_bundle.joblib")
lgbm_pipeline_final = bundle["pipeline"]
lgbm_pipeline_final.set_params(n_jobs=1)
feature_names = bundle["feature_names"]


assert list(X_test.columns) == feature_names, "X_test columns differ from saved feature_names!"

SLOWDOWN_FACTOR = 3.0  
latencies = []
n_repeats = 5

for _ in range(n_repeats):
    for i in range(len(X_test)):
        x_i = X_test.iloc[i:i+1]

        t0 = time.perf_counter()
        _ = lgbm_pipeline_final.predict_proba(x_i)[:, 1]
        t1 = time.perf_counter()

        
        inference_time = t1 - t0

        
        extra = inference_time * (SLOWDOWN_FACTOR - 1.0)
        end = time.perf_counter() + extra
        while time.perf_counter() < end:
            pass

        t2 = time.perf_counter()
        latencies.append(t2 - t0)


latencies = np.array(latencies)

print(f"Number of simulated requests: {len(latencies)}")
print(f"Median latency:  {np.median(latencies)*1e3:.2f} ms")
print(f"95th percentile: {np.percentile(latencies, 95)*1e3:.2f} ms")
print(f"Max latency:     {np.max(latencies)*1e3:.2f} ms")

plt.figure(figsize=(6, 4))
plt.hist(latencies * 1e3, bins=30)
plt.xlabel("Per-patient latency (ms)")
plt.ylabel("Count")
plt.title("LightGBM – simulated per-patient latency")
plt.grid(alpha=0.3)
plt.tight_layout()
save_fig("lgbm_deployment_latency_hist", subfolder="model_performance")
plt.close()


# In[ ]:


import pandas as pd


latencies_ms = latencies * 1e3

n_requests = len(latencies_ms)
n_patients = len(X_test)    
df_lat = pd.DataFrame({
    "request_idx": np.arange(n_requests),
    "latency_ms": latencies_ms,
})


df_lat["repeat"] = df_lat["request_idx"] // n_patients
df_lat["patient_idx"] = df_lat["request_idx"] % n_patients


df_lat["phase"] = np.where(df_lat["repeat"] == 0, "warmup", "steady")


df_lat["stage"] = pd.cut(
    df_lat["request_idx"],
    bins=[-1, n_requests // 3, 2 * n_requests // 3, n_requests],
    labels=["early", "mid", "late"]
)

df_lat.head()


# In[ ]:


mean_ms = df_lat["latency_ms"].mean()
median_ms = df_lat["latency_ms"].median()
std_ms = df_lat["latency_ms"].std()
p90_ms = np.percentile(df_lat["latency_ms"], 90)
p95_ms = np.percentile(df_lat["latency_ms"], 95)
p99_ms = np.percentile(df_lat["latency_ms"], 99)
max_ms = df_lat["latency_ms"].max()
cv = std_ms / mean_ms  
throughput_rps = 1000.0 / median_ms  

print("=== Global latency metrics ===")
print(f"Requests           : {n_requests}")
print(f"Mean latency       : {mean_ms:.3f} ms")
print(f"Median latency     : {median_ms:.3f} ms")
print(f"Std dev            : {std_ms:.3f} ms")
print(f"90th percentile    : {p90_ms:.3f} ms")
print(f"95th percentile    : {p95_ms:.3f} ms")
print(f"99th percentile    : {p99_ms:.3f} ms")
print(f"Max latency        : {max_ms:.3f} ms")
print(f"Coeff. variation   : {cv:.3f}")
print(f"Throughput (approx): {throughput_rps:.1f} req/s")


print("\n=== Warmup vs steady-state (phase) ===")
phase_stats = df_lat.groupby("phase")["latency_ms"].agg(
    mean_ms="mean",
    median_ms="median",
    std_ms="std",
    p95_ms=lambda s: np.percentile(s, 95),
    p99_ms=lambda s: np.percentile(s, 99),
    max_ms="max",
)
print(phase_stats)


print("\n=== Early / mid / late (stage) ===")
stage_stats = df_lat.groupby("stage")["latency_ms"].agg(
    mean_ms="mean",
    median_ms="median",
    std_ms="std",
    p95_ms=lambda s: np.percentile(s, 95),
    p99_ms=lambda s: np.percentile(s, 99),
    max_ms="max",
)
print(stage_stats)


df_lat["delta_ms"] = df_lat["latency_ms"].diff().abs()
jitter_mean = df_lat["delta_ms"].mean()
jitter_p95 = np.percentile(df_lat["delta_ms"].dropna(), 95)
print("\n=== Jitter (|latency_i - latency_{i-1}|) ===")
print(f"Mean jitter        : {jitter_mean:.3f} ms")
print(f"95th percentile    : {jitter_p95:.3f} ms")


# In[ ]:


plt.figure(figsize=(8, 4))
plt.plot(df_lat["request_idx"], df_lat["latency_ms"], marker=".", linestyle="-", alpha=0.6)
plt.xlabel("Request index (time)")
plt.ylabel("Latency (ms)")
plt.title("Per-request latency over the run")
plt.grid(alpha=0.3)
plt.tight_layout()
save_fig("lgbm_latency_over_time", subfolder="model_performance")
plt.close()


# In[ ]:


plt.figure(figsize=(5, 4))
df_lat.boxplot(column="latency_ms", by="phase")
plt.ylabel("Latency (ms)")
plt.title("Latency distribution: warmup vs steady")
plt.suptitle("")  # remove automatic extra title
plt.grid(alpha=0.3)
plt.tight_layout()
save_fig("lgbm_latency_warmup_vs_steady", subfolder="model_performance")
plt.close()


# In[ ]:


plt.figure(figsize=(5, 4))
df_lat.boxplot(column="latency_ms", by="stage")
plt.ylabel("Latency (ms)")
plt.title("Latency distribution by stage of run")
plt.suptitle("")
plt.grid(alpha=0.3)
plt.tight_layout()
save_fig("lgbm_latency_by_stage", subfolder="model_performance")
plt.close()


# In[ ]:


import joblib
from pathlib import Path
import pprint

bundle = joblib.load(Path("deployment_bundle") / "lgbm_deployment_bundle.joblib")
pprint.pp(bundle["feature_names"])

