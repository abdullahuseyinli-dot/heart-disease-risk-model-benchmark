#!/usr/bin/env python3
"""Validate tracked benchmark evidence using only the Python standard library."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "README.md",
    "data/heart_disease_processed.parquet",
    "results/main/metrics/test/holdout_models.csv",
    "results/main/metrics/test/holdout_bootstrap_ci.csv",
    "results/main/metrics/test/shap_lgb_vs_lr_spearman.csv",
    "results/dask/metrics/lightgbm_distributed_summary_colab.csv",
    "results/edge/metrics/latency_summary.json",
)

EXPECTED_HOLDOUT = {
    "LightGBM": (0.842391304347826, 0.863849765258216, 0.8962219033955046, 0.11916372256997074),
    "LogisticRegression": (0.8369565217391305, 0.8584905660377359, 0.9082974653275945, 0.11828327559659454),
    "XGBoost": (0.8586956521739131, 0.8773584905660378, 0.9010043041606887, 0.11509418536924956),
    "TabNet": (0.8532608695652174, 0.8744186046511628, 0.9301769488283118, 0.10658166429997569),
}

README_VALUES = (
    "0.8370",
    "0.8585",
    "0.9083",
    "0.1183",
    "0.8424",
    "0.8638",
    "0.8962",
    "0.1192",
    "0.8587",
    "0.8774",
    "0.9010",
    "0.1151",
    "0.8533",
    "0.8744",
    "0.9302",
    "0.1066",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"validation failed: {message}")


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    require(not missing, f"missing required files: {', '.join(missing)}")

    parquet = ROOT / "data/heart_disease_processed.parquet"
    with parquet.open("rb") as handle:
        require(handle.read(4) == b"PAR1", "processed dataset has no Parquet header")
        handle.seek(-4, 2)
        require(handle.read(4) == b"PAR1", "processed dataset has no Parquet footer")


def validate_holdout_results() -> None:
    rows = {row["model"]: row for row in read_csv("results/main/metrics/test/holdout_models.csv")}
    require(set(rows) == set(EXPECTED_HOLDOUT), "unexpected holdout model set")

    for model, expected in EXPECTED_HOLDOUT.items():
        actual = tuple(float(rows[model][name]) for name in ("accuracy", "f1", "auc", "brier"))
        require(
            all(math.isclose(a, e, rel_tol=0.0, abs_tol=1e-12) for a, e in zip(actual, expected)),
            f"holdout metrics changed for {model}",
        )

    intervals = read_csv("results/main/metrics/test/holdout_bootstrap_ci.csv")
    expected_interval_models = {"LightGBM", "LogReg", "XGBoost", "TabNet"}
    require(
        {row["model"] for row in intervals} == expected_interval_models,
        "bootstrap model set changed",
    )
    for row in intervals:
        for metric in ("auc", "f1"):
            low = float(row[f"{metric}_ci_low"])
            mean = float(row[f"{metric}_mean"])
            high = float(row[f"{metric}_ci_high"])
            require(0.0 <= low <= mean <= high <= 1.0, f"invalid {metric} interval for {row['model']}")


def validate_system_results() -> None:
    edge = json.loads((ROOT / "results/edge/metrics/latency_summary.json").read_text(encoding="utf-8"))
    expected_edge = {
        "n_requests": 2760,
        "mean_ms": 14.610648687317054,
        "p95_ms": 33.25178699992646,
        "throughput_rps": 91.05737784979893,
    }
    for key, expected in expected_edge.items():
        require(math.isclose(float(edge[key]), expected, rel_tol=0.0, abs_tol=1e-12), f"edge metric changed: {key}")

    dask = {row["model"]: row for row in read_csv("results/dask/metrics/lightgbm_distributed_summary_colab.csv")}
    require(math.isclose(float(dask["LightGBM_single_node"]["auc"]), 0.8962219033955046), "single-node AUC changed")
    require(math.isclose(float(dask["LightGBM_DaskClassifier"]["auc"]), 0.5329985652797704), "Dask AUC changed")

    shap = read_csv("results/main/metrics/test/shap_lgb_vs_lr_spearman.csv")
    require(len(shap) == 1, "expected one SHAP agreement row")
    require(math.isclose(float(shap[0]["rho"]), 0.7711799815081232), "SHAP rank agreement changed")


def validate_readme() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    require("not a medical device" in readme, "medical-use limitation is missing")
    for value in README_VALUES:
        require(value in readme, f"README is missing holdout value {value}")

    for target in re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", readme):
        target = target.strip().split(maxsplit=1)[0].strip("<>")
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        local = unquote(target.split("#", 1)[0])
        require((ROOT / local).exists(), f"broken README link: {target}")


def validate_public_paths() -> None:
    marker = "c:" + chr(92) + "users"
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in {".md", ".py", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        require(marker not in text, f"local user path in {path.relative_to(ROOT)}")


def main() -> None:
    validate_required_files()
    validate_holdout_results()
    validate_system_results()
    validate_readme()
    validate_public_paths()
    print("Repository validation passed.")


if __name__ == "__main__":
    main()
