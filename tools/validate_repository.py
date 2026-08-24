#!/usr/bin/env python3
"""Validate tracked benchmark evidence using only the Python standard library."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "LICENSE",
    "README.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    ".zenodo.json",
    "docs/ARTIFACTS.md",
    "docs/ARCHITECTURE.md",
    "docs/BENCHMARK_CARD.md",
    "docs/DATA_ACQUISITION_RUNBOOK.md",
    "docs/HARDWARE.md",
    "docs/PROJECT_STATUS.md",
    "docs/RELEASE_EVIDENCE_GATE.md",
    "docs/USAGE.md",
    "docs/VERSIONING.md",
    "docs/legacy/LEGACY_BENCHMARK.md",
    "configs/research/method_registry_v2.yaml",
    "configs/research/method_registry_v3.yaml",
    "configs/release/release_gate_policy_v1.json",
    "configs/schema/release_gate_policy.schema.json",
    "manifests/datasets/uci_heart_v1.json",
    "manifests/datasets/uci_diabetes_readmission_v1.json",
    "manifests/datasets/eicu_crd_demo_v2.0.1.json",
    "manifests/environment/confirmatory_machine_20260824.json",
    "manifests/evidence/heart_prediction_table_v1.json",
    "manifests/evidence/heart_primary_estimands_v1.json",
    "manifests/evidence/heart_outer_v5_report_v1.json",
    "paper/OUTLINE.md",
    "paper/CLAIM_EVIDENCE_CROSSWALK.md",
    "paper/references.bib",
    "data/splits/uci_heart_loho_v2.json",
    "data/raw/eicu-crd-demo/2.0.1/LICENSE-ODbL-1.0.md",
    "data/processed/eicu-demo-shiftguard-smoke-v1/LICENSE-ODbL-1.0.md",
    "data/heart_disease_processed.parquet",
    "results/main/metrics/test/holdout_models.csv",
    "results/main/metrics/test/holdout_bootstrap_ci.csv",
    "results/main/metrics/test/shap_lgb_vs_lr_spearman.csv",
    "results/dask/metrics/lightgbm_distributed_summary_colab.csv",
    "results/edge/metrics/latency_summary.json",
)

EXPECTED_HOLDOUT = {
    "LightGBM": (0.842391304347826, 0.863849765258216, 0.8962219033955046, 0.11916372256997074),
    "LogisticRegression": (
        0.8369565217391305,
        0.8584905660377359,
        0.9082974653275945,
        0.11828327559659454,
    ),
    "XGBoost": (0.8586956521739131, 0.8773584905660378, 0.9010043041606887, 0.11509418536924956),
    "TabNet": (0.8532608695652174, 0.8744186046511628, 0.9301769488283118, 0.10658166429997569),
}

LEGACY_VALUES = (
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

CURRENT_README_VALUES = (
    "0.602509",
    "0.625949",
    "0.600458",
    "0.625463",
    "0.672257",
    "0.719939",
    "0.619081",
    "0.615985",
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
            all(
                math.isclose(a, e, rel_tol=0.0, abs_tol=1e-12)
                for a, e in zip(actual, expected, strict=True)
            ),
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
            require(
                0.0 <= low <= mean <= high <= 1.0, f"invalid {metric} interval for {row['model']}"
            )


def validate_system_results() -> None:
    edge = json.loads(
        (ROOT / "results/edge/metrics/latency_summary.json").read_text(encoding="utf-8")
    )
    expected_edge = {
        "n_requests": 2760,
        "mean_ms": 14.610648687317054,
        "p95_ms": 33.25178699992646,
        "throughput_rps": 91.05737784979893,
    }
    for key, expected in expected_edge.items():
        require(
            math.isclose(float(edge[key]), expected, rel_tol=0.0, abs_tol=1e-12),
            f"edge metric changed: {key}",
        )

    dask = {
        row["model"]: row
        for row in read_csv("results/dask/metrics/lightgbm_distributed_summary_colab.csv")
    }
    require(
        math.isclose(float(dask["LightGBM_single_node"]["auc"]), 0.8962219033955046),
        "single-node AUC changed",
    )
    require(
        math.isclose(float(dask["LightGBM_DaskClassifier"]["auc"]), 0.5329985652797704),
        "Dask AUC changed",
    )

    shap = read_csv("results/main/metrics/test/shap_lgb_vs_lr_spearman.csv")
    require(len(shap) == 1, "expected one SHAP agreement row")
    require(math.isclose(float(shap[0]["rho"]), 0.7711799815081232), "SHAP rank agreement changed")


def validate_document_links(relative_path: str) -> None:
    document = ROOT / relative_path
    text = document.read_text(encoding="utf-8")
    for target in re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", text):
        target = target.strip().split(maxsplit=1)[0].strip("<>")
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        local = unquote(target.split("#", 1)[0])
        require(
            (document.parent / local).resolve().exists(),
            f"broken link in {relative_path}: {target}",
        )


def validate_public_documentation() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    require("not a medical device" in readme, "medical-use limitation is missing")
    require("Python 3.10" not in readme, "README advertises the archived Python environment")
    for value in CURRENT_README_VALUES:
        require(value in readme, f"README is missing current result value {value}")

    legacy = (ROOT / "docs/legacy/LEGACY_BENCHMARK.md").read_text(encoding="utf-8")
    for value in LEGACY_VALUES:
        require(value in legacy, f"legacy benchmark record is missing holdout value {value}")

    legacy_bundle = (ROOT / "deployment_bundle/README.md").read_text(encoding="utf-8")
    legacy_service = (ROOT / "scripts/edge_inference_service.py").read_text(encoding="utf-8")
    require("archived v1 systems experiment" in legacy_bundle, "legacy bundle is not archived")
    require(
        "not a deployable" in legacy_bundle and "medical model" in legacy_bundle,
        "legacy bundle lacks use boundary",
    )
    require("Archived, non-clinical" in legacy_service, "legacy service lacks use boundary")

    heart_result = (ROOT / "docs/HEART_OUTER_V5_RESULT.md").read_text(encoding="utf-8")
    require("deployable report" not in heart_result, "heart result uses deployment language")
    require("professional hospital" not in heart_result, "heart result uses marketing language")

    for relative_path in (
        "README.md",
        "docs/README.md",
        "docs/BENCHMARK_CARD.md",
        "docs/ARCHITECTURE.md",
        "docs/DATA_ACQUISITION_RUNBOOK.md",
        "docs/HARDWARE.md",
        "docs/RELEASE_EVIDENCE_GATE.md",
        "docs/USAGE.md",
        "docs/VERSIONING.md",
        "docs/legacy/LEGACY_BENCHMARK.md",
        "paper/README.md",
        "paper/OUTLINE.md",
        "paper/CLAIM_EVIDENCE_CROSSWALK.md",
    ):
        validate_document_links(relative_path)


def validate_licensing() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    for marker in (
        "MIT License",
        "Copyright (c) 2026 Abdulla Huseyinli",
        "Permission is hereby granted, free of charge",
        'THE SOFTWARE IS PROVIDED "AS IS"',
    ):
        require(marker in license_text, f"LICENSE is missing: {marker}")

    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for marker in (
        "CC BY 4.0",
        "10.24432/C52P4X",
        "heart+disease",
        "Open Database License",
        "10.13026/4mxk-na84",
        "Prior Labs License 1.2",
    ):
        require(marker in notices, f"dataset notice is missing: {marker}")

    for relative_path in (
        "data/raw/eicu-crd-demo/2.0.1/LICENSE-ODbL-1.0.md",
        "data/processed/eicu-demo-shiftguard-smoke-v1/LICENSE-ODbL-1.0.md",
    ):
        scoped_notice = (ROOT / relative_path).read_text(encoding="utf-8")
        for marker in ("Open Database License", "10.13026/4mxk-na84", "MIT License"):
            require(marker in scoped_notice, f"{relative_path} is missing: {marker}")


def validate_public_paths() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    allowed_historical_prefixes = (
        "artifacts/failures/",
        "artifacts/locks/",
        "artifacts/runs/",
    )
    allowed_historical_files = {
        "data/splits/uci_heart_loho_v1.json",
    }
    text_suffixes = {".cff", ".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
    marker = re.compile(r"[a-z]:[\\/]+users[\\/]", flags=re.IGNORECASE)
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8")
        path = ROOT / relative
        normalized = relative.replace("\\", "/")
        if (
            not path.is_file()
            or normalized.startswith(allowed_historical_prefixes)
            or normalized in allowed_historical_files
            or path.suffix.lower() not in text_suffixes
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        require(not marker.search(text), f"local user path in {relative}")


def validate_release_metadata() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    release_policy = json.loads(
        (ROOT / "configs/release/release_gate_policy_v1.json").read_text(encoding="utf-8")
    )
    package = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    version_match = re.search(r'^version = "([^"]+)"$', package, flags=re.MULTILINE)
    require(version_match is not None, "pyproject version is missing")
    version = version_match.group(1)
    require(f"version: {version}" in citation, "CITATION.cff version does not match package")
    require(zenodo.get("version") == version, ".zenodo.json version does not match package")
    require("doi" not in zenodo, ".zenodo.json must not invent an unpublished DOI")
    require(
        release_policy.get("schema_version") == "1.0.0",
        "release gate policy schema version is unsupported",
    )


def main() -> None:
    validate_required_files()
    validate_holdout_results()
    validate_system_results()
    validate_public_documentation()
    validate_licensing()
    validate_public_paths()
    validate_release_metadata()
    print("Repository validation passed.")


if __name__ == "__main__":
    main()
