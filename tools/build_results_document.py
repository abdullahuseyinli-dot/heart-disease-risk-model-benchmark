"""Build the public result tables from the retained experiment reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HEART = "artifacts/reports/heart-outer-v5"
READMISSION = "artifacts/reports/readmission-outer-v3"
DEVELOPMENT = "artifacts/reports/heart-research-development-v1"
STABILITY = "artifacts/reports/historical-psmask-ten-seed-sensitivity-v2-stability"
REVISIONS = "artifacts/reports/shiftguard-revisions-v3-final-development"
ADAPTATION = "artifacts/figures/heartshift-v5-r2/heart_rejected_adaptation.csv"
SYNTHETIC = "artifacts/runs/synthetic-mechanism-v3/acceptance_gate.json"
ROUTER = "artifacts/runs/support-router-outer-development-v1/router_gates.json"
DOCUMENT = "docs/RESULTS.md"
NATURAL_EXPORT = "docs/research/heart_natural_metrics.csv"
MANIFEST = "docs/research/result_presentation_manifest.json"
PRIMARY = "macro_site_worst_mask_balanced_log_loss"
WORST = "worst_site_mask_balanced_log_loss"
AUC = "macro_natural_roc_auc"
NATURAL_SITES = {"cleveland": 303, "hungary": 294, "switzerland": 123, "va_long_beach": 200}
NATURAL_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "brier")
SELECTED_METHODS = {
    "psmask:v2_prior_separated": "V2 · prior separation",
    "psmask:v0_pooled_erm": "V0 · pooled ERM",
    "psmask:v5_mask_only_dro": "V5 · mask-axis DRO (preselected)",
    "classical:random_forest:site_class_balanced": "Random forest",
    "classical:logistic:site_class_balanced": "Logistic regression",
}
TABLES = {
    f"{HEART}/primary_estimands.csv": (45, ("method",)),
    f"{HEART}/site_policy_metrics.csv": (
        9720,
        ("method", "track", "outer_target", "policy", "mask_replicate"),
    ),
    f"{HEART}/paired_bootstrap_intervals.csv": (16, ("method", "reference_method")),
    f"{READMISSION}/primary_estimands.csv": (9, ("experiment",)),
    f"{READMISSION}/patient_cluster_bootstrap_intervals.csv": (5, ("experiment",)),
    f"{DEVELOPMENT}/primary_estimands.csv": (16, ("method",)),
    f"{STABILITY}/primary_estimands.csv": (22, ("method",)),
    f"{STABILITY}/three_vs_ten_seed_comparison.csv": (10, ("experiment",)),
    f"{STABILITY}/posthoc_multiplicity_sensitivity.csv": (10, ("method",)),
    f"{REVISIONS}/revision_comparison.csv": (5, ("revision",)),
    ADAPTATION: (8, ("experiment", "track")),
}
JSON_SOURCES = [
    f"{HEART}/report_manifest.json",
    f"{READMISSION}/report_manifest.json",
    f"{DEVELOPMENT}/report_manifest.json",
    f"{STABILITY}/report_manifest.json",
    SYNTHETIC,
    ROUTER,
]


def read_tables(root: Path) -> dict[str, list[dict[str, str]]]:
    tables = {}
    for path, (count, keys) in TABLES.items():
        text = (root / path).read_text(encoding="utf-8-sig")
        if text.startswith("version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Materialize the report before presenting results: {path}")
        rows = list(csv.DictReader(io.StringIO(text)))
        if len(rows) != count or len({tuple(row[key] for key in keys) for row in rows}) != count:
            raise ValueError(f"Incomplete or duplicate result set: {path}; expected {count}")
        tables[path] = rows
    for directory, key, manifest_key in (
        (HEART, "method", "methods"),
        (READMISSION, "experiment", "experiments"),
    ):
        manifest = json.loads((root / directory / "report_manifest.json").read_text())
        if {row[key] for row in tables[f"{directory}/primary_estimands.csv"]} != set(
            manifest[manifest_key]
        ):
            raise ValueError(f"Method set differs from the frozen report: {directory}")
    heart_manifest = json.loads((root / HEART / "report_manifest.json").read_text())
    if heart_manifest["tracks"] != ["dg_zero_shot"]:
        raise ValueError("Automatic adaptation cannot be pooled into the zero-shot result table")
    for directory in (DEVELOPMENT, STABILITY):
        manifest = json.loads((root / directory / "report_manifest.json").read_text())
        if (
            manifest["new_confirmatory_claim_allowed"] is not False
            or manifest["outer_outcomes_historically_consumed"] is not True
            or manifest["method_count"] != len(tables[f"{directory}/primary_estimands.csv"])
        ):
            raise ValueError(f"Development scope or method count changed: {directory}")
    for path, key, expected in (
        (
            f"{HEART}/paired_bootstrap_intervals.csv",
            "reference_method",
            "classical:logistic:site_class_balanced",
        ),
        (
            f"{READMISSION}/patient_cluster_bootstrap_intervals.csv",
            "reference_experiment",
            "logistic_environment_class_balanced",
        ),
    ):
        if {row[key] for row in tables[path]} != {expected}:
            raise ValueError(f"Registered comparison reference changed: {path}")
    return tables


def one(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    matches = [row for row in rows if row[key] == value]
    if len(matches) != 1:
        raise ValueError(f"Expected one result for {value}; found {len(matches)}")
    return matches[0]


def number(value: str | float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Non-finite metric cannot be presented as a completed result")
    return f"{numeric:.6f}"


def summarize_natural_metrics(
    site_rows: list[dict[str, str]], methods: list[str]
) -> list[dict[str, str]]:
    """Average saved natural-condition scores with equal weight for each hospital."""
    natural = [
        row for row in site_rows if row["track"] == "dg_zero_shot" and row["policy"] == "natural"
    ]
    if len(set(methods)) != len(methods) or {row["method"] for row in natural} != set(methods):
        raise ValueError("Natural metric method set differs from the frozen report")
    summaries = []
    for method in methods:
        cells = [row for row in natural if row["method"] == method]
        if len(cells) != len(NATURAL_SITES) or {r["outer_target"] for r in cells} != set(
            NATURAL_SITES
        ):
            raise ValueError(f"Natural metrics require one row per hospital: {method}")
        cells = sorted(cells, key=lambda row: row["outer_target"])
        if any(
            row["mask_replicate"] != "0" or float(row["n"]) != NATURAL_SITES[row["outer_target"]]
            for row in cells
        ):
            raise ValueError(f"Natural metric cohort size or replicate changed: {method}")
        summary = {
            "method": method,
            "track": "dg_zero_shot",
            "policy": "natural",
            "decision_threshold": "0.5",
            "aggregation": "equal_hospital_macro",
            "hospital_count": str(len(NATURAL_SITES)),
            "n_total": str(sum(NATURAL_SITES.values())),
        }
        for metric in NATURAL_METRICS:
            values = [float(row[metric]) for row in cells]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError(f"Invalid natural {metric}: {method}")
            summary[metric] = str(math.fsum(values) / len(values))
        summaries.append(summary)
    return summaries


def natural_results(tables: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    primary = tables[f"{HEART}/primary_estimands.csv"]
    rows = summarize_natural_metrics(
        tables[f"{HEART}/site_policy_metrics.csv"], [row["method"] for row in primary]
    )
    for row, reference in zip(rows, primary, strict=True):
        if not math.isclose(float(row["roc_auc"]), float(reference[AUC]), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Natural AUROC differs from the frozen aggregate: {row['method']}")
    return rows


def natural_csv(rows: list[dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def classification_table(rows: list[dict[str, str]], *, selected: bool = False) -> str:
    if selected:
        rows = [one(rows, "method", method) for method in SELECTED_METHODS]
    return table(
        [
            "Model" if selected else "Method ID",
            "Accuracy ↑",
            "Balanced accuracy ↑",
            "Precision ↑",
            "Recall ↑",
            "F1 ↑",
            "AUROC ↑",
            "Brier ↓",
        ],
        [
            [SELECTED_METHODS[row["method"]] if selected else f"`{row['method']}`"]
            + [f"{100 * float(row[metric]):.2f}%" for metric in NATURAL_METRICS[:5]]
            + [f"{float(row['roc_auc']):.4f}", number(row["brier"])]
            for row in rows
        ],
    )


def classification_summary(rows: list[dict[str, str]]) -> str:
    return (
        "How well do the models classify disease at an unseen hospital using the measurements "
        "as recorded? The table below answers this complementary question. **Natural measurements; "
        "decision threshold 0.5; equal weight for each of the four hospitals.**\n\n"
        + classification_table(rows, selected=True)
        + "\n\nThese five methods illustrate prior separation, pooled training (ERM), the "
        "preselected measurement-robust candidate (DRO), and two classical references. "
        "The classical references use site/class-balanced training. "
        "Values are descriptive point estimates, not a new model-selection result. "
        "AUROC and Brier use probabilities without a decision threshold; Brier here is "
        "unweighted within each hospital.\n\n"
        "[All 45 methods and downloadable table](docs/RESULTS.md#natural-measurement-metrics) · "
        "[Metric definitions and why log loss is primary](docs/METRICS.md)."
    )


def table(headers: list[str], rows: list[list[str]]) -> str:
    numeric = re.compile(r"(?:[+-]?\d+(?:\.\d+)?%?|\[[^]]+\])")
    alignment = [
        "---:" if all(numeric.fullmatch(row[index]) for row in rows) else "---"
        for index in range(len(headers))
    ]
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(alignment) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def section(text: str, heading: str) -> str:
    marker = f"## {heading}\n\n"
    if text.count(marker) != 1:
        raise ValueError(f"Expected one '{heading}' section")
    body = text.split(marker, 1)[1]
    return body.split("\n## ", 1)[0].rstrip()


def findings(tables: dict[str, list[dict[str, str]]], prefix: str) -> str:
    heart = tables[f"{HEART}/primary_estimands.csv"]
    readmission = tables[f"{READMISSION}/primary_estimands.csv"]
    development = tables[f"{DEVELOPMENT}/primary_estimands.csv"]
    stability = tables[f"{STABILITY}/primary_estimands.csv"]
    v2 = one(heart, "method", "psmask:v2_prior_separated")
    v5 = one(heart, "method", "psmask:v5_mask_only_dro")
    v4 = one(stability, "method", "psmask10:v4_structured_policy_bank")
    forest = one(stability, "method", "control:random_forest_natural")
    router = one(development, "method", "router:support_aware_router")
    blend = one(development, "method", "router:equal_logit_blend")
    dro = one(readmission, "experiment", "ps_maskdro")
    erm = one(readmission, "experiment", "pooled_erm")
    return (
        "Balanced log loss (BLL) measures probability quality with equal class weight; "
        "lower is better. The heart primary metric averages each hospital's worst "
        "measurement-policy loss. These experiments have different evidence scopes.\n\n"
        + table(
            ["Experiment", "Recorded result", "Interpretation"],
            [
                [
                    f"[Locked heart evaluation]({prefix}RESULTS.md#locked-heart-evaluation)",
                    f"V2 prior separation: **{number(v2[PRIMARY])}**. "
                    f"Preselected V5 mask-axis DRO: {number(v5[PRIMARY])}.",
                    "V2 had the lowest point estimate among 45 methods. "
                    "V5's registered interval against logistic regression crossed zero.",
                ],
                [
                    f"[Ten-seed sensitivity]({prefix}RESULTS.md#seed-stability)",
                    f"V4 structured policies: **{number(v4[PRIMARY])}**; "
                    f"random forest: {number(forest[PRIMARY])}.",
                    "Post-outcome analysis. Familywise intervals crossed zero; "
                    "superiority remains unconfirmed.",
                ],
                [
                    "[Independent readmission "
                    f"task]({prefix}RESULTS.md#independent-readmission-task)",
                    f"Joint PS-MaskDRO: **{number(dro['ood_worst_mask_balanced_log_loss'])}**; "
                    f"pooled ERM: {number(erm['ood_worst_mask_balanced_log_loss'])}.",
                    "Exploratory direct contrast favors DRO on worst-mask loss. "
                    "ERM retained higher natural-policy AUROC.",
                ],
                [
                    f"[Support-aware routing]({prefix}RESULTS.md#backbones-and-routing)",
                    f"Router: {number(router[PRIMARY])}; "
                    f"equal-logit blend: **{number(blend[PRIMARY])}**.",
                    "The router failed its robust-improvement gate.",
                ],
            ],
        )
        + "\n\nThe heart compatibility gate abstained in all 432 evaluated adaptation cells. "
        "The separate ShiftGuard development study failed its final acceptance gate. "
        "Both outcomes are documented in [adaptation and "
        f"diagnostics]({prefix}RESULTS.md#adaptation-and-diagnostics)."
    )


def result_table(rows: list[dict[str, str]]) -> str:
    return table(
        ["Method ID", "Primary BLL ↓", "Worst site-policy BLL ↓", "Natural macro AUROC ↑"],
        [[f"`{r['method']}`", number(r[PRIMARY]), number(r[WORST]), number(r[AUC])] for r in rows],
    )


def comparisons(rows: list[dict[str, str]], key: str) -> str:
    return table(
        ["Method ID", "Bootstrap mean Δ BLL", "95% percentile interval"],
        [
            [
                f"`{r[key]}`",
                number(r["mean_difference"]),
                f"[{number(r['ci_025'])}, {number(r['ci_975'])}]",
            ]
            for r in rows
        ],
    )


def source(path: str, label: str = "Source table") -> str:
    return f"[{label}](../{path})"


def details(title: str, content: str) -> str:
    return f"<details>\n<summary>{title}</summary>\n\n{content}\n\n</details>"


def render(
    root: Path, tables: dict[str, list[dict[str, str]]], natural: list[dict[str, str]]
) -> str:
    heart = tables[f"{HEART}/primary_estimands.csv"]
    readmission = tables[f"{READMISSION}/primary_estimands.csv"]
    sections = [
        "# Experiment results",
        "These tables reproduce the retained report aggregates. Losses and registered "
        "aggregates use six decimal places; classification rates use two percentage decimals "
        "and the natural-metric table uses four for AUROC. "
        "Source CSVs retain their stored precision. "
        "Method IDs match the report keys, including training weights, calibration, and "
        "seed count. Each primary table includes its complete reported method set.",
        "[Heart](#locked-heart-evaluation) · [Readmission](#independent-readmission-task) · "
        "[Backbones and routing](#backbones-and-routing) · [Seed stability](#seed-stability) · "
        "[Adaptation](#adaptation-and-diagnostics) · [Legacy "
        "work](#legacy-and-systems-experiments)",
        "## Findings",
        findings(tables, ""),
        "## Reading the metrics",
        table(
            ["Quantity", "Definition", "Interpretation"],
            [
                [
                    "Primary heart BLL",
                    "Mean across hospitals of each hospital's worst-policy balanced log loss",
                    "Lower is better; four hospitals receive equal weight.",
                ],
                [
                    "Worst site-policy BLL",
                    "Maximum loss over all hospital/policy cells",
                    "Lower is better; exposes the most difficult observed cell.",
                ],
                [
                    "Natural macro AUROC",
                    "Mean hospital AUROC with the naturally recorded feature panel",
                    "Higher is better; measures discrimination rather than probability quality.",
                ],
                [
                    "Readmission OOD worst-mask BLL",
                    "Worst deletion-policy BLL in the held-out admission-source environment",
                    "A separate endpoint and domain definition; do not pool with heart scores.",
                ],
                [
                    "Bootstrap mean difference",
                    "Mean candidate-minus-reference difference over paired resamples",
                    "Not the subtraction of the two point estimates. Negative "
                    "favors the candidate.",
                ],
            ],
        ),
        "Heart intervals resample records within the four observed hospitals and condition on "
        "the fitted models. Readmission intervals resample patient clusters. Neither estimates "
        "performance over a population of future hospitals. Repeated policies and seeds do not "
        "increase the number of patients. Bootstrap fractions favoring a method are descriptive, "
        "not p-values. [Metric contract](BENCHMARK_CARD.md).",
        "## Locked heart evaluation",
        "**Scope:** zero-shot disease classification on 920 records from four historical "
        "referred cohorts. The endpoint is angiographic disease status (`num > 0`). "
        "The final report combines frozen classical/modern predictions, recovered MIRRAMS "
        "aggregation, and the once-run PS-MaskDRO evaluation. The two mechanical recoveries "
        "remain disclosed in the [execution report](HEART_OUTER_V5_RESULT.md).",
        "![Selected locked heart methods compared on robust probability loss and "
        "natural-policy discrimination](../assets/research/"
        "locked_heart_overview.png)",
        "The figure shows nine named methods for readability, including the preselected "
        "candidate, the logistic reference, random forest, pooled ERM, and MIRRAMS. "
        "It is a descriptive selection made after evaluation. The complete table below "
        "contains all 45 methods, including calibrated variants and weaker results. "
        "[Figure data](../assets/research/locked_heart_overview.csv) · "
        "[SVG](../assets/research/locked_heart_overview.svg).",
        details(
            "All 45 locked methods",
            result_table(heart) + "\n\n" + source(f"{HEART}/primary_estimands.csv"),
        ),
        "### Natural measurement metrics",
        "These scores describe the naturally recorded measurements, including their existing "
        "missingness, with no additional feature deletion. They use the same 45 frozen zero-shot "
        "methods as the primary table. Accuracy, balanced accuracy, precision, recall, and F1 "
        "use the report's fixed probability threshold of **0.5**; AUROC and Brier do not require "
        "a threshold. Positive means angiographic disease (`num > 0`).",
        "Each value is the arithmetic mean of four separately computed hospital scores: "
        "Cleveland (303 records), Hungary (294), Switzerland (123), and VA Long Beach (200). "
        "Every hospital receives equal weight regardless of size. F1 is averaged after "
        "calculation within each hospital. Brier is unweighted within each hospital. "
        "These are not pooled-patient metrics or worst-deletion-policy results. "
        "The point estimates are descriptive and retain the original report order. "
        "[How to interpret the metrics](METRICS.md).",
        details(
            "All 45 methods: accuracy, precision, recall, F1, AUROC, and Brier",
            classification_table(natural),
        ),
        "[Download all 45 rows](research/heart_natural_metrics.csv) · "
        + source(f"{HEART}/site_policy_metrics.csv", "Original hospital/policy scores")
        + " · [Metric implementation](../src/heartshift/metrics.py). "
        "The export records the condition, threshold, and averaging convention. "
        "The generator checks all four hospitals, the complete method set, and agreement "
        "with the frozen natural AUROC aggregate.",
        "### Registered comparisons",
        "All 16 registered comparisons use `classical:logistic:site_class_balanced` as "
        "the reference and 2,000 paired replicates. These are the saved marginal 95% "
        "intervals, without a new familywise adjustment. In particular, the preselected "
        "mask-axis candidate's interval crosses zero. A lowest point estimate across "
        "45 methods does not establish general superiority.",
        details(
            "All 16 registered contrasts",
            comparisons(tables[f"{HEART}/paired_bootstrap_intervals.csv"], "method")
            + "\n\n"
            + source(f"{HEART}/paired_bootstrap_intervals.csv"),
        ),
        "![Registered heart comparisons against logistic regression with paired "
        "bootstrap intervals](../artifacts/figures/heartshift-v5-r2/"
        "heart_registered_bootstrap_forest.png)",
        "### Hospital heterogeneity",
        "![Worst-policy balanced log loss by hospital for the publication figure's "
        "selected methods](../artifacts/figures/heartshift-v5-r2/"
        "heart_site_worst_heatmap.png)",
        "The heatmap is a selected descriptive view. The [complete site-policy table]"
        f"(../{HEART}/site_policy_metrics.csv) contains all methods. Switzerland has only "
        "eight negative records; aggregate precision must not conceal that limitation. "
        "[Recorded-sex strata]"
        f"(../{HEART}/descriptive_sex_subgroup_metrics.csv) are descriptive subgroup evidence, "
        "not a comprehensive fairness evaluation.",
        "## Independent readmission task",
        "**Scope:** patient-disjoint any-readmission classification, with admission source as "
        "a domain proxy. The evaluation covers 54,287 encounters from 39,597 patients. "
        "This task supplies cross-task evidence about measurement robustness. It does not "
        "validate heart-disease prediction. ID and OOD refer to the study's "
        "admission-source environments.",
        table(
            [
                "Method ID",
                "ID natural BLL ↓",
                "OOD natural BLL ↓",
                "OOD worst-mask BLL ↓",
                "OOD natural AUROC ↑",
            ],
            [
                [f"`{r['experiment']}`"]
                + [
                    number(r[k])
                    for k in (
                        "id_natural_balanced_log_loss",
                        "ood_natural_balanced_log_loss",
                        "ood_worst_mask_balanced_log_loss",
                        "ood_natural_roc_auc",
                    )
                ]
                for r in readmission
            ],
        ),
        source(
            f"{READMISSION}/primary_estimands.csv", "All nine methods and additional Brier metrics"
        ),
        "![All nine readmission methods under measurement-policy "
        "shift](../artifacts/figures/heartshift-v5-r2/"
        "readmission_worst_mask.png)",
        "All five registered contrasts below use `logistic_environment_class_balanced` "
        "as the reference and 2,000 patient-cluster replicates. Direct comparisons of "
        "PS-MaskDRO with ERM, prior separation, or ANE are exploratory; their interpretation "
        "and paired-replicate derivation are in the [readmission "
        "report](READMISSION_OUTER_V3_RESULT.md). "
        "Pooled ERM retains the highest natural OOD AUROC, so the robust-loss result does "
        "not imply superiority on every metric.",
        comparisons(tables[f"{READMISSION}/patient_cluster_bootstrap_intervals.csv"], "experiment"),
        source(f"{READMISSION}/patient_cluster_bootstrap_intervals.csv"),
        "## Backbones and routing",
        "**Scope:** development after the heart outcomes had already been inspected. "
        "Source-only selection and a pre-execution development freeze constrain the "
        "implementation; they do not make these reused outcomes confirmatory. The "
        "matched report contains 16 methods. Ensembles and individual architectures "
        "retain their distinct identifiers.",
        details(
            "All 16 development methods",
            result_table(tables[f"{DEVELOPMENT}/primary_estimands.csv"])
            + "\n\n"
            + source(f"{DEVELOPMENT}/primary_estimands.csv"),
        ),
    ]
    gate = json.loads((root / ROUTER).read_text())
    sections.extend(
        [
            table(
                ["Router check", "Recorded outcome"],
                [
                    [key.replace("_", " "), "Passed" if value["passed"] else "Failed"]
                    for key, value in gate["checks"].items()
                ],
            ),
            source(ROUTER, "Router gate record") + ". Passing non-collapse and AUROC checks does "
            "not override the failed robust-improvement criterion. "
            "[Paired contrasts]" + f"(../{DEVELOPMENT}/paired_contrast_intervals.csv) · "
            "[Individual-seed results]" + f"(../{DEVELOPMENT}/individual_seed_sensitivity.csv) · "
            "[Selection and interpretation](HEART_RESEARCH_DEVELOPMENT_RESULT.md).",
            "## Seed stability",
            "**Scope:** post-outcome exact seed extension. All ten historical PS-MaskDRO "
            "variants were extended from three to ten seeds using their retained selections. "
            "The original three-seed predictions reproduced exactly. The stability report "
            "contains those 20 ensembles and two classical controls; they are shown together "
            "only within this sensitivity study.",
            "![All ten seed extensions with familywise uncertainty for the "
            "highlighted V4 comparison](../assets/research/ensemble_stability.png)",
            details(
                "All 22 stability-report methods",
                result_table(tables[f"{STABILITY}/primary_estimands.csv"])
                + "\n\n"
                + source(f"{STABILITY}/primary_estimands.csv"),
            ),
            "V4 was highlighted after outcome inspection. Its familywise intervals against "
            "natural random forest cross zero. The complete ten-method adjustment is shown "
            "below; the highlighted method is not treated as a new prespecified winner.",
            details(
                "Familywise intervals for all ten extended variants",
                table(
                    [
                        "Method ID",
                        "Bonferroni percentile 95% interval",
                        "Joint max-error 95% interval",
                    ],
                    [
                        [
                            f"`{r['method']}`",
                            f"[{number(r['bonferroni_percentile_ci_lower'])}, "
                            f"{number(r['bonferroni_percentile_ci_upper'])}]",
                            f"[{number(r['simultaneous_max_error_ci_lower'])}, "
                            f"{number(r['simultaneous_max_error_ci_upper'])}]",
                        ]
                        for r in tables[f"{STABILITY}/posthoc_multiplicity_sensitivity.csv"]
                    ],
                )
                + "\n\n"
                + source(f"{STABILITY}/posthoc_multiplicity_sensitivity.csv"),
            ),
            "[Three-versus-ten-seed data]" + f"(../{STABILITY}/three_vs_ten_seed_comparison.csv) · "
            "[Exact reproduction audit]" + f"(../{STABILITY}/exact_reproduction.json) · "
            "[Single-seed sensitivity]" + f"(../{STABILITY}/ten_seed_sensitivity.csv).",
            "## Adaptation and diagnostics",
            "### Heart abstention",
            "The compatibility gate accepted zero of 432 adaptable method/site/policy/replicate "
            "cells. The automatic report therefore contains only zero-shot predictions. "
            "The following post-hoc analysis scores the already-fixed research probabilities "
            "to explain what the gate rejected. These probabilities were not "
            "automatic UDA outputs.",
            table(
                ["Variant", "Research track", "Macro worst-policy BLL ↓"],
                [
                    [
                        f"`{r['experiment']}`",
                        r["track"],
                        number(r["macro_site_worst_balanced_log_loss"]),
                    ]
                    for r in tables[ADAPTATION]
                ],
            ),
            source(ADAPTATION, "Complete plotted-data export") + " · "
            "[Original abstention "
            "analysis](HEART_OUTER_V5_RESULT.md#adaptation-abstention-result).",
            "### Registered synthetic mechanisms",
            "The earlier synthetic v3 study passed all eight registered checks on its named "
            "mechanisms. This is a separate protocol from the later ShiftGuard revisions. "
            "Success on this mechanism bank does not establish unrestricted MNAR or "
            "concept-shift robustness.",
        ]
    )
    synthetic = json.loads((root / SYNTHETIC).read_text())
    sections.extend(
        [
            table(
                ["Registered check", "Observed", "Requirement", "Outcome"],
                [
                    [
                        f"`{r['name']}`",
                        number(r["observed"]),
                        f"{r['operator']} {number(r['threshold'])}",
                        "Passed" if r["passed"] else "Failed",
                    ]
                    for r in synthetic["checks"]
                ],
            ),
            source(SYNTHETIC, "Synthetic v3 gate")
            + " · [Execution and reconstruction](SYNTHETIC_V3_RESULT.md).",
            "### ShiftGuard development",
            "Five full revisions are reported below; v5/v6 pilots remain in the "
            "[experiment ledger](research/EXPERIMENT_LEDGER.md). The revision mechanism banks "
            "and power differ, so this is a development sequence rather than a matched "
            "cross-version efficacy experiment. Every full revision exceeds the 5% maximum "
            "observable-invalid acceptance gate.",
            "![Full ShiftGuard revisions with invalid acceptance and valid "
            "label-shift acceptance](../assets/research/"
            "shiftguard_revision_limits.png)",
            table(
                [
                    "Full revision",
                    "Valid acceptance",
                    "Observable-invalid acceptance",
                    "Concept-control acceptance",
                    "Concept Δ log loss",
                ],
                [
                    [
                        f"`{r['revision']}`",
                        *(
                            f"{100 * float(r[k]):.2f}%"
                            for k in (
                                "valid_acceptance",
                                "observable_invalid_acceptance",
                                "concept_acceptance",
                            )
                        ),
                        number(r["concept_gated_minus_zero_log_loss"]),
                    ]
                    for r in tables[f"{REVISIONS}/revision_comparison.csv"]
                ],
            ),
            source(f"{REVISIONS}/revision_comparison.csv")
            + ". Positive concept-control Δ log loss "
            "means adaptation worsened the score. The concept-reversal counterexample remains "
            "visible alongside observable-shift failures. [Revision "
            "interpretation](SHIFTGUARD_DEVELOPMENT_RESULT.md).",
            "## Legacy and systems experiments",
            "The [original four-model holdout table](legacy/LEGACY_BENCHMARK.md) belongs to a "
            "repeatedly inspected coursework split. It is historical development evidence and "
            "is not comparable to the leave-one-hospital-out results above. "
            "[Recovered coursework](legacy/COURSEWORK_PROVENANCE.md) includes per-fold CV, "
            "calibration, SHAP, descriptive subgroups, and training diagnostics.",
            "The Dask trial retains its row/label alignment failure. The Colab latency trial "
            "measures its recorded loopback workload; its historical `throughput_rps` field "
            "is a reciprocal-latency proxy. The eICU public-demo run is a "
            "[pipeline smoke test](EICU_DEMO_SMOKE_RESULT.md). None supplies "
            "clinical deployment evidence.",
            "## Reproduce these tables",
            "```powershell\nuv run python tools/build_results_document.py --check\n"
            "uv run python tools/plot_research_overview.py --check\n```",
            "Omit `--check` to rebuild the presentation from the same saved reports. No model "
            "is fitted or selected. The [presentation manifest](research/"
            "result_presentation_manifest.json) "
            "binds all input tables and gate records to this document. Original prediction "
            "reconstruction is a separate [full-evidence check](USAGE.md#evidence-checkout). "
            "[Development and attribution](RESEARCH_ATLAS.md) · "
            "[Complete run ledger](research/EXPERIMENT_LEDGER.md).",
        ]
    )
    return "\n\n".join(sections) + "\n"


def manifest(root: Path, outputs: dict[str, str], summaries: dict[str, str]) -> str:
    payload: dict[str, Any] = {
        "schema_version": "1.1.0",
        "scope": "presentation_of_existing_results_no_new_evaluation",
        "generator": "tools/build_results_document.py",
        "sources": [
            {"path": path, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()}
            for path in [*TABLES, *JSON_SOURCES]
        ],
        "outputs": [
            {"path": path, "sha256": hashlib.sha256(content.encode()).hexdigest()}
            for path, content in outputs.items()
        ],
        "readme_findings_sha256": hashlib.sha256(summaries["Findings"].encode()).hexdigest(),
        "readme_classification_sha256": hashlib.sha256(
            summaries["Classification results"].encode()
        ).hexdigest(),
        "readme_section_normalization": "utf8_lf_without_trailing_newline",
        "natural_metrics": {
            "source": f"{HEART}/site_policy_metrics.csv",
            "track": "dg_zero_shot",
            "policy": "natural",
            "mask_replicate": 0,
            "decision_threshold": 0.5,
            "thresholded_metrics": list(NATURAL_METRICS[:5]),
            "aggregation": "arithmetic_mean_of_four_hospital_metrics",
            "hospital_records": NATURAL_SITES,
            "within_hospital_brier_weighting": "unweighted",
            "method_count": 45,
        },
        "expected_rows": {path: count for path, (count, _) in TABLES.items()},
    }
    return json.dumps(payload, indent=2) + "\n"


def build(root: Path, *, check: bool) -> None:
    tables = read_tables(root)
    natural = natural_results(tables)
    outputs = {DOCUMENT: render(root, tables, natural), NATURAL_EXPORT: natural_csv(natural)}
    summaries = {
        "Findings": findings(tables, "docs/"),
        "Classification results": classification_summary(natural),
    }
    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    current = {heading: section(readme, heading) for heading in summaries}
    outputs[MANIFEST] = manifest(root, outputs, summaries)
    if check:
        for heading, summary in summaries.items():
            if current[heading] != summary:
                raise ValueError(f"README {heading.lower()} do not match the report tables")
        for path, expected in outputs.items():
            if (root / path).read_bytes() != expected.encode("utf-8"):
                raise ValueError(
                    f"Result presentation is stale or its source bindings changed: {path}"
                )
    else:
        for heading, summary in summaries.items():
            readme = readme.replace(
                f"## {heading}\n\n{current[heading]}", f"## {heading}\n\n{summary}", 1
            )
        readme_path.write_text(readme, encoding="utf-8", newline="\n")
        for path, content in outputs.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(content, encoding="utf-8", newline="\n")
    print(
        "Complete result tables, natural metrics, README summaries, and source bindings verified."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    build(ROOT, check=parser.parse_args().check)


if __name__ == "__main__":
    main()
