"""Generate manuscript figures and their exact plotted-data tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from heartshift.data.uci import sha256_file
from heartshift.metrics import balanced_log_loss

SHORT_LABELS = {
    "psmask:v2_prior_separated": "V2 prior-separated",
    "psmask:v3_mcar_augmentation": "V3 MCAR augmentation",
    "psmask:v4_structured_policy_bank": "V4 structured policies",
    "psmask:v5_mask_only_dro": "V5 mask-axis DRO",
    "psmask:v5_site_mask_dro": "V5 joint DRO",
    "psmask:v5_site_mask_dro_brier": "V5 joint DRO + Brier",
    "psmask:v5_site_only_dro": "V5 site-axis DRO",
    "psmask:v7_ane": "V7 ANE",
    "psmask:v0_pooled_erm": "V0 pooled ERM",
    "classical:logistic:site_class_balanced": "Logistic reference",
    "classical:random_forest:site_class_balanced": "Random forest",
    "mirrams:mirrams_equation9": "MIRRAMS Eq. 9",
    "tabpfn_v3:tabpfn:pooled": "TabPFN v3",
    "modern_2026:tabicl:pooled": "TabICL",
    "modern_v2:tabpfn:pooled": "TabPFN v2",
}
SCATTER_HIGHLIGHTS = [
    "psmask:v2_prior_separated",
    "psmask:v5_mask_only_dro",
    "psmask:v0_pooled_erm",
    "classical:logistic:site_class_balanced",
    "classical:random_forest:site_class_balanced",
    "mirrams:mirrams_equation9",
    "tabpfn_v3:tabpfn:pooled",
]
READMISSION_LABELS = {
    "ps_maskdro": "Joint PS-MaskDRO",
    "ps_maskdro_ane": "Joint PS-MaskDRO + ANE",
    "prior_separated": "Prior-separated",
    "pooled_erm": "Pooled observed-set ERM",
    "mirrams_equation9": "MIRRAMS Equation 9",
    "catboost_environment_class_balanced": "CatBoost (environment/class balanced)",
    "lightgbm_environment_class_balanced": "LightGBM (environment/class balanced)",
    "logistic_pooled": "Logistic (pooled)",
    "logistic_environment_class_balanced": "Logistic reference",
}


def _label(method: str) -> str:
    if method in SHORT_LABELS:
        return SHORT_LABELS[method]
    calibration_suffix = ":source_oof_platt"
    if method.endswith(calibration_suffix):
        return _label(method.removesuffix(calibration_suffix)) + " + source-OOF Platt"
    return method.replace("classical:", "").replace("modern_2026:", "")


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=220, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def _heart_site_worst(metrics: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["track"].eq("dg_zero_shot") & metrics["method"].isin(methods)
    ]
    policy_means = selected.groupby(
        ["method", "outer_target", "policy"], as_index=False
    ).agg(balanced_log_loss=("balanced_log_loss", "mean"))
    site_worst = policy_means.groupby(["method", "outer_target"], as_index=False).agg(
        worst_policy_balanced_log_loss=("balanced_log_loss", "max")
    )
    site_worst["method_label"] = site_worst["method"].map(_label)
    return site_worst


def _research_adaptation(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[predictions["variant"].isin(["v2", "v5_mask"])]
    score_columns = {
        "zero-shot": "y_score_zero_shot",
        "equal-prior calibrated": "y_score_calibrated_equal_prior",
        "soft-BBSE (research only)": "y_score_uda_soft_bbse_research",
        "MLLS (research only)": "y_score_uda_mlls_research",
    }
    records = []
    grouping = ["experiment", "outer_target", "policy", "mask_replicate"]
    for keys, group in selected.groupby(grouping, sort=False):
        for track, column in score_columns.items():
            records.append(
                {
                    **dict(zip(grouping, keys, strict=True)),
                    "track": track,
                    "balanced_log_loss": balanced_log_loss(group["target"], group[column]),
                }
            )
    cells = pd.DataFrame(records)
    policy_means = cells.groupby(
        ["experiment", "track", "outer_target", "policy"], as_index=False
    ).agg(balanced_log_loss=("balanced_log_loss", "mean"))
    summaries = []
    for (experiment, track), group in policy_means.groupby(
        ["experiment", "track"], sort=False
    ):
        summaries.append(
            {
                "experiment": experiment,
                "experiment_label": _label(f"psmask:{experiment}"),
                "track": track,
                "macro_site_worst_balanced_log_loss": float(
                    group.groupby("outer_target")["balanced_log_loss"].max().mean()
                ),
                "worst_site_policy_balanced_log_loss": float(
                    group["balanced_log_loss"].max()
                ),
            }
        )
    return pd.DataFrame(summaries)


def build_publication_figures(
    repo_root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write figures, plotted data, and a post-hoc provenance manifest."""
    output_dir.mkdir(parents=True, exist_ok=False)
    sns.set_theme(style="whitegrid", context="paper")
    heart_dir = repo_root / str(config["heart_report_dir"])
    readmission_dir = repo_root / str(config["readmission_report_dir"])
    primary = pd.read_csv(heart_dir / "primary_estimands.csv")
    intervals = pd.read_csv(heart_dir / "paired_bootstrap_intervals.csv")
    metrics = pd.read_csv(heart_dir / "site_policy_metrics.csv")
    readmission = pd.read_csv(readmission_dir / "primary_estimands.csv")
    psmask_predictions = pd.read_parquet(repo_root / str(config["psmask_predictions"]))

    data_paths: dict[str, Path] = {}
    figure_paths: list[Path] = []

    scatter = primary.loc[
        :,
        ["method", "macro_site_worst_mask_balanced_log_loss", "macro_natural_roc_auc"],
    ].copy()
    scatter["method_label"] = scatter["method"].map(_label)
    scatter["highlight"] = scatter["method"].isin(SCATTER_HIGHLIGHTS)
    data_paths["heart_robustness_vs_auc"] = output_dir / "heart_robustness_vs_auc.csv"
    scatter.to_csv(data_paths["heart_robustness_vs_auc"], index=False)
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    background = scatter.loc[~scatter["highlight"]]
    axis.scatter(
        background["macro_site_worst_mask_balanced_log_loss"],
        background["macro_natural_roc_auc"],
        color="0.72",
        alpha=0.7,
        s=28,
        label="Other registered methods",
    )
    palette = sns.color_palette("colorblind", n_colors=len(SCATTER_HIGHLIGHTS))
    for color, method in zip(palette, SCATTER_HIGHLIGHTS, strict=True):
        row = scatter.loc[scatter["method"].eq(method)].iloc[0]
        axis.scatter(
            row["macro_site_worst_mask_balanced_log_loss"],
            row["macro_natural_roc_auc"],
            color=color,
            edgecolor="white",
            linewidth=0.5,
            s=54,
            label=row["method_label"],
        )
    axis.set_xlabel("Macro site-worst mask balanced log loss (lower is better)")
    axis.set_ylabel("Macro natural-policy AUROC (higher is better)")
    axis.set_title("HeartShift robustness-discrimination trade-off")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    figure_paths.extend(_save_figure(figure, output_dir, "heart_robustness_vs_auc"))

    forest = intervals.sort_values("mean_difference").copy()
    forest["method_label"] = forest["method"].map(_label)
    data_paths["heart_registered_bootstrap_forest"] = (
        output_dir / "heart_registered_bootstrap_forest.csv"
    )
    forest.to_csv(data_paths["heart_registered_bootstrap_forest"], index=False)
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    y = np.arange(len(forest))
    axis.errorbar(
        forest["mean_difference"],
        y,
        xerr=np.vstack(
            [
                forest["mean_difference"] - forest["ci_025"],
                forest["ci_975"] - forest["mean_difference"],
            ]
        ),
        fmt="o",
        color=sns.color_palette("deep")[0],
        ecolor="0.35",
        capsize=2.5,
    )
    axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(y, forest["method_label"])
    axis.invert_yaxis()
    axis.set_xlabel("Candidate minus logistic primary BLL (paired 95% interval)")
    axis.set_title("Registered heart comparisons")
    figure_paths.extend(_save_figure(figure, output_dir, "heart_registered_bootstrap_forest"))

    site_worst = _heart_site_worst(
        metrics, [str(value) for value in config["selected_site_methods"]]
    )
    data_paths["heart_site_worst_heatmap"] = output_dir / "heart_site_worst_heatmap.csv"
    site_worst.to_csv(data_paths["heart_site_worst_heatmap"], index=False)
    matrix = site_worst.pivot(
        index="method_label", columns="outer_target", values="worst_policy_balanced_log_loss"
    )
    matrix = matrix.reindex([_label(str(value)) for value in config["selected_site_methods"]])
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="mako_r",
        ax=axis,
        cbar_kws={"label": "BLL"},
    )
    axis.set_xlabel("Held-out hospital")
    axis.set_ylabel("")
    axis.set_title("Worst measurement-policy loss by observed hospital")
    figure_paths.extend(_save_figure(figure, output_dir, "heart_site_worst_heatmap"))

    adaptation = _research_adaptation(psmask_predictions)
    adaptation_order = [
        "zero-shot",
        "equal-prior calibrated",
        "soft-BBSE (research only)",
        "MLLS (research only)",
    ]
    adaptation["track"] = pd.Categorical(
        adaptation["track"], categories=adaptation_order, ordered=True
    )
    adaptation = adaptation.sort_values(["track", "experiment_label"])
    data_paths["heart_rejected_adaptation"] = output_dir / "heart_rejected_adaptation.csv"
    adaptation.to_csv(data_paths["heart_rejected_adaptation"], index=False)
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    sns.barplot(
        data=adaptation,
        x="track",
        y="macro_site_worst_balanced_log_loss",
        hue="experiment_label",
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("Macro site-worst balanced log loss")
    axis.set_title("Rejected label-shift corrections would cause negative transfer")
    axis.tick_params(axis="x", rotation=18)
    axis.legend(title="")
    figure_paths.extend(_save_figure(figure, output_dir, "heart_rejected_adaptation"))

    readmission = readmission.sort_values("ood_worst_mask_balanced_log_loss").copy()
    readmission["method_label"] = readmission["experiment"].map(READMISSION_LABELS)
    data_paths["readmission_worst_mask"] = output_dir / "readmission_worst_mask.csv"
    readmission.to_csv(data_paths["readmission_worst_mask"], index=False)
    figure, axis = plt.subplots(figsize=(8.0, 5.4))
    sns.barplot(
        data=readmission,
        y="method_label",
        x="ood_worst_mask_balanced_log_loss",
        color=sns.color_palette("deep")[0],
        ax=axis,
    )
    axis.set_xlabel("OOD worst-mask balanced log loss (lower is better)")
    axis.set_ylabel("")
    axis.set_title("Independent patient-disjoint readmission result")
    figure_paths.extend(_save_figure(figure, output_dir, "readmission_worst_mask"))

    input_paths = {
        "heart_report_audit": heart_dir / "evidence_audit.json",
        "psmask_run_audit": repo_root / str(config["psmask_audit"]),
        "readmission_report_audit": readmission_dir / "evidence_audit.json",
    }
    output_hashes = {
        path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in sorted([*data_paths.values(), *figure_paths])
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "posthoc_descriptive_publication_figures",
                "protocol_version": config["protocol_version"],
                "input_sha256": {
                    name: sha256_file(path) for name, path in input_paths.items()
                },
                "output_sha256": output_hashes,
                "caution": (
                    "Figures summarize locked results; research-only adaptation scoring is "
                    "post-hoc and not a deployable UDA track."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {**data_paths, "manifest": manifest_path}
