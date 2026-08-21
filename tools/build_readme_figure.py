#!/usr/bin/env python3
"""Build the README performance figure from tracked holdout metrics."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.hashsalt"] = "heart-disease-risk-model-benchmark"


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "main" / "metrics" / "test"
OUTPUT_DIR = ROOT / "assets"

MODEL_NAMES = {
    "LightGBM": "LightGBM",
    "LogReg": "Logistic regression",
    "XGBoost": "XGBoost",
    "TabNet": "TabNet",
}
MODEL_ORDER = ("LogReg", "LightGBM", "XGBoost", "TabNet")


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    """Index a metrics CSV by model name."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["model"]: row for row in csv.DictReader(handle)}


def normalize_svg(path: Path) -> None:
    """Remove generator whitespace so Git's patch checks remain clean."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    axis.set_xlim(0.78, 0.98)
    axis.set_xticks([0.80, 0.85, 0.90, 0.95])
    axis.grid(axis="x", color="#DCE3EA", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, labelsize=10)
    axis.tick_params(axis="x", colors="#536273", labelsize=9)


def main() -> None:
    intervals = read_rows(METRICS_DIR / "holdout_bootstrap_ci.csv")
    y_positions = list(range(len(MODEL_ORDER)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    fig.patch.set_facecolor("white")

    panels = (("auc", "ROC-AUC"), ("f1", "F1 score"))
    for axis, (metric, title) in zip(axes, panels, strict=True):
        style_axis(axis, title)
        means = [float(intervals[model][f"{metric}_mean"]) for model in MODEL_ORDER]
        winner = max(range(len(means)), key=means.__getitem__)

        for index, model in enumerate(MODEL_ORDER):
            row = intervals[model]
            mean = float(row[f"{metric}_mean"])
            low = float(row[f"{metric}_ci_low"])
            high = float(row[f"{metric}_ci_high"])
            color = "#087F5B" if index == winner else "#2563EB"
            axis.errorbar(
                mean,
                index,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2.2,
                capsize=4,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.1,
                zorder=3,
            )
            axis.text(
                min(high + 0.004, 0.971),
                index,
                f"{mean:.3f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#172033",
                fontweight="semibold" if index == winner else "normal",
            )

        axis.set_yticks(y_positions, [MODEL_NAMES[model] for model in MODEL_ORDER])
        axis.invert_yaxis()

    fig.suptitle(
        "Holdout performance with uncertainty",
        x=0.075,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#172033",
    )
    fig.text(
        0.075,
        0.91,
        "Mean and 95% bootstrap interval on the stratified test split",
        ha="left",
        fontsize=10,
        color="#536273",
    )
    fig.text(
        0.075,
        0.03,
        "Intervals overlap; highlighted points indicate the observed leader for each metric.",
        ha="left",
        fontsize=9,
        color="#536273",
    )
    fig.subplots_adjust(left=0.19, right=0.98, top=0.82, bottom=0.15, wspace=0.28)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "heart-disease-risk-model-benchmark", "Date": None}
    fig.savefig(OUTPUT_DIR / "holdout_performance_intervals.png", dpi=180, metadata=metadata)
    svg_path = OUTPUT_DIR / "holdout_performance_intervals.svg"
    fig.savefig(svg_path, metadata=metadata)
    normalize_svg(svg_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
