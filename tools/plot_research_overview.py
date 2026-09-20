"""Plot the complete seed extension and full ShiftGuard revisions from saved tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "assets/research"
STABILITY = "artifacts/reports/historical-psmask-ten-seed-sensitivity-v2-stability"
SHIFTGUARD = "artifacts/reports/shiftguard-revisions-v3-final-development"
SOURCES = [
    f"{STABILITY}/three_vs_ten_seed_comparison.csv",
    f"{STABILITY}/primary_estimands.csv",
    f"{STABILITY}/paired_bootstrap_intervals.csv",
    f"{STABILITY}/posthoc_multiplicity_sensitivity.csv",
    f"{SHIFTGUARD}/revision_comparison.csv",
]
LABELS = {
    "v0_pooled_erm": "V0 pooled ERM",
    "v1_site_balanced": "V1 site balanced",
    "v2_prior_separated": "V2 prior separated",
    "v3_mcar_augmentation": "V3 MCAR augmentation",
    "v4_structured_policy_bank": "V4 structured policies",
    "v5_site_only_dro": "V5 site-axis DRO",
    "v5_mask_only_dro": "V5 mask-axis DRO",
    "v5_site_mask_dro": "V5 joint DRO",
    "v5_site_mask_dro_brier": "V5 joint DRO + Brier",
    "v7_ane": "V7 acquisition-neutral evidence",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(root: Path, relative: str) -> list[dict[str, str]]:
    with (root / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def exactly_one(rows: list[dict[str, str]], **fields: str) -> dict[str, str]:
    matches = [row for row in rows if all(row[key] == value for key, value in fields.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one evidence row for {fields}; got {len(matches)}")
    return matches[0]


def figure_data(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    rows = read_rows(root, SOURCES[0])
    if len(rows) != len(LABELS) or {row["experiment"] for row in rows} != set(LABELS):
        raise ValueError("Seed figure must include all ten frozen variants exactly once")
    seeds = []
    for experiment, label in LABELS.items():
        row = exactly_one(rows, experiment=experiment)
        seeds.append(
            {
                "experiment": experiment,
                "label": label,
                "three_seed_bll": float(row["macro_site_worst_mask_balanced_log_loss_3seed"]),
                "ten_seed_bll": float(row["macro_site_worst_mask_balanced_log_loss_10seed"]),
                "evidence_scope": "post_outcome_sensitivity",
            }
        )
    paired = exactly_one(
        read_rows(root, SOURCES[2]),
        method="psmask10:v4_structured_policy_bank",
        reference_method="control:random_forest_natural",
    )
    adjusted = exactly_one(
        read_rows(root, SOURCES[3]),
        method="psmask10:v4_structured_policy_bank",
    )
    intervals = []
    for label, row, lower, upper in [
        ("Unadjusted percentile", paired, "percentile_ci_025", "percentile_ci_975"),
        (
            "Bonferroni (10 methods)",
            adjusted,
            "bonferroni_percentile_ci_lower",
            "bonferroni_percentile_ci_upper",
        ),
        (
            "Joint max-error",
            adjusted,
            "simultaneous_max_error_ci_lower",
            "simultaneous_max_error_ci_upper",
        ),
    ]:
        intervals.append(
            {
                "interval": label,
                "observed_difference": float(paired["observed_difference"]),
                "lower": float(row[lower]),
                "upper": float(row[upper]),
                "evidence_scope": "post_outcome_sensitivity",
            }
        )
    revisions = read_rows(root, SOURCES[4])
    if [row["revision"] for row in revisions] != [
        "v1_learned_mean",
        "v2_omnibus_mean",
        "v3_max_moment",
        "v4_spectral_ridge",
        "v7_power_guard",
    ]:
        raise ValueError("Unexpected full-revision selection; update the figure scope explicitly")
    return seeds, intervals, revisions


def csv_text(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def plot(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    seeds, intervals, revisions = figure_data(root)
    out = root / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": "#273b4c",
            "text.color": "#172f42",
            "axes.edgecolor": "#d5dee6",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "svg.hashsalt": "heartshift-research-overview-v1",
            "svg.fonttype": "none",
        }
    )

    def finish(fig: Any, name: str) -> None:
        fig.savefig(out / f"{name}.png", dpi=180, facecolor="white")
        fig.savefig(out / f"{name}.svg", metadata={"Date": None}, facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.22, right=0.97, top=0.79, bottom=0.30, wspace=0.55)
    fig.text(
        0.04, 0.94, "Averaging helped; superiority remains unconfirmed", fontsize=21, weight="bold"
    )
    fig.text(
        0.04,
        0.89,
        "HEART  |  Post-outcome sensitivity  |  Same four hospitals and consumed outcomes",
        color="#586d7e",
    )
    ax = axes[0]
    for index, row in enumerate(seeds):
        ax.plot([row["three_seed_bll"], row["ten_seed_bll"]], [index, index], color="#aebcc7", lw=2)
        ax.scatter(
            row["three_seed_bll"],
            index,
            color="#327aa2",
            marker="o",
            s=58,
            facecolors="white",
            zorder=3,
            label="3 seeds" if index == 0 else None,
        )
        ax.scatter(
            row["ten_seed_bll"],
            index,
            color="#087f75",
            marker="D",
            s=45,
            zorder=4,
            label="10 seeds" if index == 0 else None,
        )
    ax.set_yticks(range(len(seeds)), [row["label"] for row in seeds])
    ax.tick_params(axis="y", length=0, pad=10)
    ax.invert_yaxis()
    ax.set_xlabel("Macro hospital worst-policy balanced log loss\nLower is better")
    ax.set_title("All ten frozen variants", loc="left", pad=16)
    ax.grid(axis="x", color="#e2e8ee")
    ax.set_axisbelow(True)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.41, 0.11),
        ncol=2,
        frameon=False,
    )
    ax = axes[1]
    ax.axvline(0, color="#596f7e", linestyle="--", linewidth=1.2)
    for index, row in enumerate(intervals):
        ax.plot([row["lower"], row["upper"]], [index, index], color="#327aa2", linewidth=3)
        ax.scatter(row["observed_difference"], index, color="#172f42", s=48, zorder=3)
        ax.text(-0.075, index - 0.20, row["interval"], fontsize=10, color="#586d7e")
    ax.set_ylim(2.65, -0.65)
    ax.set_xlim(-0.078, 0.030)
    ax.set_yticks([])
    ax.set_title("V4 versus natural random forest", loc="left", pad=16)
    ax.set_xlabel("Balanced log-loss difference\nNegative favors V4")
    ax.grid(axis="x", color="#e2e8ee")
    fig.text(
        0.04,
        0.055,
        "Dots on the right show the observed difference; lines show 95% intervals "
        "conditional on the observed hospitals.\n"
        "Both familywise intervals cross zero. V4 was highlighted after inspecting outcomes; "
        "it is not a new confirmatory winner.",
        fontsize=10,
        color="#586d7e",
    )
    finish(fig, "ensemble_stability")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.20, right=0.95, top=0.76, bottom=0.24, wspace=0.30)
    fig.text(
        0.04,
        0.94,
        "ShiftGuard improved, then failed its acceptance gate",
        fontsize=21,
        weight="bold",
    )
    fig.text(
        0.04,
        0.885,
        "SYNTHETIC DEVELOPMENT  |  Full revisions only  |  "
        "V5 and V6 pilots remain in the experiment ledger",
        color="#586d7e",
    )
    labels = [
        "V1 learned mean",
        "V2 omnibus mean",
        "V3 max moment",
        "V4 spectral ridge",
        "V7 power guard",
    ]
    for ax, key, color, title in [
        (axes[0], "observable_invalid_acceptance", "#b54a3e", "Observable-invalid shift accepted"),
        (axes[1], "valid_acceptance", "#087f75", "Valid label shift accepted"),
    ]:
        values = [float(row[key]) for row in revisions]
        ax.barh(range(len(values)), values, color=color, height=0.52)
        ax.set_yticks(range(len(values)), labels if ax is axes[0] else [])
        ax.tick_params(axis="y", length=0, pad=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 0.67 if ax is axes[0] else 1.16)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks([0, 0.2, 0.4, 0.6] if ax is axes[0] else [0, 0.25, 0.5, 0.75, 1])
        ax.set_title(title, loc="left", pad=20)
        ax.set_xlabel("Acceptance rate (point estimate)")
        ax.grid(axis="x", color="#e2e8ee")
        ax.set_axisbelow(True)
        for index, value in enumerate(values):
            ax.text(value + 0.012, index, f"{100 * value:.2f}%", va="center", fontsize=10)
    axes[0].axvline(0.05, color="#172f42", linestyle="--", lw=1.5)
    axes[0].text(0.06, -0.62, "5% maximum gate", fontsize=10)
    last = revisions[-1]
    fig.text(
        0.04,
        0.08,
        "All five full revisions exceed the 5% invalid-acceptance gate. "
        "The mechanism families and power differ across revisions.\n"
        "V7 also accepts the unidentifiable concept-reversal control "
        f"{100 * float(last['concept_acceptance']):.1f}% of the time; "
        "adaptation raises its log loss by "
        f"{float(last['concept_gated_minus_zero_log_loss']):.5f}.",
        fontsize=10,
        color="#586d7e",
    )
    finish(fig, "shiftguard_revision_limits")

    for name, rows in [("ensemble_stability.csv", seeds), ("v4_interval_context.csv", intervals)]:
        (out / name).write_text(csv_text(rows), encoding="utf-8", newline="\n")
    (out / "shiftguard_revision_limits.csv").write_bytes((root / SOURCES[4]).read_bytes())
    files = sorted(path for path in out.iterdir() if path.suffix in {".png", ".svg", ".csv"})
    manifest = {
        "schema_version": "1.0.0",
        "scope": "presentation_of_existing_evidence_no_new_fits",
        "generator": "tools/plot_research_overview.py",
        "sources": [{"path": path, "sha256": sha256(root / path)} for path in SOURCES],
        "outputs": [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)} for path in files
        ],
    }
    (out / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def check(root: Path) -> None:
    seeds, intervals, _ = figure_data(root)
    manifest = json.loads((root / OUTPUT / "figure_manifest.json").read_text(encoding="utf-8"))
    if [row["path"] for row in manifest["sources"]] != SOURCES:
        raise ValueError("Figure input set changed")
    expected_outputs = {
        f"{OUTPUT}/{name}.{suffix}"
        for name in ("ensemble_stability", "shiftguard_revision_limits")
        for suffix in ("png", "svg", "csv")
    } | {f"{OUTPUT}/v4_interval_context.csv"}
    if {row["path"] for row in manifest["outputs"]} != expected_outputs:
        raise ValueError("Figure output set changed")
    if len(manifest["outputs"]) != len(expected_outputs):
        raise ValueError("Duplicate figure output binding")
    for row in manifest["sources"] + manifest["outputs"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != row["sha256"]:
            raise ValueError(f"Figure binding mismatch: {row['path']}")
    for name, rows in [("ensemble_stability.csv", seeds), ("v4_interval_context.csv", intervals)]:
        if (root / OUTPUT / name).read_text(encoding="utf-8") != csv_text(rows):
            raise ValueError(f"Plotted data do not match the frozen tables: {name}")
    if (root / OUTPUT / "shiftguard_revision_limits.csv").read_bytes() != (
        root / SOURCES[4]
    ).read_bytes():
        raise ValueError("Revision figure data changed")
    print("Figure input hashes, plotted tables, and output hashes verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check(ROOT)
    else:
        plot(ROOT)
        check(ROOT)


if __name__ == "__main__":
    main()
