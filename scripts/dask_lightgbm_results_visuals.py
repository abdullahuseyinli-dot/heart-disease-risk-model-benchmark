#!/usr/bin/env python
"""Render the retained single-node versus Dask LightGBM comparison."""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# ---------------------------------------------------------------------
# 0. Paths and helper
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "dask"
FIG_DIR = RESULTS_DIR / "figures" / "model_performance"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close()

# ---------------------------------------------------------------------
# 1. Load the Colab summary CSV
# ---------------------------------------------------------------------

# Try a couple of common locations
candidates = [
    PROJECT_ROOT / "lightgbm_distributed_summary_colab.csv",
    RESULTS_DIR / "metrics" / "lightgbm_distributed_summary_colab.csv",
]

summary_path = None
for p in candidates:
    if p.exists():
        summary_path = p
        break

if summary_path is None:
    raise FileNotFoundError(
        "Could not find lightgbm_distributed_summary_colab.csv in "
        "project root or results/metrics/. Move the file into one of "
        "those locations and rerun."
    )

df = pd.read_csv(summary_path)
if df.empty:
    raise ValueError("lightgbm_distributed_summary_colab.csv is empty.")

print("Loaded summary from:", summary_path)
print(df.to_string(index=False))

# Human-friendly labels
label_map = {
    "LightGBM_single_node": "Single-node",
    "LightGBM_DaskClassifier": "Dask (distributed)",
}
df["label"] = df["model"].map(label_map).fillna(df["model"])

n_models = len(df)

# ---------------------------------------------------------------------
# 2. AUC & F1 bar chart
# ---------------------------------------------------------------------

metrics_af = ["auc", "f1"]
x = np.arange(len(metrics_af))
width = 0.8 / n_models  # total width ~= 0.8

plt.figure(figsize=(6, 4))
for j, (_, row) in enumerate(df.iterrows()):
    offset = (j - (n_models - 1) / 2) * width
    vals = [row[m] for m in metrics_af]
    plt.bar(x + offset, vals, width, label=row["label"])

plt.xticks(x, [m.upper() for m in metrics_af])
plt.ylabel("Score")
plt.ylim(0.0, 1.0)
plt.title("LightGBM: single-node vs Dask – AUC and F1")
plt.legend()
save_fig("lgbm_single_vs_dask_auc_f1")

# ---------------------------------------------------------------------
# 3. Accuracy / Recall / Precision bar chart
# ---------------------------------------------------------------------

metrics_arp = ["accuracy", "recall", "precision"]
x = np.arange(len(metrics_arp))

plt.figure(figsize=(6, 4))
for j, (_, row) in enumerate(df.iterrows()):
    offset = (j - (n_models - 1) / 2) * width
    vals = [row[m] for m in metrics_arp]
    plt.bar(x + offset, vals, width, label=row["label"])

plt.xticks(x, [m.capitalize() for m in metrics_arp])
plt.ylabel("Score")
plt.ylim(0.0, 1.0)
plt.title("LightGBM: single-node vs Dask – accuracy / recall / precision")
plt.legend()
save_fig("lgbm_single_vs_dask_arp")

# ---------------------------------------------------------------------
# 4. Brier score bar chart (lower is better)
# ---------------------------------------------------------------------

plt.figure(figsize=(4.8, 4))
plt.bar(df["label"], df["brier"])
plt.ylabel("Brier score (lower is better)")
plt.title("LightGBM: single-node vs Dask – Brier score")
save_fig("lgbm_single_vs_dask_brier")

# ---------------------------------------------------------------------
# 5. Training time bar chart
# ---------------------------------------------------------------------

plt.figure(figsize=(4.8, 4))
plt.bar(df["label"], df["train_time"])
plt.ylabel("Training time (s)")
plt.title("LightGBM: single-node vs Dask – training time")
save_fig("lgbm_single_vs_dask_train_time")

# ---------------------------------------------------------------------
# 6. AUC vs training time scatter (quality vs cost)
# ---------------------------------------------------------------------

plt.figure(figsize=(5, 4))
for _, row in df.iterrows():
    plt.scatter(row["train_time"], row["auc"])
    plt.text(row["train_time"], row["auc"], row["label"],
             ha="right", va="bottom")

plt.xlabel("Training time (s)")
plt.ylabel("AUC")
plt.title("LightGBM: single-node vs Dask – AUC vs training time")
save_fig("lgbm_single_vs_dask_auc_vs_time")

# ---------------------------------------------------------------------
# 7. Radar (spider) chart of metric profiles
# ---------------------------------------------------------------------

metrics_radar = ["accuracy", "recall", "precision", "f1", "auc"]
n_metrics = len(metrics_radar)

angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
angles = np.concatenate([angles, [angles[0]]])  # close loop

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, polar=True)

for _, row in df.iterrows():
    vals = [row[m] for m in metrics_radar]
    vals = np.concatenate([vals, [vals[0]]])
    ax.plot(angles, vals, label=row["label"])
    ax.fill(angles, vals, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([m.capitalize() for m in metrics_radar])
ax.set_ylim(0.0, 1.0)
ax.set_title("LightGBM: single-node vs Dask – metric profile")
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
save_fig("lgbm_single_vs_dask_radar")

# ---------------------------------------------------------------------
# 8. Optional: panel combining a few key plots
# ---------------------------------------------------------------------

panel_figs = [
    "lgbm_single_vs_dask_auc_f1.png",
    "lgbm_single_vs_dask_brier.png",
    "lgbm_single_vs_dask_train_time.png",
]
images = [mpimg.imread(FIG_DIR / f) for f in panel_figs]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = ["AUC & F1", "Brier score", "Training time"]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)

plt.tight_layout()
panel_path = FIG_DIR / "lgbm_single_vs_dask_panel.png"
plt.savefig(panel_path, dpi=200)
plt.close()
print("Saved panel figure:", panel_path)

