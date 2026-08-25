from __future__ import annotations

import json
from pathlib import Path

import pytest

from heartshift.config import load_yaml
from heartshift.data.uci import sha256_file
from heartshift.reporting.publication_figures import build_publication_figures

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.full_evidence
def test_publication_figures_reconstruct_with_hashed_plot_data(tmp_path: Path) -> None:
    config = load_yaml(REPO_ROOT / "configs/reporting/publication_figures_v2.yaml")
    output = tmp_path / "publication-figures"
    paths = build_publication_figures(REPO_ROOT, config, output)
    assert set(paths) == {
        "heart_robustness_vs_auc",
        "heart_registered_bootstrap_forest",
        "heart_site_worst_heatmap",
        "heart_rejected_adaptation",
        "readmission_worst_mask",
        "manifest",
    }
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["status"] == "posthoc_descriptive_publication_figures"
    assert len(manifest["output_sha256"]) == 15
    for relative, expected in manifest["output_sha256"].items():
        assert sha256_file(output / relative) == expected
    assert len(list(output.glob("*.png"))) == 5
    assert len(list(output.glob("*.pdf"))) == 5
