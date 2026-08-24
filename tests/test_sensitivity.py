from __future__ import annotations

import pandas as pd

from heartshift.reporting.outer_report import cell_metrics
from heartshift.reporting.sensitivity import (
    leave_one_site_out_ranking,
    site_policy_class_losses,
    site_worst_contrasts,
)


def _predictions() -> pd.DataFrame:
    records = []
    for site in ("a", "b", "c"):
        for row in range(8):
            target = row % 2
            for policy in ("natural", "drop"):
                for method, confidence in (("reference", 0.7), ("candidate", 0.8)):
                    records.append(
                        {
                            "sample_id": f"{site}-{row}",
                            "outer_target": site,
                            "target": target,
                            "policy": policy,
                            "mask_replicate": 0,
                            "observed_fraction": 1.0 if policy == "natural" else 0.5,
                            "method": method,
                            "track": "dg_zero_shot",
                            "score": confidence if target else 1.0 - confidence,
                        }
                    )
    return pd.DataFrame(records)


def test_sensitivity_tables_keep_sites_and_class_components_visible() -> None:
    predictions = _predictions()
    metrics = cell_metrics(predictions)
    contrasts = site_worst_contrasts(
        metrics, reference_method="reference", comparison_methods=["candidate"]
    )
    jackknife = leave_one_site_out_ranking(metrics)
    classes = site_policy_class_losses(predictions)
    assert len(contrasts) == 3
    assert contrasts["difference_candidate_minus_reference"].lt(0).all()
    assert set(jackknife["omitted_site"]) == {"a", "b", "c"}
    assert classes[["class_0_log_loss", "class_1_log_loss"]].notna().all().all()
