from __future__ import annotations

import numpy as np
import pandas as pd

from heartshift.reporting.hierarchical import (
    hierarchical_hospital_patient_bootstrap,
    holm_adjust,
)


def _predictions() -> pd.DataFrame:
    records = []
    for hospital in range(6):
        for patient in range(20):
            target = patient % 2
            for policy in ("natural", "mcar_50"):
                for method, positive_score in (("reference", 0.7), ("candidate", 0.8)):
                    for training_seed in (1, 2):
                        records.append(
                            {
                                "sample_id": f"h{hospital}-p{patient}",
                                "patient_id": f"h{hospital}-p{patient}",
                                "hospital_id": f"h{hospital}",
                                "target": target,
                                "policy": policy,
                                "mask_replicate": 0,
                                "method": method,
                                "training_seed": training_seed,
                                "score": positive_score if target else 1 - positive_score,
                            }
                        )
    return pd.DataFrame(records)


def test_hierarchical_bootstrap_samples_hospitals_patients_and_seed_layer() -> None:
    replicates, intervals = hierarchical_hospital_patient_bootstrap(
        _predictions(),
        reference_method="reference",
        comparison_methods=["candidate"],
        repetitions=30,
        seed=7,
    )
    assert len(replicates) == 30
    assert set(replicates["sampled_training_seed"]) <= {1, 2}
    assert intervals.loc[0, "percentile_ci_975"] < 0
    assert bool(intervals.loc[0, "training_seed_layer_included"])


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    values = np.asarray([adjusted[name] for name in ("a", "b", "c")])
    assert np.all(np.diff(values) >= 0)
    assert np.all((values >= 0) & (values <= 1))
