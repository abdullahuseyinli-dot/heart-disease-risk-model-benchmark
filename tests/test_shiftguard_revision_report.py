from __future__ import annotations

import numpy as np
import pandas as pd

from heartshift.reporting.shiftguard_revisions import (
    _affected_probability,
    _sample_derived_log_losses,
)


def test_revision_report_rebuilds_episode_log_loss_from_samples() -> None:
    samples = pd.DataFrame(
        {
            "episode_id": ["a", "a", "b", "b"],
            "target": [0, 1, 0, 1],
            "zero_shot_score": [0.2, 0.8, 0.4, 0.6],
            "gated_score": [0.1, 0.9, 0.3, 0.7],
        }
    )
    rebuilt = _sample_derived_log_losses(samples)
    assert len(rebuilt) == 2
    assert np.isfinite(rebuilt.filter(like="log_loss").to_numpy()).all()
    assert rebuilt.loc[0, "gated_log_loss_rebuilt"] < rebuilt.loc[0, "zero_shot_log_loss_rebuilt"]


def test_affected_probability_matches_synthetic_mechanism_contract() -> None:
    assert _affected_probability("covariance_shear", 0.2) == 0.2
    assert _affected_probability("conditional_scale", 0.2) == 0.8
    assert np.isclose(_affected_probability("outcome_dependent_dropout", 0.2), 0.18)
    assert _affected_probability("tail_contamination", 0.2) == 0.15
