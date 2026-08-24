from __future__ import annotations

from pathlib import Path

import numpy as np

from heartshift.data.uci import FEATURE_COLUMNS, load_uci_heart
from heartshift.evaluation.research_controls import augment_training_policies

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = REPO_ROOT / "data/raw/uci_heart/doi-10.24432-C52P4X/extracted"


def test_structured_classical_augmentation_never_reveals_features() -> None:
    data = load_uci_heart(EXTRACTED).iloc[:100]
    augmented = augment_training_policies(
        data,
        augmentation="structured_policy_bank",
        training_replicates=2,
        seed=9,
    )
    assert set(augmented["training_policy"]) == {
        "natural",
        "mcar_10",
        "mcar_30",
        "mcar_50",
        "drop_routine",
        "drop_exercise",
        "drop_advanced",
        "mar_age_sex_30",
        "empirical_source",
    }
    natural = data.loc[:, FEATURE_COLUMNS].notna().to_numpy()
    for _, group in augmented.groupby(["training_policy", "training_mask_replicate"]):
        observed = group.loc[:, FEATURE_COLUMNS].notna().to_numpy()
        assert not np.any(observed & ~natural)
