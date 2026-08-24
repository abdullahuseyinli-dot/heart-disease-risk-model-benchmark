from __future__ import annotations

from pathlib import Path

import numpy as np

from heartshift.data.uci import FEATURE_COLUMNS, load_uci_heart
from heartshift.masks import (
    MaskPolicy,
    apply_mask_policy,
    default_policy_bank,
    mask_frame,
    observed_mask_codes,
    observed_mask_hashes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = REPO_ROOT / "data/raw/uci_heart/doi-10.24432-C52P4X/extracted"


def test_every_policy_only_removes_observations_and_is_deterministic() -> None:
    data = load_uci_heart(EXTRACTED).iloc[:80]
    natural = data.loc[:, FEATURE_COLUMNS].notna().to_numpy()
    for policy in default_policy_bank(include_failure_stress=True):
        kwargs = {"empirical_mask_pool": natural} if policy.kind == "empirical" else {}
        first = apply_mask_policy(data, policy, base_seed=17, replicate=2, **kwargs)
        second = apply_mask_policy(data, policy, base_seed=17, replicate=2, **kwargs)
        assert np.array_equal(first, second)
        assert not np.any(first & ~natural)
        assert first.any(axis=1).all()


def test_empirical_policy_intersects_source_and_target_masks() -> None:
    data = load_uci_heart(EXTRACTED).iloc[:20]
    natural = data.loc[:, FEATURE_COLUMNS].notna().to_numpy()
    pool = np.zeros((4, len(FEATURE_COLUMNS)), dtype=bool)
    pool[:, :3] = True
    policy = MaskPolicy("empirical", "empirical")
    shifted = apply_mask_policy(data, policy, base_seed=1, empirical_mask_pool=pool)
    assert np.array_equal(shifted, natural & shifted)
    assert not shifted[:, 3:].any()


def test_mask_frame_preserves_row_order_and_metadata() -> None:
    data = load_uci_heart(EXTRACTED).iloc[:30]
    policy = MaskPolicy("drop_advanced", "panel", panel="advanced")
    observed = apply_mask_policy(data, policy, base_seed=2)
    shifted = mask_frame(data, observed)
    assert shifted["sample_id"].tolist() == data["sample_id"].tolist()
    assert shifted[["ca", "thal"]].isna().all().all()


def test_mask_codes_and_hashes_distinguish_equal_observed_fractions() -> None:
    observed = np.asarray([[True, False, True], [False, True, True]], dtype=bool)
    codes = observed_mask_codes(observed)
    hashes = observed_mask_hashes(observed, ("a", "b", "c"))
    assert codes.tolist() == [5, 6]
    assert len(set(hashes)) == 2
    assert np.array_equal(codes, observed_mask_codes(observed))
    assert np.array_equal(hashes, observed_mask_hashes(observed, ("a", "b", "c")))
