from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from heartshift.data.uci import FEATURE_COLUMNS, load_uci_heart
from heartshift.models.observed_set import NeuralPreprocessor, ObservedFeatureSetEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = REPO_ROOT / "data/raw/uci_heart/doi-10.24432-C52P4X/extracted"


def _model(preprocessor: NeuralPreprocessor, ane: bool) -> ObservedFeatureSetEncoder:
    return ObservedFeatureSetEncoder(
        n_features=len(FEATURE_COLUMNS),
        continuous_indices=preprocessor.continuous_indices,
        categorical_cardinalities=preprocessor.categorical_cardinalities,
        d_model=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        ane=ane,
    ).eval()


def test_unobserved_values_cannot_change_score() -> None:
    data = load_uci_heart(EXTRACTED).iloc[:40]
    preprocessor = NeuralPreprocessor.fit(data)
    values, observed = preprocessor.transform(data.iloc[:8])
    observed[:, -2:] = False
    changed = values.copy()
    changed[:, -2:] = 999.0
    model = _model(preprocessor, ane=False)
    with torch.no_grad():
        first = model(torch.tensor(values), torch.tensor(observed))
        second = model(torch.tensor(changed), torch.tensor(observed))
    assert torch.allclose(first, second, atol=1e-6)


def test_ane_neutral_reference_has_mask_independent_global_score() -> None:
    data = load_uci_heart(EXTRACTED).iloc[:80]
    preprocessor = NeuralPreprocessor.fit(data)
    _, observed = preprocessor.transform(data.iloc[:7])
    reference = torch.tensor(preprocessor.reference_values, dtype=torch.float32)
    references = reference.unsqueeze(0).expand(len(observed), -1)
    model = _model(preprocessor, ane=True)
    with torch.no_grad():
        score = model(references, torch.tensor(observed), reference)
    assert torch.allclose(score, model.global_bias.expand_as(score), atol=1e-6)


def test_preprocessor_is_finite_and_preserves_natural_mask() -> None:
    data = load_uci_heart(EXTRACTED)
    train = data.loc[data["site"].isin(["cleveland", "hungary"])]
    target = data.loc[data["site"].eq("switzerland")]
    preprocessor = NeuralPreprocessor.fit(train)
    values, observed = preprocessor.transform(target)
    assert np.isfinite(values).all()
    assert np.array_equal(observed, target.loc[:, FEATURE_COLUMNS].notna().to_numpy())
