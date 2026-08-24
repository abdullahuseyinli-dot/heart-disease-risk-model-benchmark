from __future__ import annotations

import numpy as np
import pandas as pd

from heartshift.evaluation.external_smoke import _mask_generic_frame


def test_generic_mask_frame_hides_values_without_revealing_missingness() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [2, 3]})
    observed = np.asarray([[True, False], [False, True]])
    masked = _mask_generic_frame(frame, ("a", "b"), observed)
    assert masked.loc[0, "a"] == 1.0
    assert pd.isna(masked.loc[0, "b"])
    assert pd.isna(masked.loc[1, "a"])
    assert masked.loc[1, "b"] == 3
