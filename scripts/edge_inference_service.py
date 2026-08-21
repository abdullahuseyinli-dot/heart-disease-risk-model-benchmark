"""FastAPI service for edge-style latency simulation of the heart disease risk model."""

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------------------------------------------------------
# 0. Load deployment bundle (model + feature names)
# -------------------------------------------------------------------

BUNDLE_PATH = Path("deployment_bundle") / "lgbm_deployment_bundle.joblib"
if not BUNDLE_PATH.exists():
    raise RuntimeError(f"Bundle not found at {BUNDLE_PATH}")

bundle = joblib.load(BUNDLE_PATH)

# Bundle contract: {"pipeline": fitted_pipeline, "feature_names": [...]}.
MODEL = bundle["pipeline"]
FEATURE_NAMES = bundle["feature_names"]

# Optional: force single-thread for nicer latency analysis
try:
    MODEL.set_params(n_jobs=1)
except Exception:
    pass

# SHAP explainer for LightGBM model
try:
    EXPLAINER = shap.TreeExplainer(MODEL)
except Exception:
    EXPLAINER = None

THRESHOLD = 0.30  # decision threshold chosen during evaluation


# -------------------------------------------------------------------
# 1. FastAPI setup
# -------------------------------------------------------------------

app = FastAPI(
    title="Heart Disease Risk Edge API",
    description="Edge inference simulation for LightGBM-based heart disease risk prediction.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    probability: float
    label: int


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float


class PredictionWithSHAPResponse(BaseModel):
    probability: float
    label: int
    top_features: List[SHAPFeature]


# -------------------------------------------------------------------
# 2. Helper: JSON -> one-row DataFrame
# -------------------------------------------------------------------

def _make_dataframe(features: Dict[str, float]) -> pd.DataFrame:
    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features in request: {missing}",
        )

    row = [features[f] for f in FEATURE_NAMES]
    return pd.DataFrame([row], columns=FEATURE_NAMES)


# -------------------------------------------------------------------
# 3. Routes
# -------------------------------------------------------------------

@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictRequest):
    x = _make_dataframe(req.features)
    proba = float(MODEL.predict_proba(x)[0, 1])
    label = int(proba >= THRESHOLD)
    return PredictionResponse(probability=proba, label=label)


@app.post("/predict_with_shap", response_model=PredictionWithSHAPResponse)
def predict_with_shap(req: PredictRequest):
    if EXPLAINER is None:
        raise HTTPException(status_code=501, detail="SHAP explainer is unavailable for the loaded bundle.")
    x = _make_dataframe(req.features)

    proba = float(MODEL.predict_proba(x)[0, 1])
    label = int(proba >= THRESHOLD)

    shap_vals_all = EXPLAINER.shap_values(x)

    # shap can return list (per class) or array depending on version
    if isinstance(shap_vals_all, list):
        shap_vals = np.array(shap_vals_all[1][0])  # class 1, first sample
    else:
        shap_vals = np.array(shap_vals_all[0])

    top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
    top_feats = [
        SHAPFeature(feature=FEATURE_NAMES[i], shap_value=float(shap_vals[i]))
        for i in top_idx
    ]

    return PredictionWithSHAPResponse(
        probability=proba,
        label=label,
        top_features=top_feats,
    )
