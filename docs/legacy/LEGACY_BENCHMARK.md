# Archived v1 benchmark

This page records the original Heart Disease model benchmark. It is retained for
provenance and reproducibility, but it is separate from the current HeartShift
hospital-shift protocol. Its stratified holdout was repeatedly inspected during
development and cannot provide confirmatory evidence.

## Saved holdout results

The values below are read from
`results/main/metrics/test/holdout_models.csv`.

| Model | Accuracy | F1 | ROC-AUC | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.8370 | 0.8585 | 0.9083 | 0.1183 |
| LightGBM | 0.8424 | 0.8638 | 0.8962 | 0.1192 |
| XGBoost | 0.8587 | 0.8774 | 0.9010 | 0.1151 |
| TabNet | 0.8533 | 0.8744 | 0.9302 | 0.1066 |

| Model | ROC-AUC 95% interval | F1 95% interval |
| --- | ---: | ---: |
| Logistic regression | [0.8633, 0.9495] | [0.8041, 0.9065] |
| LightGBM | [0.8450, 0.9406] | [0.8098, 0.9083] |
| XGBoost | [0.8497, 0.9447] | [0.8235, 0.9202] |
| TabNet | [0.8876, 0.9645] | [0.8203, 0.9148] |

The intervals overlap, so the archived result does not establish a single
superior model. The associated figure remains at
`assets/holdout_performance_intervals.png`.

## Archived design

- 920 processed records and 28 columns.
- Binary endpoint derived from `num > 0`.
- Stratified 80/20 holdout with seed 42.
- Saved cross-validation summaries for the fitted model families.
- Brier decomposition, calibration curves, and threshold analysis.
- 1,000 holdout bootstrap resamples.
- SHAP, a shallow surrogate tree, and descriptive recorded-sex subgroups.

The saved systems checks include a local FastAPI latency benchmark and a Dask
LightGBM run. The Dask output has a documented row-label alignment defect and is
retained as failed execution evidence, not as a distributed-model comparison.
The latency values describe one recorded machine and workload rather than a
deployment service-level objective.

The archived pipeline tuned some model choices before the final holdout split.
Its saved holdout and bootstrap values are therefore development evidence, not
an unbiased estimate of a locked model-selection procedure.

## Archived reproduction

The original scripts and dependency files remain at their historical paths:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/heart_disease_model_benchmark.py
```

Full regeneration requires the original root-level input expected by the
archived script. The current HeartShift workflow instead uses the versioned raw
archive, canonical parser, nested hospital splits, and locked `uv` environment
documented in the current [installation and verification guide](../USAGE.md).

The archived validity assessment is [VALIDITY_AUDIT.md](VALIDITY_AUDIT.md).
