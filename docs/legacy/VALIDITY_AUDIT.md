# Legacy benchmark validity audit

Audit date: 2026-08-23

Audited commit: `551d4706b02538e9a096b4a4b5b544f0484091a5`
Preservation tag: `legacy-v1-development-consumed`

## Status

All metrics at the audited commit are retained as historical development evidence. They are not confirmatory estimates and must not be compared directly with locked HeartShift outer-hospital results.

The task is classification of angiographically defined disease status in four historical referred cohorts. It is not prospective population risk prediction.

## Immutable legacy hashes

| Artifact | SHA-256 |
| --- | --- |
| `data/heart_disease_processed.parquet` | `19A53EDE51FB66C1EB9317A8CD88DA9BE4440BE0729ADF2C6C8CFEF22EA92399` |
| `scripts/heart_disease_model_benchmark.py` | `5C1CBDC7CFFBFB05D49B01C701D16CF1A63C63E9302842FBE5AC4DD9C7C271B0` |
| `results/main/metrics/test/holdout_models.csv` | `9BE303E7CBF92A8D5116FBF798009E146CFC7F99A3EAC4E97E94D1328209B38F` |

## Confirmed validity failures

1. Hospital provenance is assigned to `groups_dataset` at lines 183--185 of the legacy script but is not carried into the model table. The tracked processed parquet has 920 rows and 28 columns but no `dataset` field.
2. Missing categorical values are converted to the string `"nan"` because `astype(str)` precedes `fillna` at line 209.
3. Imputation, missing indicators, and one-hot encoding are performed before the split at lines 183--226.
4. Optuna searches on the complete `X, y` table at lines 549--580 before the holdout is created at lines 587--589.
5. TabNet uses the final holdout as its early-stopping set at lines 666--670 and subsequently reports metrics on that same holdout.
6. The same holdout is reused for calibration, thresholds, subgroup analyses, SHAP, a surrogate tree, counterfactual examples, and repeated learning-curve diagnostics.
7. Two exact duplicate rows exist in the tracked processed table; the legacy random split places an exact duplicate pair on opposite sides of the split.
8. Dask constructs partitions from a non-monotonic test index at line 879 and compares computed predictions with `y_test.values` at line 902. Dask sorts the frame index, so rows and labels are misaligned. The reported approximately 0.533 AUROC is an alignment artifact.
9. The repository validator checks static artifact consistency but does not train models, validate split lineage, or detect leakage.

## Consequences

- The reported TabNet AUROC of approximately 0.9302 is development-consumed and optimistically biased.
- The Dask result is not evidence that distributed LightGBM performs poorly; it is a row-alignment defect.
- The random pooled split permits hospital prevalence and measurement patterns to act as shortcuts.
- All HeartShift results must start from official raw files, preserve hospital identity, and use nested source-only leave-one-hospital-out evaluation.

No legacy artifact is deleted or overwritten by the HeartShift study.
