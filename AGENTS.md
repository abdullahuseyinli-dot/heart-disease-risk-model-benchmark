# HeartShift Repository Operating Rules

- Preserve the commit tagged `legacy-v1-development-consumed`; its metrics are historical, non-confirmatory evidence.
- The endpoint is angiographically defined disease status (`num > 0`) in historical referred cohorts. Do not call it prospective population risk.
- Hospital/source identity is required for splitting and auditing. It must never be silently dropped or supplied as a disease-prediction feature.
- Outer leave-one-hospital-out targets are locked. Preprocessing, model selection, early stopping, calibration, threshold choice, and error-auditor training must use source data only.
- Keep zero-shot domain generalization, unlabelled-target adaptation, and few-shot labelled-target adaptation as separate experiments and result tables.
- Preserve raw archives, extracted source files, checksums, manifests, split manifests, configurations, prediction-level outputs, failed runs, and legacy artifacts.
- Do not invent results or imply clinical deployment validity. UCI-only evidence cannot establish a clinically deployable model or a general new architecture.
- Synthetic MNAR and concept-shift failures must remain visible. Never claim unrestricted MNAR robustness.
- Every reported aggregate must be reproducible from immutable sample-level predictions joined by `sample_id`.
- Run data validation, leakage checks, unit tests, lint, configuration validation, and evidence validation before commits or tags.
- Do not open a locked outer target during method selection. Final outer evaluations are run once from a frozen configuration.
- Do not use destructive cleanup or remove legacy/raw evidence.
