# Outer v3 failure and v4 neural recovery

## Scope and disclosure

The v3 candidate was frozen at Git commit
`e787c199b59ef77b839f22c8415375b83f573074`. The classical, modern-v2,
modern-2026, TabPFN-v3, and independent readmission outer evaluations completed
under that lock. Their sample-level predictions and independent reconstruction
audits are preserved unchanged.

The first MIRRAMS v3 shard failed before any held-out heart endpoint was loaded.
`_adapt_predictions` attempted to filter an empty source-calibration frame by
`evaluation_policy` before testing whether the method was adaptable. The resulting
`KeyError: evaluation_policy` is preserved in
`artifacts/runs/mirrams-outer-v3/failure.json`. No MIRRAMS v3 prediction or metric
result exists. PS-MaskDRO v3 was deliberately not launched because its first
non-adaptable ablation would encounter the same defect.

The completed v3 baseline results were known when this recovery was prepared.
Accordingly, v4 is not represented as a new joint preregistration of those already
completed runs. It is a mechanical recovery registration for the two unopened
neural evaluations, while cryptographically binding the completed and failed v3
evidence so it cannot be changed or hidden.

## Permitted recovery change

The only execution change is:

- when `adaptable` is false, `_adapt_predictions` no longer reads the empty
  source-calibration frame;
- when `adaptable` is true, the function now explicitly requires the two policy
  keys and a matching, non-empty source-calibration group.

The regression test
`test_non_adaptable_outer_does_not_require_source_predictions` verifies the
non-adaptable branch returns zero-shot predictions, `not_applicable` adaptation
status, no adaptation scores, and empty diagnostic tables.

The recovery does **not** change model equations, features, selected
hyperparameters, fixed epochs, training seeds, mask policies, mask replicates,
adaptable variants, diagnostic settings, acceptance threshold, bootstrap plan, or
report comparisons. No completed outer metric was used to modify either neural
method.

## Versioned execution

- MIRRAMS: `configs/benchmark/mirrams_outer_v4.yaml` ->
  `artifacts/runs/mirrams-outer-v4`
- PS-MaskDRO: `configs/benchmark/psmask_outer_v4.yaml` ->
  `artifacts/runs/psmask-outer-v4`
- recovery lock: `artifacts/locks/heartshift_candidate_v4_neural_recovery.json`
- combined report: `configs/reporting/heart_outer_v4.yaml`

Each neural run is executed once after a clean committed freeze. The v4 report
combines immutable v3 baseline predictions with the newly locked v4 neural
predictions. This mixed-version provenance must remain visible in manuscripts and
tables.

## Interpretation boundary

This experiment evaluates historical angiographic disease status in four referred
UCI cohorts under hospital and simulated measurement-policy shifts. It is not a
prospective cardiovascular-risk model, and UCI-only results cannot establish
clinical deployment validity. Synthetic MNAR and concept-shift failures remain
part of the evidence and prohibit claims of unrestricted missingness robustness.
