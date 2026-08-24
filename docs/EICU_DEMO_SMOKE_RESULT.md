# Public eICU demo execution smoke result

## Claim boundary

This is an end-to-end software and data-contract smoke test, not scientific
evaluation. The public eICU demo yielded only 134 held-out target encounters and
is a deliberately nonrepresentative subset. Its result file sets
`scientific_result=false` and `confirmation_claim_allowed=false`. It must not be
used to rank methods, estimate multi-hospital generalization, or support a
clinical claim.

## Completed checks

- Downloaded raw demo files are retained under `data/raw/eicu-crd-demo/2.0.1`
  with source hashes.
- The preparation contract retained 1,044 active encounters, quarantined 794,
  and assigned 97 eligible hospital identifiers deterministically to 57
  development, 26 architecture-selection, and 14 locked-demo roles.
- The model smoke ran logistic regression, random forest, observed-set
  attention, and observed-set DeepSets under natural, MCAR 30/50, and complete
  vital/laboratory panel-loss policies.
- Every model-policy prediction is retained at sample level and the report
  artifacts have a complete SHA-256 audit.

## Descriptive smoke values

The natural-policy target prevalence was 0.1642 (`n=134`). Natural AUROC values
were 0.7082 for logistic regression, 0.7853 for random forest, 0.7192 for
attention, and 0.6981 for DeepSets. These numbers establish only that the models,
metrics, masking policies, and endpoint pipeline execute on the external schema.
They are too small and too selected for method conclusions.

Authorised full GOSSIS/eICU data remain required for the frozen multi-hospital
development and one-time locked confirmation described in the ShiftGuard
protocol.

Evidence: `artifacts/runs/eicu-demo-model-smoke-v1` and
`data/processed/eicu-demo-shiftguard-smoke-v1`.
