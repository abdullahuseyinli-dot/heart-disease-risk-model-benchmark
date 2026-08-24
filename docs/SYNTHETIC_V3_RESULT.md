# HeartShift synthetic mechanism protocol-v3 result

Status: passed registered mechanism gate; independent validation complete

Run commit: `4ce3223a4a055e460869c1f61b04acca9d543c48`

## Outcome

The run completed all 36 registered mechanism-by-method-by-seed cells and wrote
18,000 unique-patient source predictions, 18,000 unique-patient target
predictions, 144 per-view diagnostic records, and complete training histories.
All eight registered checks passed. The exact gate and file hashes are under
`artifacts/runs/synthetic-mechanism-v3`.

The result fixes the two failures exposed by protocol v2 without altering or
reinterpreting the failed v2 record:

- under pure label shift, both registered adaptation methods accepted all three
  seeds, versus the required minimum of two;
- under conditional shift, all methods rejected all three seeds, versus the
  allowed maximum acceptance rate of 0.50.

MAR acquisition shift and target-only MNAR were also rejected in every cell.
All target masks retained local source support, so these rejections came from
distribution incompatibility rather than an unsupported-pattern shortcut.

## Registered gate values

| Check | Observed | Required |
|---|---:|---:|
| Seeds per cell | 3 | at least 3 |
| Label-only minimum diagnostic acceptance | 1.0000 | at least 0.6667 |
| Label-only maximum mean MLLS prevalence error | 0.006960 | at most 0.12 |
| Label-only maximum adapted-minus-equal log loss | -0.023172 | at most 0 |
| MAR MaskDRO-minus-V4 balanced log loss | 0.013700 | at most 0.02 |
| MAR maximum diagnostic acceptance | 0.0000 | at most 0.50 |
| Conditional maximum diagnostic acceptance | 0.0000 | at most 0.50 |
| Target-only MNAR maximum diagnostic acceptance | 0.0000 | at most 0.50 |

For label-only shift, mean MLLS prevalence absolute error was 0.002728 for
MaskDRO and 0.006960 for prior separation. Mean adapted log loss improved by
0.023172 and 0.027723 respectively. Composite p-values ranged from 0.655 to
0.945. These are the intended success-regime results.

For conditional shift, the core, evidence, and composite views reached the
minimum bootstrap p-value of 0.005 in every cell, while the unchanged mask view
did not reject. Under MAR and target-only MNAR, the mask and composite views
reached 0.005 in every cell while the stable core view did not reject. This
view-specific pattern agrees with the mechanisms rather than merely producing a
single unexplained alarm.

## Independent reconstruction

An independent post-run check reloaded the Parquet and round-trip CSV artifacts,
reconstructed every stored metric and summary aggregate from sample-level
predictions, reconstructed all automatic decisions from the three views plus
their composite and mask-support guard, recomputed the gate exactly, rebuilt all
13-bit mask codes, and matched every pre-existing audit hash.

The first calibration-equation check used an unnecessarily strict absolute
tolerance of `1e-12` and stopped. The maximum discrepancy from reconstructing a
calibrated value using CSV coefficients and stored raw float32 logits was
`6.46e-7`; all model metrics already matched exactly. Repeating only that storage
equation check at an explicit `1e-6` tolerance passed. This validator event is
retained in `independent_validation.json` rather than silently omitted.

## Interpretation limits

This is strong controlled-mechanism evidence for an assumption-aware adaptation
guard, not evidence of unrestricted MNAR robustness, clinical deployment, or a
new universally superior architecture. Adapted research probabilities are
retained even in invalid regimes, but automatic deployment probabilities are
missing there because the diagnostic abstained. Heart outer targets remained
locked throughout this run. The next authorization gates are independent
readmission source-only evidence, a clean immutable freeze, and then the
once-only hospital outer evaluation.
