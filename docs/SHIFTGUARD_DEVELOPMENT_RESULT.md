# ShiftGuard iterative development result

## Decision

ShiftGuard does **not** advance to external confirmation in its current form.
The best full revision, v7, preserves valid label-shift adaptation and prevents
aggregate adaptation harm, but fails the predeclared maximum 5% observable-invalid
acceptance gate and does not reach 80% selective coverage. This is retained as a
research negative result, not converted into a pass.

All runs in this document are source-only synthetic mechanism development after
UCI outcomes were consumed. They cannot establish an independent method claim.

## Reconstructed revision comparison

Ordinary population log loss was rebuilt from patient-level predictions for
every version. Balanced log loss was not used as the adaptation-success metric.

| Revision | Valid acceptance | Observable-invalid acceptance | Valid gated minus zero-shot log loss | Invalid gated minus zero-shot log loss |
| --- | ---: | ---: | ---: | ---: |
| v1 learned mean | 99.9% | 54.80% | -0.09482 | -0.04944 |
| v2 omnibus mean | 99.8% | 27.42% | -0.09368 | -0.02680 |
| v3 max moment | 99.6% | 21.26% | -0.09175 | -0.02120 |
| v4 spectral ridge | 99.7% | 15.97% | -0.09186 | -0.02013 |
| v7 quantile-copula + power guard | 96.8% | 11.21% | -0.08491 | -0.01629 |

V7 reduces invalid acceptance by 79.5% relative to v1, but its observed 11.21%
rate has a 95% Wilson interval of 10.50% to 11.97% and is unequivocally above the
5% gate. Its valid acceptance is 96.8% (95% Wilson interval 95.52% to 97.72%).

The unidentifiable concept-reversal control is accepted 96.8% of the time and
increases ordinary log loss by 0.67916 when adapted. This is the expected
identifiability failure: its unlabelled feature law is deliberately identical to
valid label shift. No target-unlabelled method can use this experiment to claim
arbitrary concept-shift detection.

## V7 mechanism and batch-size result

| Mechanism | n=32 | n=64 | n=128 | n=256 | n=512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conditional scale | 29.5% | 18.0% | 18.5% | 15.5% | 13.0% |
| Conditional translation | 18.0% | 15.5% | 10.0% | 11.5% | 8.5% |
| Covariance shear | 35.5% | 33.0% | 21.5% | 22.0% | 17.0% |
| Nonlinear bimodal | 13.0% | 9.0% | 7.0% | 7.5% | 7.5% |
| Outcome-dependent dropout | 18.5% | 12.0% | 11.5% | 7.5% | 6.0% |
| Support translation | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Tail contamination | 6.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Covariance and conditional scale remain the main weaknesses. V7's mean selective
coverage under valid label shift is 68.8%, so the protocol's 80% and 90%
risk-coverage analysis points cannot be claimed.

## Pilot optimism and stability

The two-seed v7 pilot produced 5.71% invalid acceptance, but the 10-seed result
was 11.21%. On the eight seeds not used in that pilot, mechanism-specific rates
remain materially similar to the full result. This demonstrates why the new
protocol requires at least ten development seeds and why pilot values must not
be promoted.

The joint-null v5 pilot was rejected immediately because invalid acceptance rose
to 40.0%. The v6 quantile-copula pilot reached 10.71% but had a small harmful
adaptation delta of +0.00073. Both failures remain in the artifact tree.

## Power analysis

For every mechanism/prior/batch cell, the audit derives the expected number of
rows actually affected by the synthetic violation and the exact binomial
probability that fewer than five are affected. In v7:

- invalid acceptance is 41.7% in cells with fewer than five expected affected
  rows;
- it is 4.47% in cells with at least twenty expected affected rows;
- Spearman correlation between expected affected-row count and acceptance is
  -0.546.

This explains, but does not waive, the failed aggregate gate. The appropriate
research response is a larger target batch or power-aware abstention, not a
robustness claim in underpowered cells.

## Next scientific route

1. Do not tune further on these same mechanisms.
2. Treat v7 as a frozen hypothesis only after an architecture-selection dataset
   with genuine independent hospitals is available.
3. Add a labelled audit subset or substantially larger target batches for rare
   outcomes; unlabelled detection is weak when few rows carry the violation.
4. Compare against permutation spectral-MMD, classifier two-sample, and
   missingness-conditional diagnostics on external development hospitals.
5. Advance only if all safety, harm, and selective-coverage gates pass on a new
   mechanism family and then once on locked hospitals.

Evidence: `artifacts/reports/shiftguard-revisions-v3-final-development`,
`artifacts/runs/shiftguard-synthetic-development-v7-power-guard`, and the
versioned v1-v6 run directories.
