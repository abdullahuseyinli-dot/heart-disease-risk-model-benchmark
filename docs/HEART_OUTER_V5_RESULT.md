# Heart outer v5 result

## Evidence status

The mixed-provenance final report combines immutable v3 classical/modern
predictions, no-refit v5 finalization of the fixed v4 MIRRAMS shards, and the
once-run v5 PS-MaskDRO predictions. Both MIRRAMS failures remain preserved. This
sequence is not represented as a fresh joint preregistration; see the v3/v4 and
v4/v5 recovery records.

Both neural runs pass independent reconstruction with zero maximum difference in
canonical labels, three-seed ensembles, and metrics. The report then passed a
second complete deterministic replay, including all 2,000 paired-bootstrap
replicates:

- 2,235,600 normalized sample-level predictions;
- 9,720 site-policy metric cells;
- 45 primary method estimands;
- 360 descriptive sex-stratum rows;
- 32,000 bootstrap rows for 16 prespecified comparisons;
- zero maximum difference for every reconstructed table, replicate, and interval.

The machine-readable record is
`artifacts/reports/heart-outer-v5/independent_validation.json`.

The publication figure bundle is under
`artifacts/figures/heartshift-v5-r2/`. Every figure has a PDF, PNG, and exact
plotted-data CSV; `figure_manifest.json` binds those outputs to the audited heart,
PS-MaskDRO, and readmission evidence. These are post-hoc descriptive figures and
do not alter the registered analyses.

## Primary robust-probability result

The primary estimand is the mean across the four held-out hospitals of each
hospital's worst measurement-policy balanced log loss. Lower is better. Bootstrap
intervals are paired over patients within each observed hospital and are
conditional on these four hospitals; `P(better)` is the fraction of replicates
with candidate-minus-logistic difference below zero, not a frequentist p-value.

| Method | Primary BLL | Worst site-policy BLL | Natural macro AUROC | Bootstrap mean difference vs logistic [95% interval] | P(better) |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 prior-separated | **0.602509** | 0.693879 | 0.818322 | **-0.040378 [-0.061661, -0.019375]** | 1.0000 |
| V3 MCAR augmentation | 0.615956 | 0.671808 | 0.803983 | not prespecified | — |
| V5 joint DRO + Brier | 0.616442 | 0.697024 | 0.812055 | -0.025633 [-0.054501, 0.000946] | 0.9685 |
| V4 structured policy mean | 0.618071 | 0.676198 | 0.806451 | -0.024840 [-0.049031, -0.000697] | 0.9790 |
| V5 mask-axis DRO (preselected candidate) | 0.625949 | 0.699207 | 0.801486 | -0.014664 [-0.046858, 0.020709] | 0.8065 |
| Random forest | 0.627031 | **0.660195** | 0.804629 | -0.013824 [-0.040897, 0.004482] | 0.9080 |
| Site/class-balanced logistic reference | 0.634591 | 0.688636 | 0.798881 | reference | — |
| MIRRAMS Equation 9 | 0.686618 | 0.731403 | 0.793852 | +0.048310 [0.008696, 0.085842] | 0.0090 |

V2 is first of 45 methods on the primary point estimate. Its largest site-policy
loss occurs in VA Long Beach under `drop_exercise` (0.693879); its site worst
losses are 0.540320 Cleveland, 0.527482 Hungary, 0.648354 Switzerland, and
0.693879 VA Long Beach. This heterogeneity must remain visible.

The preselected V5 mask-axis DRO candidate is directionally better than logistic,
but its paired interval crosses zero. A post-hoc contrast derivable from the same
paired replicates favors V2 over V5-mask by -0.025714
[-0.055335, -0.002025], with bootstrap probability 0.9865. Accordingly, the
heart data do **not** confirm that mask-axis DRO improves on prior separation.
V0 pooled ERM has the highest natural macro AUROC (0.828615) but a materially
worse primary robust loss (0.647779), illustrating the registered discrimination
versus robustness/calibration trade-off.

## Adaptation abstention result

The acquisition-aware diagnostic accepted 0 of 432 evaluated adaptable
method-site-policy-replicate cells. The deployable report therefore contains only
the zero-shot DG track and no automatic UDA probabilities. This is an intended
abstention outcome, not missing result data.

After endpoints were opened, an explicitly post-hoc safety analysis scored the
research-only probabilities that had already been fixed before endpoint access.
For V2, macro site-worst balanced log loss was 0.602509 zero-shot, 0.610084 after
equal-prior calibration, 1.687476 with ungated soft-BBSE, and 1.929936 with
ungated MLLS. For V5-mask the corresponding values were 0.625949, 0.626195,
1.879558, and 2.014169. The gate therefore prevented severe degradation in this
experiment. This label-informed analysis is explanatory and must not be promoted
to a deployable UDA comparison.

## Scientific conclusion

The supported heart finding is a professional hospital/measurement-shift
benchmark plus a strong prior-separated observed-set model and an effective
adaptation-abstention mechanism. The registered new mask-axis DRO objective is a
negative/non-confirmatory heart result, not a superiority claim. MIRRAMS is also
inferior to the logistic reference on the primary contrast. Independent
readmission evidence is reported separately and is essential to any broader
measurement-robustness argument.

These data are four small historical referred cohorts with angiographic disease
status, not prospective cardiovascular risk. The results do not establish
clinical validity, deployment safety, or inference over future hospitals.
