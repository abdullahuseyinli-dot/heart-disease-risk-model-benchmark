# Post-outcome HeartShift v5 sensitivity analysis

## Status

This report is conditional sensitivity analysis on already consumed UCI Heart
outer outcomes. It is not a new confirmation experiment and does not support
future-hospital inference. The bootstrap resamples records within each of the
four observed hospital-by-outcome strata; fitted-model uncertainty is not part
of those intervals.

## Paired robust-score contrasts

The estimand is the observed difference in macro hospital worst-policy balanced
log loss relative to site/class-balanced logistic regression. Negative is
better. Two thousand stratified bootstrap replicates were used.

| Method | Observed difference | Percentile 95% interval | Basic 95% interval |
| --- | ---: | ---: | ---: |
| V2 prior-separated ensemble | -0.03208 | [-0.06135, -0.02140] | [-0.04277, -0.00281] |
| V5 joint DRO + Brier | -0.01815 | [-0.05178, -0.00023] | [-0.03607, 0.01548] |
| V5 mask-only DRO | -0.00864 | [-0.04579, 0.01722] | [-0.03451, 0.02851] |
| Site/class-balanced random forest | -0.00756 | [-0.03910, 0.00422] | [-0.01934, 0.02398] |

Only V2 excludes zero under both displayed interval constructions. This remains
conditional on the four observed hospitals and should not be described as a
population-of-hospitals confidence interval.

## Seed-ensemble dependence

V2's individual-seed mean robust score is 0.63856, which is worse than logistic
regression at 0.63459. Its three-seed probability ensemble improves sharply to
0.60251. Random forest changes only from 0.62715 to 0.62703 when ensembled. The
V2 result is therefore an ensemble result, not evidence that an arbitrary single
fit dominates the classical baseline. Equal ensemble budgets and at least ten
development seeds are mandatory for new architecture comparisons.

The V2 benefit is also heterogeneous: it helps Cleveland and Hungary but hurts
Switzerland and VA Long Beach relative to logistic regression. Site-level and
leave-one-site-out tables remain in the evidence package.

Evidence: `artifacts/reports/heart-outer-v5-sensitivity-v1`.
