# ShiftGuard power-aware compatibility specification

## Scope

ShiftGuard is an unlabelled-target adaptation guard for the restricted case in
which the target differs from the source only through outcome prevalence. It is
not a concept-shift detector and does not certify arbitrary MAR or MNAR changes.
Its safe output is no adaptation.

All fitting and calibration use three disjoint source partitions:

1. diagnostic-representation training;
2. representation validation;
3. pure-label-shift resampling calibration.

After representation selection, the source reference class means use the union
of partitions 1 and 2; partition 3 remains disjoint and is used only for null
resampling.

Target outcomes are prohibited. The current UCI experiments are consumed
mechanism development; only a frozen external hospital split can confirm the
method.

## Diagnostic views

For candidate target prevalence `pi`, each source-fitted diagnostic view `j`
compares its target feature mean with the source class-conditional mixture mean:

`delta_j(pi) = mean_target(phi_j) - [(1-pi) mean_0(phi_j) + pi mean_1(phi_j)]`.

Implemented views include:

- a learned testability representation with a balanced predictive anchor;
- source-standardised linear and second-order polynomial moments;
- Gaussian random Fourier features;
- marginal empirical-CDF indicators and pairwise quantile-copula indicators.

The spectral-ridge statistic is

`D_j(pi) = n * delta_j(pi)^T [Sigma_j(pi) + lambda s_j(pi) I]^-1 delta_j(pi) / d_j`,

where `Sigma_j(pi)` is the source-fitted covariance of the candidate class
mixture and `s_j(pi)` is its average marginal variance. Mean-square and
max-square statistics remain explicit ablations. Every statistic is calibrated
by resampling a held-out source pool under pure label shift.

## Prevalence confidence set

For each view and prior, the critical value is the empirical upper quantile from
held-out pure-label-shift episodes. The Bonferroni omnibus set intersects
view-specific sets while controlling the view family:

`C(B) = intersection_j {pi: D_j(B, pi) <= c_j(pi)}`.

This set serves estimation and uncertainty. Its nominal level is not silently
changed to improve rejection power.

## Separate global compatibility guard

A nonempty confidence set can be too permissive as an automatic-adaptation rule
because the composite null searches across many candidate priors. The global
guard therefore uses a distinct statistic:

`T(B) = min_pi max_j D_j(B, pi) / c_j(pi)`.

Independent source-only resampling episodes—different random draws from those
used for `c_j`—calibrate a batch-size-specific threshold for `T`. The operational
threshold is the declared 90th percentile of the null statistic, matching the
protocol's minimum 90% valid-adaptation acceptance. It does not alter the 95%
prevalence confidence set. Adaptation requires both a nonempty set and a passing
global guard; otherwise all patient-level outputs fall back to zero-shot scores.

Each run retains the guard threshold, number of null episodes, empirical null
pass rate, median and 95th percentile, pre-guard set status, final guard status,
and patient-level predictions.

## Identifiability and power boundary

An unlabelled method cannot detect a target concept reversal constructed with
the same feature distribution as valid label shift. That control must be
accepted and harmful adaptation must remain visible.

Power also depends on how many target rows carry a class-conditional violation.
At prevalence 0.05 and batch size 32, the probability of observing no positive
row is `0.95^32`, and most batches contain fewer than five positive rows. The
benchmark therefore reports acceptance by mechanism, prior, batch size, and
expected affected-row count. Low power is never relabelled as successful
robustness; the defensible operational response is abstention or a larger target
batch.

## Development status

The learned-mean, omnibus-mean, max-moment, spectral-ridge, joint-null, and
quantile-copula revisions are all retained. Iterative synthetic development may
select a candidate for external freezing, but it cannot provide an independent
method claim. Credentialed multi-hospital data and a one-time locked hospital
evaluation remain mandatory.
