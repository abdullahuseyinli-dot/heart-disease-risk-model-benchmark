# HeartShift synthetic mechanism protocol v3

Status: registered after the immutable protocol-v2 failure and before the full
protocol-v3 neural run

Date registered: 2026-08-24

## Scope and interpretation

This experiment asks whether unlabelled-target prevalence adaptation is used
only when the observed target distribution is compatible with a mixture of
source class-conditional distributions. It does not validate clinical use,
unrestricted domain adaptation, or unrestricted robustness to MNAR missingness.
It was designed after inspecting the failed v2 evidence and is therefore not an
independent preregistration.

The v2 artifacts and gate remain immutable. In particular, v3 does not relax the
v2 significance level or acceptance thresholds. It removes the invalid
intersection of target masks with already-incomplete source rows and prohibits
repeated counterfactual masks of one source patient from being treated as
independent diagnostic observations.

## Frozen synthetic mechanisms

Every experiment uses source environments e0 and e1 for fitting, source
environment e2 for early stopping, source-only calibration, and diagnostic
reference evidence, and environment e3 as the unlabelled adaptation target.
Environment prevalences are 0.20, 0.40, 0.60, and 0.80. Three seeds are fixed at
6101, 6102, and 6103, with 500 records per environment.

1. `label_only`: the acquisition rule and latent class-conditionals are fixed;
   only prevalence changes. The label-shift null should be accepted.
2. `mar_policy_shift`: the ignorable age/sex-dependent acquisition intercept
   changes by environment. Mask robustness is evaluated, but prevalence
   adaptation should abstain when the multiview null is unsupported.
3. `conditional_only`: acquisition is fixed, while target thalach and chest-pain
   class-conditionals are reversed. Automatic adaptation should be rejected.
4. `mnar_outcome_only`: acquisition is fixed for e0-e2 and gains a direct
   outcome-dependent term only in e3. Automatic adaptation should be rejected.

The legacy `label_mar`, `conditional_shift`, and `mnar_outcome` generators remain
unchanged for exact protocol-v1/v2 reproduction.

## Frozen methods

The full run compares only the source-selected mechanisms needed to answer the
v3 questions:

- `prior_separated` (`v2`);
- `structured_policy_bank` (`v4`);
- `mask_only_dro` (`v5_mask`), the deterministic post-v1 source-only pivot.

Network size, optimizer, early stopping, policy bank, and all remaining training
parameters are fixed in `configs/synthetic/mechanism_v3.yaml`. No synthetic
target label may affect fitting, calibration, prevalence estimation, the
diagnostic, or the automatic deployment decision.

## Acquisition-aware composite label-shift diagnostic

One natural-policy prediction is retained for each held-out source patient and
each target patient. The three views are:

1. monotonically Platt-calibrated prior-free evidence;
2. always-observed age, sex, and chest-pain values;
3. all 13 binary observed-feature indicators.

Each view is standardized using source statistics. A Gaussian-kernel random
Fourier map uses 128 deterministic features and a source-only median bandwidth.
The composite representation concatenates the three maps with equal
`1/sqrt(3)` weighting. For every view and the composite, the statistic is the
minimum squared distance between the target mean embedding and
`(1-p) * source_negative_mean + p * source_positive_mean` over the fixed prior
grid 0.05 through 0.95 in steps of 0.05.

The null distribution uses 199 deterministic plug-in bootstrap repetitions. A
pseudo-target is drawn at the fitted prior, source class references are
resampled, and the prior is re-minimized in every repetition. A view accepts at
`p >= 0.05`; automatic adaptation is allowed only if the evidence, core, mask,
and composite tests all accept and the separate mask-support audit passes. The
support audit requires every target feature state to occur in source and the
95th percentile nearest-source Hamming distance to be at most 0.25. It is a
support guard, not a replacement for the mask-distribution test.

Equal-prior probabilities, always-adapted research probabilities, soft-BBSE
probabilities, oracle-prevalence diagnostic probabilities, and automatically
deployed probabilities are stored separately. Oracle values are computed only
after the unlabelled-target decision is complete.

## Pre-run acceptance gate

The persisted summary CSV, read with round-trip floating-point parsing, must
satisfy all of the following:

1. every required mechanism-by-method cell has all three seeds;
2. the minimum `label_only` diagnostic acceptance rate across
   `prior_separated` and `mask_only_dro` is at least 2/3;
3. their maximum mean `label_only` MLLS prevalence absolute error is at most
   0.12;
4. their maximum mean `label_only` adapted-minus-equal-prior log loss is at most
   0;
5. under `mar_policy_shift`, `mask_only_dro` mean balanced log loss minus
   `structured_policy_bank` mean balanced log loss is at most 0.02;
6. the maximum `mar_policy_shift` diagnostic acceptance rate across the two
   adaptation methods is at most 0.50;
7. the maximum `conditional_only` acceptance rate is at most 0.50;
8. the maximum `mnar_outcome_only` acceptance rate is at most 0.50.

Failure of any check closes the heart outer evaluation for this protocol. The
entire failed run must remain visible. Any subsequent methodological change must
receive a new protocol identifier and must not be described as independently
preregistered.

## Required evidence

The run must preserve its exact configuration and Git state, sample-level source
and target predictions, calibration parameters, fit seeds, training histories,
per-view diagnostic statistics and bootstrap p-values, support-audit values,
cell results, aggregate summary, machine-readable gate, file hashes, and any
failure analysis. All aggregate values must be reproducible from the retained
sample-level predictions.
