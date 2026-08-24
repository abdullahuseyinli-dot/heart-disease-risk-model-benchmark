# ShiftGuard v1 development and external-confirmation protocol

## Status and claim boundary

This protocol begins only after the locked HeartShift v5 outcomes were opened.
All four UCI Heart hospitals and the UCI readmission split are consumed evidence.
They may be used for debugging, descriptive sensitivity analysis, mechanism
development, and explicit hypothesis generation. They cannot confirm ShiftGuard,
select its final architecture, or supply a new superiority claim.

The historical heart endpoint remains angiographic disease status (`num > 0`) in
referred cohorts. It is not prospective cardiovascular risk. The external ICU
endpoint is in-hospital mortality and validates a general method, not a heart
disease model.

The study separates:

- `dg_zero_shot`: no target data are used during model fitting or routing;
- `uda_unlabelled`: unlabelled target batches may be used by a frozen diagnostic;
- `few_shot_labelled`: any labelled-target calibration is a separate future track.

No result moves between leaderboards.

## Current evidence retained as hypothesis generation

The v5 V2 ensemble ranks first on the four-site macro site-worst balanced log
loss, but its advantage is driven mainly by Cleveland, depends on three-seed
averaging, and does not win the absolute worst cell. The preselected V5-mask
hypothesis is non-confirmatory. Joint policy DRO is stronger on the larger
readmission task. The real-data adaptation gate abstains everywhere and prevents
severe post-hoc adaptation harm, but supplies no positive real-data coverage.

These observations motivate, but do not test, the two new components below.

## Hypotheses

### H1: support-aware shrinkage

A source-only router over fixed prior-separated, robust, and stable-anchor experts
reduces hospital/policy/class robust loss by shrinking high-variance evidence
where mask support is weak, without more than 0.01 absolute natural AUROC loss.

### H2: testability-trained adaptation

A diagnostic representation trained to accept pure label-shift episodes and
reject observable class-conditional/acquisition shifts yields more safe-adaptation
coverage than the fixed v3 gate without increasing harmful adaptation.

### H3: set-valued selective action

Inverting the source-calibrated compatibility test into a prevalence set and
patient posterior interval improves risk-coverage behaviour relative to point
MLLS/BBSE and always-adapt controls.

No hypothesis asserts detection of arbitrary concept shift or unrestricted MNAR
robustness. A concept change with unchanged unlabelled feature distribution is
not identifiable by this diagnostic and remains a visible failure control.

## Methods

### Support-aware shrinkage predictor

The fixed experts are:

1. prior-separated observed-set evidence;
2. joint measurement-policy DRO evidence;
3. a stable classical or TabM anchor selected under an equal compute budget.

The router receives only the mask bitset, observed fraction, nearest source-mask
distance, exact support, source pattern rarity, expert disagreement, prediction
entropy, and explicitly declared source-selected expert-context ablations.
Hospital identity and target outcomes are prohibited inputs.

For expert probabilities `p_ij`, the prediction is

`p_i = sigmoid(sum_j alpha_j(z_i) * logit(p_ij))`,

where `alpha` is a simplex-valued MLP. Source OOF training minimizes a smooth
site x policy x class robust loss, robust regret versus the best fixed expert, and
a natural-policy non-inferiority penalty. DG routing uses patient-level and frozen
source summaries only. Any target-batch routing is a separate UDA ablation.

An audit found that the proposed three-hospital source meta-cycle did not provide
a fully untouched third hospital: although each evaluated row was predicted by a
base learner that excluded that row's hospital, base learners producing router
training rows could have used the nominal meta-evaluation hospital. The v1/v2
source-meta design was therefore superseded before a full run and its smoke
artifacts are retained as code diagnostics only.

The corrected v3 development route selects backbones, anchors, fixed blends, and
router feature mode from source-OOF evidence only. Three router checkpoints rotate
the source early-stopping hospital, and their target predictions are ensembled.
Every endpoint-free outer probability is persisted before target labels are
loaded. This is clean with respect to the current fit but remains post-outcome,
descriptive evidence because earlier project versions consumed all four UCI outer
outcomes. Complete mask codes and all feature-level mask bits must match before
experts can be joined. Only authorized, preregistered external hospitals can
provide new confirmation.

### ShiftGuard diagnostic

The diagnostic representation concatenates calibrated evidence, declared stable
core values, the full observed mask, and an optional detached predictive
representation. Source-only pseudo-target episodes include:

- valid class-prevalence changes with fixed class-conditionals;
- class-conditional translations and scales;
- acquisition/mask changes, including outcome-dependent deletion;
- support translations and withheld invalid-mechanism families.

For source class means `mu_0`, `mu_1`, target mean `mu_B`, and candidate prior
`pi`, the compatibility discrepancy is

`D(B, pi) = ||mu_B - ((1-pi)mu_0 + pi mu_1)||^2`.

Training minimizes valid-shift discrepancy and prevalence error, applies a margin
against observable invalid shifts, retains a balanced predictive anchor, and
penalizes representation collapse through whitening. Critical values are learned
only from held-out source pure-label-shift episodes.

The inverted set is

`C_alpha(B) = {pi: D(B, pi) <= c_alpha(pi)}`.

If the set is empty or a support guard fails, adaptation is rejected. Otherwise
the system reports posterior bounds over all accepted priors. A positive or
negative action is emitted only when the complete posterior interval lies on one
side of the declared threshold; all other cases abstain.

The v1 synthetic falsification gate showed that a learned first-moment view can
accept conditional scale and covariance changes. The versioned revision adds
source-standardized second moments and a Gaussian random-Fourier kernel view.
The omnibus set is the intersection of view-specific prevalence sets, with the
familywise valid-shift error controlled by Bonferroni calibration. The failed v1
evidence remains part of the record and the revision uses a new protocol version.

Subsequent iterative development remains non-confirmatory. V2, v3, and v4 reduce
observable-invalid acceptance from 54.80% to 27.42%, 21.26%, and 15.97%,
respectively. V7 adds quantile-copula indicators and a separately calibrated
global minimum-over-priors compatibility guard, reducing the full 10-seed rate
to 11.21% while retaining 96.8% valid acceptance. V7 still fails the 5% gate and
reaches only 68.8% valid selective coverage, so it does not advance. See the
[power-aware specification](SHIFTGUARD_POWER_GUARD_SPEC.md) and
[development result](SHIFTGUARD_DEVELOPMENT_RESULT.md).

## Data sequence

1. UCI Heart and readmission: mechanism development and descriptive replication
   only; no new confirmation.
2. Public eICU demo: schema, split, and execution smoke tests only.
3. Credentialed raw GOSSIS-1-eICU/eICU: primary multi-hospital development and
   locked confirmation using genuine hospital and patient identifiers.
4. MIMIC-IV or AmsterdamUMCdb: harmonized external/temporal confirmation for the
   same first-24-hour in-hospital mortality contract.
5. A future contemporary multi-centre CAD cohort is required for a clinical heart
   claim.

External hospitals are assigned by a salted identifier hash to 60% development,
20% architecture selection, and 20% locked confirmation before outcome analysis.
Patients crossing roles are quarantined rather than moved. Prespecified minimum
hospital and class counts determine estimability; failures remain in manifests.

## Measurement-policy families

- natural hospital-native missingness;
- demographics, vitals, laboratory, history, medication, and advanced-test panel
  loss where available;
- delayed feature availability at fixed clinical time windows;
- MCAR 10/30/50 curves;
- age/demographic MAR;
- empirical source-hospital interface masks;
- outcome-dependent/MNAR falsification policies, never robustness claims;
- leave-one-policy-family-out evaluation for unseen mechanisms.

Every prediction stores the complete schema-bound mask hash. Policies may hide
only naturally observed values.

## Comparator hierarchy

Primary comparators receive equal tuning and ensemble budgets:

- balanced logistic, random forest, CatBoost, LightGBM/XGBoost;
- structured-mask-augmented classical models;
- V2 prior separation, V4 structured augmentation, joint policy DRO;
- a non-attention DeepSets/MLP observed-set encoder;
- NeuMiss/NeuMISE, MIRRAMS, and missingness-avoiding trees where reproducible;
- GroupDRO/REx, CORAL, IRM, and CausTab as DG controls;
- TabM, RealMLP, FT-Transformer, released TabPFN-3, TabICLv2, and DistPFN where
  licences and resources permit;
- BBSE, calibrated MLLS, fixed RFF/DFM-style diagnostics;
- ordinary, missingness-conditional, and weighted conformal/selective baselines.

Unavailable or irreproducible methods remain named omissions, not silently
discarded comparisons.

## Estimands and inference

The primary prediction estimand is macro across locked hospitals of each
hospital's worst real policy balanced log loss, explicitly described as an
equal-prior evidence score. The absolute worst hospital-policy cell is key
secondary. Natural AUROC, AUPRC, ordinary log loss, Brier, calibration, and
risk-coverage are mandatory co-outcomes.

Inference reports the observed paired contrast. A hierarchical bootstrap samples
hospitals and then patients within hospital; a refit/seed layer estimates training
variation. Policy-family and leave-one-hospital sensitivities remain separate.
One primary candidate/comparator contrast is tested; secondary families use Holm
control. No patient bootstrap is described as future-hospital inference.

Prevalence adaptation is assessed with ordinary population log loss, Brier
score, calibration, and selective utility. Balanced log loss deliberately fixes
equal class weight and is therefore not an appropriate success criterion for a
target-prevalence correction; it remains a mandatory prediction-robustness
co-outcome and any degradation is reported.

## Acceptance gates

The predictor advances only if all are satisfied on architecture-selection
hospitals:

1. robust BLL is better than V2 and the strongest classical comparator;
2. natural AUROC loss is no greater than 0.01;
3. the absolute worst result does not regress materially;
4. improvement survives leave-one-hospital and unseen-policy-family analyses;
5. gains persist under equal ensembles and at least ten development seeds;
6. routing does not merely collapse to one expert unless that fixed expert itself
   satisfies every superiority condition.

The diagnostic advances only if:

1. valid pure-label-shift acceptance is at least 90% at nominal alpha 0.05;
2. false acceptance under held-out observable invalid mechanisms is at most 5%,
   with uncertainty reported;
3. harmful adapted-minus-zero-shot loss is not increased;
4. safe-adaptation coverage exceeds the fixed v3 gate;
5. selective risk materially improves at 80% and 90% coverage;
6. performance survives target batch sizes 32, 64, 128, 256, and 512.

The study stops or reports a negative result if a fixed convex blend/ordinary
stacking matches the router, gains vanish externally, AUROC loses more than 0.01,
the method abstains above 90% without useful risk reduction, intervals are
uninformative for more than 80% of patients, or the result depends on consumed
UCI targets.

## Execution gates

1. Validate data contracts, splits, patient/hospital isolation, and raw hashes.
2. Complete fully nested source OOF predictions for every equal-budget baseline.
3. Fit and ablate the support-aware router using source OOF predictions only.
4. Train ShiftGuard using source pseudo-target episodes with held-out mechanisms.
5. Run synthetic success and falsification gates.
6. Freeze source code, configurations, package versions, seeds, selected experts,
   masks, thresholds, and confirmation hospital IDs.
7. Run the locked confirmation once.
8. Reconstruct all aggregates from sample-level predictions and preserve every
   pass, failure, quarantine, and abstention.
