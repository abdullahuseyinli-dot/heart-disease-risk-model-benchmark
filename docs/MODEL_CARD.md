# Prior-Separated Measurement-Policy DRO research model card

Status: evaluated experimental research family; not a medical device and not
approved for clinical use.

## Intended scientific use

PS-MP-DRO tests whether a predictor can retain useful evidence under hospital,
outcome-prevalence, and measurement-policy shifts by optimizing robustness over
the policy axis. The intended use is
benchmark research, mechanism falsification, and comparison with strong tabular
baselines. It must not be used to diagnose a person, allocate treatment, or claim
prospective risk.

## Architecture and objective

The backbone represents each observed feature-value pair as a token and applies a
small Transformer to the set. Hidden values are removed by an attention padding
mask before encoding. Hospital identity is used only to define training groups.

The ablation path is:

1. pooled empirical risk minimization;
2. source-hospital balancing;
3. hospital-by-outcome balancing to encourage equal-prior evidence;
4. random measurement deletion;
5. a structured natural/MCAR/MAR/panel/empirical policy bank;
6. smooth entropic DRO over policy risks after averaging source hospitals, with
   the two outcome classes averaged equally inside each hospital-policy risk.

Joint hospital-by-policy DRO and Acquisition-Neutral Evidence are retained as
prespecified controls. The joint candidate failed its registered source gate;
that negative result is part of the model card rather than being hidden.

The separated adaptation layer calibrates source out-of-fold evidence, estimates
an unlabelled target prevalence with MLLS or soft BBSE, and emits adapted
probabilities only when an energy-distance class-conditional-mixture diagnostic
accepts the label-shift approximation. A rejected diagnostic produces no adapted
point probability.

## Evaluation contract

- Hyperparameters and epoch counts are selected on source hospitals only.
- Outer predictions use a frozen commit, frozen configurations, and fixed seeds.
- Every method sees the same patient-level stochastic masks.
- Zero-shot domain generalization, unlabelled-target adaptation, and any future
  labelled-target calibration are reported separately.
- Primary claims use balanced proper scores and worst environment/policy outcomes;
  AUROC is secondary.
- Patient bootstrap intervals are paired within each observed hospital. They do
  not imply sampling inference over a population of hospitals.

## Known failure modes

Label-shift correction is invalid under material class-conditional or concept
shift. Outcome-dependent MNAR missingness can change both evidence and mask signal
in ways this method cannot identify from unlabelled data. ANE may remove useful
acquisition information. Small historical cohorts make neural comparisons noisy.
The adaptation diagnostic has finite power and is an abstention guard, not proof
that label shift holds.

The protocol-v2 candidate was chosen with a deterministic rule after observing
the protocol-v1 source-confirmation failure. Its source metrics are therefore
selection evidence. Source-OOF Platt calibration may also fail to transport under
hospital or missingness shift; uncalibrated probabilities remain reportable.

## Locked evaluation outcome

On the four heart cohorts, V2 prior separation ranks first on the primary robust
balanced-log-loss point estimate (0.602509) and its paired comparison with the
logistic reference excludes zero. The preselected V5 mask-axis DRO candidate is
directionally better than logistic but its interval crosses zero, and a post-hoc
paired contrast favors V2 over V5-mask. The heart experiment therefore does not
confirm the new mask-axis DRO objective.

On the independent patient-disjoint readmission task, joint PS-MaskDRO ranks first
on OOD worst-mask balanced log loss (0.672257) and exploratory paired contrasts
favor it over pooled ERM and prior separation. ANE is slightly but consistently
worse than ordinary joint PS-MaskDRO.

The heart adaptation diagnostic accepted no target cell. Post-hoc scoring of the
pre-fixed research-only MLLS/soft-BBSE outputs showed severe degradation, so
abstention was protective. These mixed results support a benchmark and
mechanism-gating contribution; they do not support universal superiority or
clinical use. Exact tables are in `docs/HEART_OUTER_V5_RESULT.md` and
`docs/READMISSION_OUTER_V3_RESULT.md`.

## Reproducibility and access

Resolved configurations, package versions, GPU identity, Git state, data hashes,
fit histories, prediction-level artifacts, failed runs, selection tables, and the
pre-outer candidate lock are retained. TabPFN checkpoints require the upstream
licence and are not redistributed. Credentials remain outside the repository.
