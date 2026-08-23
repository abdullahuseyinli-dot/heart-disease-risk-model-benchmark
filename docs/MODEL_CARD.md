# PS-MaskDRO research model card

Status: experimental research candidate; not a medical device and not approved for
clinical use.

## Intended scientific use

PS-MaskDRO tests whether a predictor can retain useful evidence under simultaneous
hospital, outcome-prevalence, and measurement-policy shifts. The intended use is
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
6. smooth entropic DRO over hospital-by-policy risks, with the two outcome
   classes averaged equally inside each risk; and
7. Acquisition-Neutral Evidence (ANE), subtracting a fold-local balanced-reference
   score evaluated under the same observed-feature set.

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

## Reproducibility and access

Resolved configurations, package versions, GPU identity, Git state, data hashes,
fit histories, prediction-level artifacts, failed runs, selection tables, and the
pre-outer candidate lock are retained. TabPFN checkpoints require the upstream
licence and are not redistributed. Credentials remain outside the repository.
