# Publication and claim guide

## Appropriate contribution

The repository supports a benchmark and mechanism-safety contribution. It does
not support a universal superiority or clinical-deployment claim for the
preselected mask-axis DRO objective.

A defensible central question is:

> How do hospital and measurement-policy shifts change probability robustness,
> and when should unlabelled prevalence adaptation abstain?

The evidence supports four connected contributions:

1. a source-only, prediction-reconstructable evaluation across hospital,
   prevalence, and measurement-policy environments;
2. an observed-feature-set ablation family separating prior balancing,
   structured deletion, policy-axis DRO, joint DRO, and
   Acquisition-Neutral Evidence;
3. a compatibility gate that abstains on every heart target cell and prevents
   severe degradation from the fixed ungated corrections; and
4. explicit cross-task heterogeneity: V2 prior separation leads the locked
   heart point estimate, whereas joint PS-MaskDRO leads the readmission robust
   proper-score result.

The later ten-seed study adds a post-outcome stability finding. V4
structured-policy averaging records the best observed heart point estimate
(0.600458), while its familywise interval against random forest crosses zero.
This result cannot replace the locked three-seed evidence.

## Findings supported by the repository

- V2 prior-separated training ranks first on the locked heart primary point
  estimate; its registered contrast with the logistic reference excludes zero.
- The preselected V5 mask-axis candidate is directionally better than logistic
  but non-confirmatory, and an exploratory paired contrast favors V2 over V5.
- Joint PS-MaskDRO improves readmission OOD worst-mask balanced log loss over
  pooled ERM and prior separation in exploratory patient-cluster contrasts.
- The heart adaptation diagnostic rejects every cell. Scoring of the already
  fixed research-only probabilities shows that the withheld corrections would
  have caused substantial loss.
- Natural AUROC and measurement-policy proper scores can rank methods
  differently.
- The support-aware router fails its robust-improvement gate; equal-logit
  blending is the stronger router comparator.
- The ten-seed extension exactly reproduces the historical seed bank and shows
  substantial fit-to-fit variability.

## Claims not supported

- prospective cardiovascular risk prediction;
- clinical utility, safety, fairness, or transport to a population of hospitals;
- unrestricted MNAR robustness;
- universal superiority of PS-MaskDRO or the V5 mask-axis objective;
- successful real-data unlabelled adaptation;
- a fresh joint preregistration of all final runs after the two disclosed
  execution recoveries;
- architecture novelty based only on combining established Transformer, DRO,
  calibration, and label-shift components;
- confirmatory or state-of-the-art status for the post-outcome V4 result; or
- router superiority.

## Manuscript structure

1. Motivation: hospital case mix and measurement availability change together.
2. Benchmark contract: nested source-only selection, deterministic masks,
   proper scores, paired record uncertainty, and immutable predictions.
3. Methods: observed-set encoding, prior separation, structured policies, DRO
   axes, and gated prevalence adaptation.
4. Synthetic mechanisms: accepted label shift and rejected observable
   acquisition/conditional shifts, alongside preserved failure regimes.
5. Heart evaluation: method rankings, site heterogeneity, calibration and
   discrimination trade-offs, and the non-confirmatory V5 result.
6. Readmission evaluation: patient-disjoint replication and the reversal in
   which joint DRO performs best on the robust score.
7. Abstention analysis: zero accepted heart cells and the degradation avoided by
   the gate.
8. Stability and negative results: ten-seed averaging, hard-selection
   instability, and failed routing.
9. Limitations and provenance: four heart hospitals, conditional inference,
   admission-source proxy, recovery history, and consumed outcomes.

## Figures and tables

The manuscript-facing figure bundle is
`artifacts/figures/heartshift-v5-r2/`. Every figure has PDF and PNG outputs,
the exact plotted-data CSV, and a manifest binding it to the audited heart and
readmission reports.

Recommended primary displays are:

- method robustness versus natural-policy AUROC;
- registered heart contrast intervals;
- site-specific worst-policy loss;
- adaptation rejection and withheld negative transfer;
- readmission worst-mask comparisons; and
- a concise protocol and recovery timeline.

## Release boundary

All manuscript claims must resolve to a machine-readable report, immutable
prediction set, and evidence status. The four UCI heart outcomes are consumed,
so no additional analysis of those outcomes can be described as an independent
confirmation. The eICU demo is a pipeline smoke test only. No DOI, clinical
validation, or publication acceptance is claimed by the repository metadata.
