# HeartShift confirmatory research protocol

Protocol version: 0.1.0

Status: development; outer targets locked
Primary method working name: Prior-Separated Site x Mask Distributionally Robust Learning (`PS-MaskDRO`)

## Scientific question

Can a prevalence-separated evidence model trained robustly across hospital and measurement-policy environments improve worst-environment clinical tabular prediction without concealing failures under nonignorable missingness or conditional shift?

The UCI outcome is `num > 0`, an angiographic disease-status endpoint in historical referred cohorts. Results are not estimates of future population risk and are not clinical-device evidence.

## Confirmatory hypotheses

- H1: class-balanced prior separation improves worst-hospital balanced log loss over pooled ERM.
- H2: structured site x mask DRO improves worst-mask balanced log loss and the missingness-degradation curve over random mask augmentation.
- H3: when a label-shift mixture diagnostic is accepted, target-prior adaptation improves calibration and log loss without material ranking change.
- H4: the mixture diagnostic detects simulated conditional-shift and outcome-dependent missingness violations and prevents harmful automatic adaptation.
- H5, initially exploratory: Acquisition-Neutral Evidence centering reduces mask-only shortcut use without more than 0.01 absolute in-distribution AUROC loss.

## Locked evaluation hierarchy

1. Outer leave-one-hospital-out evaluation over Cleveland, Hungary, Switzerland, and VA Long Beach.
2. Inner leave-one-source-hospital-out model selection using only the other three hospitals.
3. All preprocessing, early stopping, hyperparameter choice, calibration, thresholds, and error-auditor fitting occur inside source-only folds.
4. Predictions are stored with immutable `sample_id`, outer site, mask policy, mask replicate, model, seed, and configuration hash.
5. Pooled nested CV is a labelled historical reference, never the headline result.

## Deployment tracks

- `dg_zero_shot`: target data are unavailable during training; report evidence-score ranking, balanced proper scores, prevalence sensitivity, and abstention. Do not claim calibrated target risk.
- `uda_unlabelled`: an unlabelled target batch may estimate prevalence after the label-shift diagnostic. Target labels remain unavailable.
- `few_shot_labelled`: a prespecified target calibration subset is sampled repeatedly; its complement is the disjoint test set.

Results from these tracks are never combined in a single leaderboard.

## Primary estimands

- Macro mean across outer sites of each site's worst-mask balanced log loss.
- Worst hospital x mask balanced log loss.
- Balanced Brier score.
- Area under the performance-versus-missingness curve.
- Area under the risk-coverage curve.

Secondary outcomes are AUROC, AUPRC, calibration intercept/slope where calibration is permitted, balanced accuracy, sensitivity at fixed specificity, and decision-curve net benefit at preregistered thresholds.

Patient-level bootstrap intervals are paired across models. Stochastic mask comparisons use identical patient-mask seeds. Four UCI hospitals do not justify asymptotic claims over a hospital population; site-specific estimates and uncertainty remain visible.

## Structured mask-policy bank

- Natural masks.
- Empirical masks sampled from source hospitals.
- MCAR deletion rates 0.10, 0.30, and 0.50.
- Feature-specific deletion.
- Whole-panel deletion for routine/resting, exercise, and advanced/imaging panels.
- MAR policies driven only by observed core features.
- Outcome-dependent and MNAR policies as explicit failure stress tests, never robustness claims.

Synthetic masks hide only genuinely observed values.

## Required controls and baselines

Controls: majority, prevalence-only, mask-only, hospital-from-mask, core-feature-only, and complete-case sensitivity analyses.

Classical models: elastic-net logistic regression, EBM, random forest, XGBoost, LightGBM, and CatBoost.

Modern models: TabM, RealMLP, FT-Transformer, TabPFN, and TabICL where reproducible implementations and contamination audits permit use.

Robust/missingness methods: site-balanced ERM, GroupDRO, site-only DRO, mask-only DRO, site x mask DRO, CORAL, IRM, CausTab, MIRRAMS-style random masking, and NeuMiss/NeuMISE where implementations are reproducible.

Adaptation baselines: BBSE, calibrated maximum-likelihood label shift, DistPFN for PFN models, intercept/temperature recalibration, and a few-shot audited conformal layer.

## PS-MaskDRO ablations

- V0: identical backbone with pooled ERM.
- V1: site-balanced ERM.
- V2: class-balanced, prior-separated loss.
- V3: MCAR mask augmentation.
- V4: structured policy bank.
- V5-site: smooth entropic hospital-only DRO after averaging policies.
- V5-mask: smooth entropic policy-only DRO after averaging hospitals.
- V5: smooth entropic site x policy DRO.
- V6: target-prior estimator plus mixture diagnostic.
- V7: Acquisition-Neutral Evidence centering.
- V8: optional cross-fitted mask-shift error auditor.

Site grouping, mask grouping, structured masking, prior adaptation, mask-signal access, and backbone choice are also tested factorially.

## Assumptions and falsification

The prior-separated interpretation requires stable class-conditionals and ignorable site-specific missingness. Synthetic experiments independently vary prevalence, MAR measurement policy, conditional covariate shift, concept shift, and outcome-dependent MNAR missingness.

If the target score distribution is incompatible with any mixture of source class-conditional score distributions, automatic prior correction is rejected. The system returns an uncertainty interval or abstains rather than fabricating a calibrated point probability.

## Independent evidence

UCI Heart is the historical case study. A general method claim additionally requires controlled missingness tasks and at least one genuinely multi-hospital dataset such as GOSSIS/eICU. Different endpoints are described as general-method validation, not external heart-disease validation. A clinical heart claim requires a contemporary multi-centre dataset with a compatible endpoint.

## Preregistered kill criteria

- Pivot to a benchmark or negative-results contribution if leakage-safe CatBoost, TabM, or TabPFN matches the candidate on median and worst site x mask outcomes.
- Reject gains that occur only on UCI or vanish under equal tuning and compute.
- Reject a robustness trade-off that loses more than 0.01 absolute in-distribution AUROC without reliable worst-group benefit.
- Remove automatic point-prior adaptation if mixture-fit assumptions fail for most natural targets.
- Describe pure calibration or threshold gains as adaptation, not a new architecture.
- Never claim unrestricted MNAR robustness.

## Acceptance gates

1. Legacy evidence is tagged, hashed, and audited.
2. Official raw files, checksums, deterministic parsing, schema tests, and immutable splits validate in a clean environment.
3. Leakage-safe baselines and row-level prediction artifacts validate before candidate development.
4. Synthetic experiments confirm expected success and failure regimes.
5. Candidate configuration is frozen using source-only evidence.
6. Locked outer UCI evaluation runs once.
7. Independent-dataset evidence and limitations are complete.
8. Tests, lint, manifests, reporting checklists, data/model cards, and aggregate-from-prediction validation pass before release.

## Development-to-confirmation amendment

The single-seed PS-MaskDRO and MIRRAMS grids are explicitly developmental. After
that sweep, but before any multi-seed confirmation or outer-label access, fixed
candidate settings and quantitative source gates were recorded in
`configs/release/freeze_v1.yaml`. The fixed confirmation uses three seeds, common
random initialization within each fold, identical evaluation masks, and no further
hyperparameter search. Because the numerical source thresholds were informed by
developmental source results, they are not represented as an independent
preregistration. The outer targets remain untouched and provide the independent
test of the frozen choices.
