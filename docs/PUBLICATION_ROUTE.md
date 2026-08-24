# Publication route after locked evaluation

## Recommended paper claim

The strongest defensible paper is a benchmark and mechanism-safety contribution,
not a claim that the preselected mask-axis DRO objective universally wins.

Proposed central question:

> When do measurement-robust clinical tabular models improve probability quality,
> and when should unlabelled label-shift adaptation abstain?

The result supports four connected contributions:

1. A leakage-audited, prediction-reconstructable benchmark spanning hospital,
   prevalence, and measurement-policy shift with 45 heart methods and a large
   patient-disjoint readmission task.
2. A compact observed-feature-set Transformer ablation family that separates
   prior balancing, policy augmentation, policy-axis DRO, joint DRO, and ANE.
3. A mechanism-aware adaptation gate that abstains on every real heart target and
   prevents severe negative transfer from already-fixed MLLS/soft-BBSE scores.
4. Honest cross-task heterogeneity: prior separation is best on the small heart
   benchmark, while joint PS-MaskDRO is best on the independent readmission robust
   proper-score estimand and ANE is negative.

A later, explicitly post-outcome ten-seed stability study adds a fifth mechanism
finding: seed/objective marginalization is more reliable than hard source-only
architecture or router selection. Ten-seed V4 structured-policy ensembling has
the best observed point estimate (0.600458), but its familywise-adjusted
post-hoc interval versus random forest crosses zero. This sensitivity result does
not replace the locked three-seed claim.

## Findings that can be claimed

- V2 prior separation ranks first on the heart primary point estimate and its
  paired contrast with the logistic reference excludes zero.
- The registered V5 mask-axis candidate is directionally better than logistic but
  non-confirmatory; it is worse than V2 in an exploratory paired contrast.
- Joint PS-MaskDRO improves readmission OOD worst-mask balanced log loss over
  pooled ERM and prior separation in exploratory patient-cluster contrasts.
- The real-data adaptation diagnostic rejects all cells, and post-hoc scoring
  shows that deploying the rejected corrections would have caused large loss.
- Strong AUROC does not imply strong probability robustness: pooled models can
  rank well on natural discrimination while losing on worst-policy proper scores.
- The support-aware router failed its improvement-over-strongest-fixed gate;
  equal-logit averaging is the stronger router comparator.
- The exact ten-seed extension reproduces all historical three-seed predictions
  and supports seed/model averaging as a high-value robustness mechanism.

## Claims that must not be made

- prospective cardiovascular risk prediction;
- clinical utility, safety, fairness, or transport to future hospitals;
- unrestricted MNAR robustness;
- universal superiority of PS-MaskDRO or the V5 mask-axis objective;
- successful real-data UDA, because the registered gate abstained;
- a fresh joint preregistration of all final runs, because recovery followed two
  preserved orchestration failures;
- architecture novelty based solely on combining known Transformer, DRO, and
  label-shift components.
- confirmatory or state-of-the-art status for the post-outcome 0.600458 V4 point
  estimate;
- router superiority, because its prespecified robust-improvement gate failed.

## Suggested main paper structure

1. **Motivation:** deployed clinical tables change in hospital case mix,
   prevalence, and which measurements are acquired.
2. **Benchmark contract:** leave-one-hospital-out targets, source-only selection,
   deterministic masks, proper scores, patient-paired uncertainty, and immutable
   prediction evidence.
3. **Method family:** observed-set encoder, prior separation, structured policy
   bank, axis-specific DRO, ANE, and gated prevalence adaptation.
4. **Mechanism validation:** passed label-shift acceptance plus MAR, conditional,
   and MNAR rejection; preserve failed v2 and MNAR/concept limitations.
5. **Heart results:** full 45-method ranking, site heterogeneity, robustness versus
   natural-AUROC trade-off, and non-confirmatory V5-mask result.
6. **Independent readmission results:** patient-disjoint replication and the
   reversal in which joint DRO becomes strongest.
7. **Negative-transfer case study:** zero accepted UDA cells and catastrophic
   research-only corrections that the gate withheld.
8. **Limitations and provenance:** four heart hospitals, domain proxy in
   readmission, mixed-version recovery, multiple comparisons, and no deployment
   claims.

## Essential figures and tables

The current manuscript-ready descriptive bundle is tracked at
`artifacts/figures/heartshift-v5-r2/`, with exact plotted-data CSV files and an
input/output hash manifest. It currently provides the heart robustness/AUROC
scatter, registered-comparison forest plot, site-worst heat map, rejected-
adaptation plot, and readmission worst-mask plot.

- method-by-policy heat map of balanced log loss for each heart hospital;
- robustness-versus-natural-AUROC scatter plot, highlighting V2, V5-mask,
  logistic, random forest, TabPFN, and MIRRAMS;
- paired-bootstrap forest plot for the 16 registered heart comparisons;
- four-panel site-specific worst-policy plot;
- synthetic mechanism acceptance/rejection diagram;
- diagnostic rejection and ungated negative-transfer plot;
- readmission robust-loss table plus paired contrasts against pooled ERM;
- protocol timeline showing v3 and v4 failures and the no-refit v5 recovery.

## Next confirmatory experiment

The heart outer labels have now been consumed. They must not be used for another
round of architecture or hyperparameter selection presented as confirmation. A
new algorithm inspired by these findings should be developed only on source folds
or new developmental data, then preregistered on an untouched external target.

The most promising new hypothesis is **gate-aware risk training**: penalize source
policy representations that make class-conditional mixture compatibility fail,
while retaining an explicit abstention option. This should be tested against
plain V2 and joint PS-MaskDRO on multiple untouched hospital datasets. It is a new
study, not a permissible post-hoc extension of the current confirmatory result.

The new stability evidence suggests a complementary candidate:
**uncertainty-marginalized policy ensembling**. Cross-fitted architecture,
objective, and seed weights should be shrunk toward uniform averaging and should
fall back to uniform weights when leave-one-source-hospital selections are
unstable. It must be developed on source/synthetic or independent-task data and
frozen before external target access.

Before submission, the highest-value addition is a genuinely external,
patient-disjoint multi-hospital dataset with a prespecified endpoint and hospital
identifier. Protected datasets may require data-use agreements and cannot be
silently substituted with admission-source proxies. The current repository is
already suitable for a rigorous benchmark/negative-results manuscript; external
confirmation would strengthen generalizability and any method-level novelty
claim.
