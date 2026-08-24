# Heart research development and ten-seed stability result

## Decision

The support-aware router does **not** advance as a superior method. It passed its
natural-AUROC noninferiority and no-collapse checks but failed the required robust
loss improvement over the strongest fixed blend.

The most promising observation is instead a simpler one: averaging the observed
fixed fits lowered the robust-loss point estimate relative to a single fit or a
hard source-only architecture choice. In an explicitly post-outcome exact seed
extension, the ten-seed structured-policy PS-MaskDRO ensemble has the best point
estimate in this repository, but its intervals do not establish superiority and
it is not a new confirmatory result.

## Evidence boundary

All four UCI target outcomes had been consumed by earlier repository versions.
The new runs enforce endpoint-isolated execution, source-only selection, exact
feature-mask pairing, and durable prediction writes, but those mechanics cannot
make the reused outcomes untouched again.

The 26-file development lock at
`artifacts/locks/heart_research_development_v1_pre_outer.json` was written before
the new neural/router outer evaluations. It deliberately says
`consumed_uci_development_freeze_not_preregistration`. The later ten-seed
historical extension was initiated after reviewing consumed outer results and is
labelled post-outcome sensitivity throughout.

## Evidence integrity

The completed evidence packages contain:

- classical controls: 11,205,600 inner and 2,980,800 outer individual-seed
  predictions, with ten seeds and one exact mask for every paired key;
- equal-budget neural study: 480 fits, 3,201,600 inner predictions, 13,920 metric
  cells, and 16 source-only target/experiment selections;
- neural outer study: 1,987,200 individual-seed and 198,720 ensemble predictions,
  with all 16 endpoint-free target/architecture shards written before labels;
- router study: 360 source-validation fits, 4,471,200 endpoint-free member
  predictions, 49,680 aligned base keys, normalized expert weights, 12,000
  conditional-bootstrap rows, and 3,000 exploratory hospital/record-bootstrap
  rows;
- combined report: 794,880 predictions, exactly 16 methods on every one of 49,680
  record/mask keys, 3,456 metric cells, and 24,000 paired-bootstrap rows;
- historical seed extension: 4,968,000 individual predictions across ten frozen
  PS-MaskDRO variants and ten seeds; and
- seed-extension stability report: 1,092,960 predictions, 22 methods per exact
  key, 4,752 metric cells, 100 individual-seed estimands, 42,000 bootstrap rows,
  a four-site jackknife, and multiplicity sensitivity.

Every declared artifact hash in the completed audits matches. The first
1,490,400 rows of the ten-seed historical extension reproduce the old three-seed
run exactly: zero differences in fit seeds, epochs, record hashes, all 13 mask
bits, mask codes, observed fractions, labels, and probabilities.

The neural inner run also has a separately labelled post-run exact-tree audit.
It covers all 488 pre-existing artifacts, including all 480 durable prediction
shards. This strengthens preservation and reconstruction, but its post-run timing
does not turn it into pre-run provenance; the source/configuration bytes used for
outer evaluation are the ones separately captured in the pre-outer lock.

## Equal-budget architecture result

The primary estimand is macro hospital worst-policy balanced log loss; lower is
better. The matched ten-seed report gives:

| Method | Primary BLL | Worst site-policy BLL | Natural macro AUROC |
| --- | ---: | ---: | ---: |
| All four neural backbones/objectives, equal-logit ensemble | **0.615091** | 0.679221 | 0.819401 |
| Three fixed router experts, equal-logit blend | 0.615985 | 0.678704 | **0.822516** |
| V5 backbone ensemble | 0.617555 | 0.689133 | 0.816810 |
| V2 backbone ensemble | 0.617661 | 0.672515 | 0.820432 |
| Support-aware router | 0.619081 | 0.683776 | 0.820563 |
| Natural random forest | 0.625463 | **0.665065** | 0.803536 |
| Natural logistic regression | 0.634591 | 0.688636 | 0.798881 |

The all-backbone ensemble is directionally better than natural random forest by
-0.010372, but its explicitly post-outcome paired interval crosses zero:
percentile 95% interval [-0.030058, 0.009210]. It is therefore promising, not a
confirmed improvement.

Individual neural seeds are substantially worse and more variable than their
ensembles. For example, V2 attention has individual-seed mean 0.672706 (SD
0.022573) versus ensemble 0.625945; V5 attention has mean 0.655118 (SD 0.014607)
versus ensemble 0.620525. Seed ensembling is a core part of the result, not a
cosmetic variance bar.

## Router result

The frozen router gate reports:

- robust improvement over strongest fixed comparator: **failed**; router
  0.619081 versus equal-logit blend 0.615985;
- natural-AUROC noninferiority within 0.01: passed; router 0.820563 versus
  0.822516; and
- no collapse: passed; mean maximum ensemble weight 0.511466, below the 0.95
  collapse threshold.

The equal-logit blend minus router point difference is -0.003096. Its conditional
percentile interval is [-0.006138, 0.000090], while the exploratory four-hospital
hierarchical interval is [-0.008206, 0.001316]. The router cannot be promoted as
an improvement.

The router still learned nontrivial weights. Mean weights for prior-separated,
joint-policy-DRO, and stable-anchor experts were respectively 0.296/0.548/0.156
in Cleveland, 0.202/0.469/0.329 in Hungary, 0.325/0.334/0.342 in Switzerland,
and 0.337/0.404/0.258 in VA Long Beach. Non-collapse alone was therefore not
enough to produce superior target risk.

## Why hard source selection failed

Source validation selected DeepSets over attention in all eight family/hospital
choices, but the target comparison agreed in only three of eight. Across the 16
neural target/architecture cells, source selection score had Pearson correlation
-0.206 and Spearman correlation -0.229 with target worst-policy loss; neither was
significant. These correlations are descriptive and based on only 16 cells.

Router feature-mode selection found the best target mode in only one of four
hospitals. After removing between-target difficulty, the within-target Pearson
correlation between source and target mode scores was 0.201 (p=0.53). The source
mode margins were also very small. This is evidence against hard selection at
the available site/sample count, not evidence that target outcomes should be used
to choose the mode.

## Exact ten-seed historical sensitivity

The exact extension retains the old selected hyperparameters and architecture
names, ensuring the original three effective fit seeds reproduce bit-for-bit.
Its strongest results are:

| Method | 3-seed primary BLL | 10-seed primary BLL | 10-seed worst site-policy BLL | 10-seed natural AUROC |
| --- | ---: | ---: | ---: | ---: |
| V4 structured policy bank | 0.618071 | **0.600458** | **0.664595** | 0.818196 |
| V2 prior separated | **0.602509** | 0.603434 | 0.672357 | 0.816537 |
| V3 MCAR augmentation | 0.615956 | 0.606547 | 0.678903 | 0.805688 |
| V5 joint DRO + Brier | 0.616442 | 0.610812 | 0.697117 | 0.810965 |
| V5 mask-only DRO | 0.625949 | 0.611190 | 0.690469 | 0.813436 |
| Natural random forest | — | 0.625463 | 0.665065 | 0.803536 |

V4's site-worst policy losses are 0.565615 Cleveland, 0.515985 Hungary,
0.655638 Switzerland, and 0.664595 VA Long Beach. Natural random forest is worse
in Cleveland and Hungary, better in Switzerland, and essentially tied in VA Long
Beach. The macro result must not hide that heterogeneity.

The unadjusted post-outcome V4-minus-random-forest contrast is -0.025005,
percentile interval [-0.048843, -0.002578], basic interval
[-0.047431, -0.001167], and descriptive P(better)=0.984. However, the ten-method
family was inspected after outcomes: the Bonferroni percentile interval is
[-0.057290, 0.008648], and the joint max-error interval is
[-0.070867, 0.020857]. Neither supports a familywise superiority claim.

V4 versus ten-seed V2 is -0.002976 with percentile interval
[-0.011859, 0.003728]. The defensible result is that a robust ten-seed PS-MaskDRO
ensemble family is competitive, not that V4 uniquely wins.

V4 ranks first in one and second in the other three leave-one-hospital-out
jackknife rankings among all 22 report methods. Its individual-seed mean is
0.643132 (SD 0.026824; range 0.615934–0.712188), while its ten-seed ensemble is
0.600458. This large ensemble gain is the clearest reproducible mechanism signal.

## Supported conclusion and publication route

The scientifically supported contribution is a leakage-audited hospital and
measurement-policy benchmark with unusually complete prediction evidence, plus
three findings:

1. robust probability performance depends strongly on seed/model averaging;
2. hard source-only architecture/router selection is unreliable with three source
   hospitals; and
3. structured-policy and prior-separated PS-MaskDRO ensembles are promising, but
   the reused four-hospital evidence cannot confirm which objective is superior.

A paper may report the router as a prespecified negative result and the exact
ten-seed extension as post-outcome stability analysis. It must not relabel the
0.600458 point estimate as a preregistered or externally confirmed state of the
art.

## Reproduction entry points

The principal configurations are:

- `configs/research/heart_controls_v1.yaml`;
- `configs/research/observed_backbones_v1.yaml`;
- `configs/research/observed_backbones_outer_v1.yaml`;
- `configs/research/support_router_outer_v3.yaml`;
- `configs/reporting/heart_research_development_v1.yaml`;
- `configs/research/historical_psmask_seed_extension_v1.yaml`; and
- `configs/reporting/historical_psmask_seed_extension_v1.yaml`.

The final reports are under
`artifacts/reports/heart-research-development-v1/` and
`artifacts/reports/historical-psmask-ten-seed-sensitivity-v2-stability/`.
