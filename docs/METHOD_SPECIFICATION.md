# Prior-Separated Measurement-Policy DRO method specification

This document fixes the mathematical object implemented by the code. It is a
research hypothesis, not a declaration of novelty or clinical validity.

## Observed-feature-set encoder

For patient (i), feature (j) has value (x_{ij}) and observed indicator
(m_{ij}). A continuous value is standardized using source-fold statistics and
embedded with a learned affine map; a categorical value uses a source-fold
vocabulary and embedding. Both are added to a feature-identity embedding. Tokens
with (m_{ij}=0) are excluded by the Transformer attention padding mask. Thus a
counterfactual deletion is applied before representation learning, and a naturally
missing value cannot be recovered from its placeholder.

The ordinary logit is (f_\theta(x_i,m_i)+b). In Acquisition-Neutral Evidence
(ANE) mode it is

\[
e_i=f_\theta(x_i,m_i)-f_\theta(r,m_i)+b,
\]

where (r) is a source-fold-only reference patient: continuous features use the
mean and categorical features use a class-balanced modal category. Both terms use
the identical observed-feature mask. ANE tests whether subtracting a neutral
same-acquisition reference reduces pure mask shortcuts while retaining deviations
from that reference.

## Prior-separated site-by-policy loss

For source hospital (s), measurement policy (p), and outcome (y\in\{0,1\}),
let (L_{s,p,y}) be mean binary cross-entropy. The class-balanced environment risk
is

\[
R_{s,p}=\tfrac12 L_{s,p,0}+\tfrac12 L_{s,p,1}.
\]

Equal outcome weighting removes the empirical source prevalence from the loss; it
does not prove that a learned logit is a likelihood ratio. Source-only balanced
Platt calibration tests that interpretation later.

For (K) observed hospital-policy risks, smooth robust risk is

\[
R_{\mathrm{robust}}=\tau\left[\log\sum_{k=1}^{K}
\exp(R_k/\tau)-\log K\right].
\]

For the protocol-v1 joint candidate, the PS-MaskDRO objective is

\[
\mathcal L=(1-\lambda)\bar R+\lambda R_{\mathrm{robust}}
+\beta\bar B,
\]

where (\bar B) is the mean class-balanced Brier risk. Gradient clipping,
AdamW, early stopping, model dimension, dropout, and every value of
((\lambda,\tau,\beta)) are configuration-controlled.

Protocol v2 selects the mask-axis objective. First average each policy risk over
source hospitals,

\[
R_p=|S|^{-1}\sum_{s\in S}R_{s,p},
\]

then apply smooth DRO only across the measurement-policy axis,

\[
R_{\mathrm{mask}}=\tau\left[\log\sum_{p\in P}
\exp(R_p/\tau)-\log|P|\right].
\]

Its training loss is

\[
\mathcal L_{\mathrm{PS\text{-}MP\text{-}DRO}}
=(1-\lambda)|P|^{-1}\sum_{p\in P}R_p+\lambda R_{\mathrm{mask}}.
\]

This is implemented by `v5_mask_only_dro`. Hospital identity still defines
source folds and balanced cell risks, but it is not a prediction feature and is
not an adversarial axis in the selected robust term.

## Controlled ablations

- V0: pooled ERM on natural masks.
- V1: hospital-balanced natural-mask ERM.
- V2: class-balanced, hospital-balanced natural-mask ERM.
- V3: V2 with natural and MCAR policies.
- V4: V2 with the full structured policy bank and mean risk.
- V5-site: V4 plus smooth DRO after averaging policies within each hospital.
- V5-mask: V4 plus smooth DRO after averaging hospitals within each policy.
- V5-joint: V4 plus smooth DRO over every hospital-by-policy risk.
- V5-joint+Brier: V5-joint plus balanced Brier regularization.
- V7: V5-joint+Brier plus ANE.

MIRRAMS Equation 9 is implemented on the same backbone as an external method
control: natural supervised CE plus weighted randomly masked supervised CE plus a
confidence-gated pseudo-label consistency term. It is not relabelled as a
HeartShift invention.

The registered joint-axis source gate failed because joint DRO was worse than the
best single-axis candidate in the worst site-policy cell. The joint variants are
therefore controls in protocol v2, not the primary method.

## Source-out-of-fold probability calibration

For each outer hospital, baseline model, and seed, let (q_i) be the raw probability
for a source patient predicted when that patient's hospital was excluded from
training. A monotone Platt map is fitted as

\[
\tilde q_i=\sigma(a+\exp(c)\operatorname{logit}(q_i)),
\]

using equal total weight for each source-hospital-by-outcome cell. The positive
slope preserves within-model ranking. The same source-fitted map is applied to
the corresponding frozen outer model's raw scores, and raw and calibrated
outputs are retained as distinct methods. No unlabelled or labelled outer score
is used to estimate (a) or (c).

## Assumption-gated prevalence adaptation

The failed protocol-v2 route sampled source masks from the unlabelled target mask
distribution and used an evidence-only energy-distance diagnostic. Its preserved
synthetic failure showed that this could accept acquisition and conditional shift,
so it is not used by protocol v3.

For protocol v3, source logits are cross-fitted under the same *named*
measurement intervention used for the target evaluation unit. The intervention
is applied separately to source and target natural observations: naturally
missing values are never revealed and target mask patterns are never copied into
source patients. Predictions are averaged over training seeds before diagnostics,
leaving exactly one source and one target observation per patient, named policy,
and policy replicate. The source logits are monotonically calibrated under equal
hospital and outcome weights. If calibrated evidence (e(x)\) approximates

\[
\log p(x\mid y=1)-\log p(x\mid y=0),
\]

then target posterior probability at prevalence (pi_t) is

\[
P_t(y=1\mid x)=\sigma(e(x)+\operatorname{logit}(\pi_t)).
\]

MLLS and soft BBSE estimate (pi_t) independently. Before either is deployed, the
acquisition-aware diagnostic tests four views: calibrated evidence; the
prespecified core variables age, sex, and chest-pain type; the 13-dimensional
observed-feature mask; and their composite. A core value hidden by the evaluated
policy is represented by the out-of-domain token -1 and its observed-mask bit is
zero; the diagnostic therefore never reads a counterfactually hidden value. Each
view uses deterministic random
Fourier features (128 per base view) with a source-median bandwidth. On the
prespecified prior grid, the statistic is the minimum squared distance between the
target mean embedding and a mixture of the two source class-conditional mean
embeddings. A 199-repetition plug-in bootstrap samples at the fitted prior and
re-fits the prior in every repetition.

Automatic adaptation requires every base view and the composite to have
(p\geq0.05). It separately requires mask support: every target feature-state must
occur in source and the 95th percentile of target-to-source nearest-mask Hamming
distance must not exceed 0.25. These are compatibility checks, not proof that
label shift holds. Both prevalence estimates and their adapted probabilities are
always retained in explicitly named `*_research` columns; reportable UDA columns
are populated only when the complete gate accepts. Target labels are loaded only
after all decisions and probabilities are fixed. The registered synthetic-v3
study requires acceptance under pure label shift and rejection under MAR policy
shift, conditional shift, and target-only MNAR; it does not establish unrestricted
MNAR robustness.

## Falsifiable novelty boundary

The contribution under test is an observed-set benchmark and the combination of
prior separation, structured measurement-policy training, mask-axis DRO, and
diagnostic-gated target-prior adaptation. The v1 joint objective and ANE are
preserved negative/ablation evidence. Each component has related prior work. A
positive novelty claim requires a documented literature search, source-only
ablations, synthetic success and rejection regimes, one-time outer evidence, and
independent-task replication. If those fail, the scientifically valid result is
a benchmark or negative-results contribution.
