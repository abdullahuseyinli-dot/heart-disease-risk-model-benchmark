# HeartShift literature and method-positioning matrix

Last updated: 2026-08-24. This is a reproducible design record, not a claim that a
systematic review or exhaustive novelty search has been completed.

| Work | Setting and contribution | HeartShift action | Status / caution |
| --- | --- | --- | --- |
| [TableShift (NeurIPS Datasets and Benchmarks 2023)](https://papers.nips.cc/paper_files/paper/2023/file/a76a757ed479a1e6a5f8134bea492f83-Paper-Datasets_and_Benchmarks.pdf) | Tabular distribution-shift benchmark with fixed ID/OOD splits and strong evidence that ordinary model quality and label shift explain much OOD performance. | Use explicit environments, retain simple and boosted-tree controls, and keep ID quality separate from shift robustness. Plan an independent public TableShift clinical task. | UCI Heart alone cannot support a general tabular-DG claim. |
| [Domain Adaptation under Missingness Shift (AISTATS 2023)](https://proceedings.mlr.press/v206/zhou23b.html) | Formalizes missingness shift and derives adaptation results under restricted mechanisms. | Separate zero-shot DG from target-unlabelled adaptation; expose masks explicitly; match source calibration masks to the target measurement policy. | Its identifying assumptions do not justify unrestricted MNAR claims. |
| [Robust prediction under missingness shifts / NeuMISE (2024 preprint)](https://arxiv.org/abs/2406.16484) | Shows that ignorable and non-ignorable shifts behave differently and proposes a missingness-aware neural architecture. | Include missingness-aware and missingness-discarding ablations; make outcome-dependent missingness a visible failure regime. | Preprint evidence; NeuMISE reproduction remains an optional comparator, not an implemented result. |
| [Domain Adaptation Under MNAR Missingness (2025 preprint)](https://arxiv.org/abs/2504.00322) | Reduces MNAR adaptation to imputation under stated assumptions and extends to simultaneous covariate shift. | Treat this as a specialized UDA comparator for a future dataset with the required variables and assumptions. | Not interchangeable with a generic neural robustness claim; not yet implemented. |
| [Dual knowledge-guided data augmentation for robust clinical prediction (Scientific Reports 2026)](https://www.nature.com/articles/s41598-026-46459-0) | Combines clinically constrained value augmentation with group-based masking to improve robustness under structured tabular clinical-data imperfections. | Retain explicit clinical panel-loss policies and compare structured versus uniform masking under identical source splits. | A single-source augmentation result does not substitute for genuinely independent hospitals or validate synthetic clinical plausibility here. |
| [MIRRAMS (2025 preprint, v2)](https://arxiv.org/pdf/2507.08280) | Uses natural-label CE, additionally masked-label CE, and confidence-gated consistency (Equation 9) to address unseen missingness shifts. | Implement Equation 9 on the identical observed-set Transformer backbone, with the paper defaults `(r, lambda1, lambda2, tau)=(0.2,15,15,0.95)` plus source-only tuning of learning rate and `r`. | Must be labelled as a reproduction on a different backbone. Random masking alone is not called MIRRAMS. |
| [Group DRO (ICLR 2020)](https://openreview.net/pdf?id=ryxGuJrFvS) | Optimizes worst predefined-group loss and emphasizes the need for regularization and early stopping. | Compare pooled ERM, site balancing, prior separation, structured masking, and smooth site-by-mask DRO under identical regularization and source-only early stopping. | Four UCI sites are too few for population-level claims about hospitals. |
| [TabPFN v2 (Nature 2025)](https://www.nature.com/articles/s41586-024-08328-6) | In-context tabular foundation model for small-to-medium tables, supporting missing values. | Run leakage-safe TabPFN v2 and separately versioned v3 baselines using the same source-only outer/inner design. | Foundation-model contamination cannot be ruled out merely from strong performance; versions and checkpoint access are recorded. |
| [TabM (ICLR 2025)](https://openreview.net/pdf?id=Sd4wYYOhmY) | Parameter-efficient ensembling of MLP-like tabular networks with strong results on drift benchmarks. | Run the official PyTabKit TabM-D estimator under the identical nested hospital protocol and counterfactual mask bank. | It is a comparator, not folded into the candidate architecture; internal validation remains source-only. |
| [PyTabKit / RealMLP (NeurIPS 2024)](https://github.com/dholzmueller/pytabkit) | Reproducible tabular benchmark library and strong RealMLP defaults. | Pin PyTabKit and run RealMLP-TD-S plus its FT-Transformer implementation as modern trained-from-scratch controls. | Models that do not accept sample weights use the explicitly labelled pooled track. |
| [TabICL (ICML 2025)](https://openreview.net/pdf?id=0VvD1PmNzM) and [TabICLv2 (2026 preprint)](https://arxiv.org/abs/2602.11139) | Open tabular in-context foundation models; v2 changes the synthetic prior and architecture. | Pin the exact `tabicl-classifier-v2-20260212.ckpt` and tune only ensemble size on source folds. | V1/v2 and TabPFN results are never conflated; possible pretraining overlap remains a limitation. |
| [TabPFN-2.5 report (2025 preprint)](https://arxiv.org/abs/2511.08667) | Extends TabPFN scale and introduces distillation. | Record as a possible scale comparator for independent larger tasks. | It is not silently substituted for v2 or v3. |
| [DistPFN (2026 preprint)](https://arxiv.org/abs/2605.04363) | Test-time posterior adjustment for label shift in tabular in-context learning. | Add as a separately labelled TabPFN adaptation comparator if its released implementation can be pinned and audited. | Newer than the initial design; not yet an implemented result. |
| [Black-box shift estimation (ICML 2018)](https://proceedings.mlr.press/v80/lipton18a.html) | Estimates target class proportions from a black-box predictor under label shift. | Implement soft BBSE as a target-unlabelled prior-estimation baseline. | Correction is valid only when class-conditionals are sufficiently stable and the soft confusion matrix is identifiable. |
| [Calibrated maximum-likelihood label shift (ICML 2020)](https://proceedings.mlr.press/v119/alexandari20a.html) | Shows that maximum likelihood plus appropriate calibration is a strong label-shift baseline. | Cross-fit balanced monotone Platt calibration on source hospitals, estimate target prevalence by MLLS, and compare with soft BBSE. | Calibration never uses outer labels; automatic correction is disabled when the score-mixture diagnostic rejects compatibility. |
| [Testable Learning with Distribution Shift (COLT 2024)](https://proceedings.mlr.press/v247/klivans24a.html) | Formalizes learners that may reject an unverifiable target while accepting an in-scope target with high probability, and proves limits on what such tests can certify. | Treat no-adaptation as a first-class output and retain an unidentifiable concept-shift control. | Its theoretical setting does not itself validate the clinical or finite-sample ShiftGuard construction. |
| [Generalized kernel two-sample tests (Biometrika 2024)](https://academic.oup.com/biomet/article/111/3/755/7420214) | Shows that ordinary MMD components can cancel under variance alternatives and combines centered within/between-sample components using their covariance. | Motivates reporting mechanism-specific power and testing covariance-aware feature discrepancies instead of relying on one mean norm. | The published permutation test is not claimed as reproduced by ShiftGuard. |
| [Minimax Optimal Kernel Two-Sample Tests with Random Features (2025 preprint)](https://arxiv.org/abs/2502.20755) | Develops covariance/spectral-regularized RFF two-sample statistics, adaptive kernel/regularization selection, and a permutation implementation with power guarantees. | Add a finite-feature source-fitted ridge precision statistic as a clearly labelled development analogue, then calibrate every candidate prior on disjoint pure-label-shift episodes. | ShiftGuard v4 is an inspired finite-feature composite-null construction, not an implementation or theorem transfer from this paper. |
| [DUAL: Learning Diverse Kernels for Aggregated Two-sample and Independence Testing (2025 preprint)](https://arxiv.org/abs/2510.11140) | Learns individually powerful but non-redundant kernels and uses selection inference when aggregating them. | Reserve diverse-kernel selection for a future frozen external-development split; retain Bonferroni intersection for the current auditable source-only revision. | Adding target-selected kernels now would invalidate the current source-only protocol. |
| [Tabular Prior-data Fitted Networks under temporal drift (2024 preprint)](https://arxiv.org/abs/2411.10634) | Changes the PFN prior to represent temporal drift. | Motivates treating drift priors as a future architecture-level comparator rather than assuming ordinary TabPFN is shift-robust. | Temporal shift differs from hospital and measurement-policy shift. |

## Candidate contribution under test

The working candidate is **Prior-Separated Site-by-Mask Distributionally Robust
Learning (PS-MaskDRO)**. It combines five testable design choices:

1. An observed-feature-set Transformer that never receives hospital identity and
   cannot access a hidden feature value.
2. Equal weighting of each source-hospital/outcome cell so the learned logit is
   encouraged to encode evidence rather than the source prevalence.
3. A prespecified bank of natural, MCAR, MAR, empirical, and whole-panel
   measurement interventions that only delete observed information.
4. Smooth entropic DRO over hospital-by-measurement-policy risks, with equal
   weighting of the two outcome-class losses within each risk.
5. Optional Acquisition-Neutral Evidence (ANE), which subtracts the prediction
   for a fold-local class-balanced reference patient under the *same* observed mask.

The UDA layer is deliberately separate: source out-of-fold scores are recalculated
under masks sampled from the unlabelled target policy, calibrated as equal-prior
evidence, and passed to MLLS and soft BBSE. An energy-distance mixture diagnostic
must accept at least one source class-conditional mixture before any adapted point
probability is emitted.

The combination and ANE construction are research hypotheses, not established
novelty. A publishable claim requires (a) source-only ablations, (b) controlled
success and failure simulations, (c) a genuinely untouched external evaluation,
(d) an independent clinical shift task, and (e) a broader prior-art search before
the manuscript uses the word "novel."

## New support-aware stacking hypothesis

The current development branch also tests a prediction-level support-aware
shrinkage router. It mixes prior-separated neural evidence, joint-policy-DRO
evidence, and a stable classical anchor in logit space. Its simplex weights are
conditioned on source mask support, pattern rarity, expert disagreement, and one
of three source-selected context ablations: no expert context, all expert logits,
or anchor-relative logits. Training adds site-by-policy robust risk, regret to the
best fixed expert, and a natural-policy non-inferiority penalty.

This is an architectural research hypothesis, not an established novelty claim.
The required controls are each fixed expert, equal logit averaging,
source-selected best-expert selection, and source-selected convex stacking. The
router must beat the strongest fixed comparator rather than merely a convenient
reference. The four UCI outcomes can now provide only post-outcome development
evidence; method confirmation requires frozen external hospitals.
