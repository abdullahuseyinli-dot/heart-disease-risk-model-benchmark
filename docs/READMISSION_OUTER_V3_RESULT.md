# Independent readmission outer result

## Evidence status

The patient-disjoint readmission outer run contains 9,771,660 predictions over
54,287 encounters from 39,597 patients, with zero refit/test patient overlap. All
360 metric cells, nine primary estimands, and the 10,000 patient-cluster bootstrap
rows passed independent reconstruction. The endpoint is the TableShift-compatible
any-readmission target; admission source is a domain proxy, not a hospital ID.

## Primary result

Lower worst-mask balanced log loss is better.

| Method | ID natural BLL | OOD natural BLL | OOD worst-mask BLL | OOD worst-mask balanced Brier | OOD natural AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint PS-MaskDRO | 0.655134 | 0.653706 | **0.672257** | **0.239824** | 0.660040 |
| Joint PS-MaskDRO + ANE | **0.654018** | **0.653616** | 0.673223 | 0.240215 | 0.663105 |
| Prior-separated | 0.655243 | 0.656067 | 0.683204 | 0.244796 | 0.654332 |
| Pooled observed-set ERM | 0.687801 | 0.682186 | 0.719939 | 0.259829 | **0.674716** |
| MIRRAMS Equation 9 | 0.717441 | 0.701483 | 0.732008 | 0.265357 | 0.672731 |
| Environment/class-balanced logistic reference | 0.700163 | 0.716891 | 0.857545 | 0.303496 | 0.621875 |

Against the prespecified logistic reference, the registered patient-cluster
bootstrap mean differences are -0.185294 [-0.190676, -0.179941] for PS-MaskDRO
and -0.184328 [-0.189807, -0.178852] for PS-MaskDRO+ANE; every one of 2,000
replicates favors each candidate.

Because all registered candidates share the same bootstrap draws and reference,
exploratory direct paired contrasts can be obtained by subtracting their stored
replicate differences. Joint PS-MaskDRO versus pooled ERM is -0.047685
[-0.049329, -0.046099], and versus prior separation is -0.010937
[-0.011495, -0.010404], with all 2,000 replicates favoring PS-MaskDRO. ANE is
slightly worse than ordinary PS-MaskDRO by +0.000966 [0.000492, 0.001408]; this
ablation does not support ANE.

Pooled ERM has higher OOD natural AUROC than PS-MaskDRO (0.674716 versus
0.660040), while PS-MaskDRO is substantially better on the registered robust
proper-score estimand. Claims must therefore be specific to probability quality
under the policy bank, not generalized to universal predictive superiority.

## Interpretation

The independent task supports joint policy/environment DRO as a measurement-
robust probability objective in this large readmission setting. It does not
rescue the mask-axis DRO heart claim, validate a heart-disease model, or establish
clinical benefit. The two tasks together support a benchmark paper centered on
mechanism-separated evaluation, proper-score robustness, abstention, and honest
cross-dataset heterogeneity.
