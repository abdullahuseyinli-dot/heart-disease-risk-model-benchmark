# Claim-to-evidence crosswalk

This table is the manuscript claim boundary. A statement may be promoted only
when the named artifact, evidence class, and limitation accompany it.

| Candidate statement | Evidence | Class | Permitted wording / required limit |
| --- | --- | --- | --- |
| HeartShift evaluates joint hospital and measurement-policy shift. | `docs/BENCHMARK_CARD.md`; `configs/benchmark/classical_outer_v3.yaml`; UCI split manifests | Protocol plus locked execution | Benchmark design on four historical referred cohorts. Do not generalize to future hospitals. |
| V2 prior-separated recorded the lowest locked primary point estimate among 45 reported methods. | `artifacts/reports/heart-outer-v5/primary_estimands.csv`; `manifests/evidence/heart_outer_v5_report_v1.json` | Locked | Use “recorded,” not “is universally best” or “state of the art.” |
| The preselected V5 candidate did not establish superiority over the logistic reference. | Heart v5 paired intervals and `docs/HEART_OUTER_V5_RESULT.md` | Locked negative result | Preserve the prespecified contrast and uncertainty interval. |
| Robust proper-score rankings differed from natural-policy AUROC rankings. | Heart v5 and readmission v3 primary/secondary tables | Locked plus independent-task exploratory evidence | Describe metric trade-off; do not imply improved clinical decisions. |
| The compatibility gate prevented negative transfer by abstaining in all evaluated heart cells. | Heart v5 adaptation summary and independent validation | Locked mechanism/safety result | State that no successful real-data adaptation was demonstrated. |
| Joint PS-MaskDRO improved the readmission worst-mask point estimate relative to pooled ERM. | `artifacts/reports/readmission-outer-v3/`; `manifests/evidence/readmission_outer_v3_report_v1.json` | Independent-task exploratory contrast | Keep the any-readmission endpoint and admission-source domain proxy explicit. |
| Ten-seed analysis favored a structured-policy ensemble by point estimate. | `artifacts/reports/heart-research-development-v1/`; corresponding sidecar | Post-outcome | Familywise intervals crossed zero; not confirmation. |
| The support-aware router did not beat equal-logit blending. | Research-development primary estimands and paired intervals | Post-outcome negative result | Preserve as a negative architectural result. |
| ShiftGuard is a new generally valid shift detector. | No qualifying evidence | Unsupported | Prohibited. Current ShiftGuard evidence is development-only and synthetic failures remain visible. |
| HeartShift predicts prospective cardiovascular risk or is clinically deployable. | No qualifying evidence | Unsupported | Prohibited. Endpoint is historical angiographic disease in referred cohorts; software is not a medical device. |
| PS-MaskDRO is a novel state-of-the-art architecture. | No qualifying evidence | Unsupported | Prohibited without an exhaustive prior-art review and untouched external confirmation. |

## Promotion rule

Before submission, a reviewer must regenerate every numeric statement from the
hash-bound prediction rows, verify the sidecar, confirm its evidence class, and
record the manuscript table/figure location. Narrative files are indexes, not a
replacement for machine-readable evidence.
