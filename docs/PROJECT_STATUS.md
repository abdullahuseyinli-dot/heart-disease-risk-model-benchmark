# Project status

Status snapshot: 2026-08-24.

Earlier freezes, failure records, manifests, and reports remain immutable. This
page summarizes them; it does not replace their machine-readable status fields.

| Area | Status | Primary evidence |
| --- | --- | --- |
| Archived v1 benchmark | Preserved and independently checked | `results/`, `docs/legacy/`, tag `legacy-v1-development-consumed` |
| UCI Heart canonical data | Complete | `data/processed/uci_heart_canonical_v1.parquet`, profile and split manifests |
| Source-only method selection | Complete | source prediction artifacts and selection records under `artifacts/runs/` |
| Locked heart evaluation | Complete with two disclosed mechanical recoveries | `artifacts/reports/heart-outer-v5/` |
| Independent readmission task | Complete | `artifacts/reports/readmission-outer-v3/` |
| Ten-seed and routing study | Complete; post-outcome sensitivity | `artifacts/reports/heart-research-development-v1/` and `historical-psmask-ten-seed-sensitivity-v2-stability/` |
| ShiftGuard study | Closed as a negative result | `artifacts/reports/shiftguard-revisions-v3-final-development/` |
| eICU demo | Pipeline smoke passed; no scientific evaluation | `data/processed/eicu-demo-shiftguard-smoke-v1/` |
| Public software release | Source repository available; no immutable DOI release | `CITATION.cff`, `.zenodo.json`, release documentation |

## Main heart outcome

The locked report contains 2,235,600 normalized predictions, 9,720
site-policy metric cells, 45 method estimands, and 32,000 paired-bootstrap rows.
Independent reconstruction found zero differences in labels, three-seed
ensembles, metrics, bootstrap replicates, and intervals.

V2 prior-separated training has the lowest locked primary point estimate
(0.602509). The preselected V5 mask-axis DRO candidate records 0.625949 and its
registered interval against the logistic reference crosses zero. The heart
experiment therefore does not confirm mask-axis DRO superiority.

## Post-outcome stability outcome

The exact ten-seed extension reproduces every historical three-seed prediction.
V4 structured-policy averaging records the lowest later point estimate
(0.600458), but multiplicity-adjusted intervals versus random forest cross zero.
The source-selected support-aware router fails its robust-improvement gate.

These results support a stability and model-selection finding, not a new locked
heart claim.

## Independent-task outcome

The readmission report contains 9,771,660 predictions over 54,287 evaluated
encounters from 39,597 patients. Joint PS-MaskDRO improves the exploratory
worst-mask balanced-log-loss contrast against pooled ERM while pooled ERM retains
higher AUROC. Admission source is a domain proxy rather than a hospital
identifier.

## Interpretation boundary

The effective heart sample remains 920 records across four hospitals; repeated
policies, methods, and seeds are paired evaluations rather than new patients.
All intervals are conditional on the observed sites and analysis choices.
Clinical validity and transport to future hospitals remain unestablished.
