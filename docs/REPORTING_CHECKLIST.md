# Reporting and bias-audit checklist

This project uses TRIPOD+AI- and PROBAST-AI-informed fields as an internal audit;
completion does not imply formal compliance or low risk of bias.

| Item | Evidence | Status before locked evaluation |
| --- | --- | --- |
| Scientific question and estimand | `docs/RESEARCH_PROTOCOL.md` | Complete |
| Data origin, licence, dates, sites, endpoint | `docs/DATA_CARD.md`, raw `SOURCE.md` files | Complete |
| Participant/record flow and exclusions | canonical profiles and split manifests | Complete |
| Outcome and predictor definitions | parsers, profiles, data card | Complete |
| Missing-data handling | fold-fitted preprocessors and policy bank | Complete |
| Sample-size rationale | all eligible public records; site counts reported | Complete with limitation |
| Leakage controls | nested source-only selection, endpoint-free outer loaders, and candidate lock | Complete in code; final lock pending |
| Model specification and hyperparameters | resolved YAML, selection tables, failed-v1 gate, pivot-v2 record | Heart source runs complete; independent source run pending |
| Comparator tuning parity | source-only grids and prediction artifacts | Heart source comparisons complete; readmission pending |
| Calibration/adaptation separation | raw outputs, source-OOF sensitivity calibration, and assumption-gated UDA | Complete in code; outer evidence pending |
| Discrimination and proper scores | prediction-derived reporting code | Pending outer predictions |
| Site/policy subgroup results | cell-level output contract | Pending outer predictions |
| Uncertainty | paired patient bootstrap; site estimates retained | Complete in code |
| Fairness analysis | sex-stratified descriptive analysis | Pending outer predictions |
| Decision-curve analysis | only for an explicitly calibrated track | Not yet applicable |
| Synthetic success/failure mechanisms | protocol-v2 registered gate and immutable outputs | Pending v2 run |
| Independent evidence | patient-disjoint UCI readmission task | Source test pending |
| Data/code availability | hashes, environment lock, commands | Complete in code; release pending |
| Clinical-use limitations | model card and limitations | Complete |
| Conflicts/funding/author roles | manuscript metadata | Requires author input |
