# Reporting and bias-audit checklist

This checklist follows the structure of TRIPOD+AI and PROBAST+AI for internal
reporting review. It is not a declaration of formal compliance or low risk of
bias.

| Item | Repository evidence | Current assessment |
| --- | --- | --- |
| Scientific question and estimand | `docs/RESEARCH_PROTOCOL.md` | Defined |
| Data origin, license, dates, sites, and endpoint | `docs/DATA_CARD.md`, raw source records, third-party notices | Complete |
| Record flow and exclusions | canonical profiles and split manifests | Complete |
| Outcome and predictor definitions | parsers, profiles, data card | Complete |
| Missing-data handling | fold-fitted preprocessors and policy bank | Complete |
| Sample-size rationale | all eligible public records; site counts reported | Complete with small-site limitation |
| Leakage controls | nested source-only selection, endpoint-isolated loaders, candidate locks | Implemented and tested |
| Model specification and hyperparameters | resolved configurations, selection tables, gate and pivot records | Complete for reported methods |
| Comparator tuning parity | source-only grids, compute records, and prediction artifacts | Reported; unavailable methods remain explicit omissions |
| Calibration and adaptation separation | raw outputs, source-OOF calibration, gated unlabelled adaptation | Complete |
| Discrimination and proper scores | prediction-derived report tables | Complete |
| Site and policy results | cell-level outputs and site-worst summaries | Complete |
| Uncertainty | paired record bootstrap; site estimates retained | Complete, conditional on observed sites and fitted models |
| Recorded-sex analysis | descriptive stratum tables | Available; not a comprehensive fairness analysis |
| Decision-curve analysis | restricted to probability tracks with an applicable calibration interpretation | Not used for the headline claim |
| Synthetic mechanism tests | preserved protocol-v2 failure and registered protocol-v3 result | Complete |
| Independent evidence | patient-disjoint UCI readmission task | Complete |
| Data and code availability | hashes, lockfile, commands, LFS inventory | Available; no DOI release claimed |
| Clinical-use limitations | model card, benchmark card, and limitations | Complete |
| Conflicts, funding, and author roles | manuscript declarations | Not recorded in the software repository |

The heart outcomes are consumed, the ten-seed analysis is post-outcome, and the
readmission direct PS-MaskDRO-versus-ERM contrast is exploratory. Those labels
must accompany the corresponding result in any manuscript or derivative table.
