# Third-party notices

The repository's MIT License covers the original source code and documentation.
It does not replace the licences of the datasets described below. Dataset
licences, attribution requirements, and notices continue to apply to raw data,
adapted databases, and derived data files distributed with the repository.

## UCI Heart Disease dataset

`data/heart_disease_processed.parquet` is an adapted representation of the
[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease),
which is licensed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0).

Required attribution:

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart
> Disease [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C52P4X

Changes made for this project include converting the diagnosis to a binary
target, encoding categorical fields, adding missing-value indicators, median
or mode imputation, removing identifier and source-dataset fields, and
exporting the resulting table as Parquet. The processed dataset remains
available under CC BY 4.0; the original creators do not endorse this project.

## UCI Diabetes 130-US Hospitals dataset

The independent method-validation task uses the
[UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999+2008),
licensed under CC BY 4.0.

Required attribution:

> Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US
> Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5230J

This project preserves both the exact audited TableShift-compatible endpoint
(`readmitted != NO`) and the distinct 30-day endpoint, adds raw-line hashes,
and creates a patient-disjoint admission-source-shift split. The original
creators and TableShift authors do not endorse this project.

## eICU Collaborative Research Database Demo v2.0.1

Selected source files under `data/raw/eicu-crd-demo/2.0.1/` were obtained from
the [eICU Collaborative Research Database Demo v2.0.1](https://physionet.org/content/eicu-crd-demo/2.0.1/)
on PhysioNet. The source database is made available under the
[Open Data Commons Open Database License v1.0](https://opendatacommons.org/licenses/odbl/1-0/)
(ODbL 1.0). The repository's MIT License does not apply to those source files
or to database rights in the adapted files described below.

Required dataset citation:

> Johnson, A., Pollard, T., Badawi, O., & Raffa, J. (2021). eICU
> Collaborative Research Database Demo (version 2.0.1). PhysioNet.
> RRID:SCR_007345. https://doi.org/10.13026/4mxk-na84

`data/processed/eicu-demo-shiftguard-smoke-v1/` contains an adapted database
created from the demo. The transformation selects and joins the patient,
APACHE physiology, and APACHE result tables; deterministically resolves APACHE
result versions; derives the recorded hospital-mortality target; converts the
documented APACHE `-1` sentinel to missing values; converts the released
`> 89` age category to `90`; excludes APACHE prediction columns; and creates a
deterministic hospital-disjoint smoke-test manifest. No imputation is applied.
The directory's `LICENSE-ODbL-1.0.md` identifies the exact files in scope and
provides the required ODbL notice.

The public demo is used only for non-confirmatory pipeline and schema smoke
testing. It is not an external validation cohort, and neither PhysioNet nor the
dataset authors endorse this project.
