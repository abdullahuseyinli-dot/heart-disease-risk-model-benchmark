# Third-party notices

The repository's MIT License covers the original source code and documentation.
It does not replace the license of the dataset described below.

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
