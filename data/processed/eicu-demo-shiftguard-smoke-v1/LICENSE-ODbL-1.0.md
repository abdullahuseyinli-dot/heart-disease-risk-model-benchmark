# eICU demo derived-data licence and notice

## Scope

This notice applies only to these adapted database files in this directory:

- `canonical.parquet`
- `hospital_split.parquet`
- `profile.json`, to the extent it contains or describes the adapted database

It does not apply to HeartShift source code, documentation, configurations, or
data derived exclusively from other sources.

## Source and licence

Contains information from the
[eICU Collaborative Research Database Demo v2.0.1](https://physionet.org/content/eicu-crd-demo/2.0.1/),
which is made available under the
[Open Data Commons Open Database License v1.0](https://opendatacommons.org/licenses/odbl/1-0/)
(ODbL 1.0).

These adapted database files are made available under ODbL 1.0 to the extent
that the ODbL applies and the project has rights to offer the adaptation. The
repository's MIT License does not apply to the source database, these adapted
database files, or their database contents. Users must retain this notice and
comply with the ODbL, including its notice and share-alike conditions where
applicable. The ODbL governs database rights and does not necessarily license
independent rights in individual contents.

## Changes made

HeartShift selected and joined released patient, APACHE physiology, and APACHE
result fields; deterministically selected an APACHE result version; represented
recorded hospital mortality as a binary target; converted APACHE `-1` sentinels
to missing values; converted the released `> 89` age category to `90`; excluded
APACHE prediction columns; and created deterministic hospital-disjoint role
assignments. No imputation was applied. Exact source and output hashes are in
`../../raw/eicu-crd-demo/2.0.1/SHA256SUMS.json` and `profile.json` respectively.

## Citation and claim boundary

> Johnson, A., Pollard, T., Badawi, O., & Raffa, J. (2021). eICU
> Collaborative Research Database Demo (version 2.0.1). PhysioNet.
> RRID:SCR_007345. https://doi.org/10.13026/4mxk-na84

The adapted files support only a public, non-confirmatory software and schema
smoke test. They are not an external validation cohort and must not be used to
support a clinical or method-performance claim. The source authors and
PhysioNet do not endorse HeartShift.
