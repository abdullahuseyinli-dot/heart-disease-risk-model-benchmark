# Raw source record

- Dataset: UCI Heart Disease
- DOI: <https://doi.org/10.24432/C52P4X>
- Download URL: <https://archive.ics.uci.edu/static/public/45/heart+disease.zip>
- Downloaded: 2026-08-23
- Archive SHA-256: `B17CD273DA9CE1CAA4710FCE80227EA454D4DBF9FCBC8E6A9121672751563ADC`
- Archive bytes: 128894

The archive and every extracted file are preserved as raw evidence. Canonical modelling data use the four official `processed.*.data` files. No raw file is overwritten during preparation.

The source uses `?` for documented missing values. Physiologically invalid zero placeholders in resting blood pressure, cholesterol, and maximum heart rate are retained in the raw files, represented by explicit zero-sentinel indicators, and converted to missing values only in the canonical table.
