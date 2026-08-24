# UCI Diabetes 130-US Hospitals source record

- Dataset: Diabetes 130-US Hospitals for Years 1999-2008
- DOI: <https://doi.org/10.24432/C5230J>
- Download URL: <https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip>
- Downloaded: 2026-08-23
- Archive SHA-256: `F82AC129DA2DDD2299391FF6FBAE3A6A58B3EDCF59AC9D7BD480C00FE453112A`
- License: Creative Commons Attribution 4.0

The raw archive and extracted files are immutable evidence. The independent
benchmark records both a faithful TableShift encounter split and a strengthened
patient-disjoint split. TableShift commit
`fca9429814703a07e3902d005d46563a207b7f0a` defines admission source 7 as OOD,
uses random state 264738, drops rows with missing race, and maps every non-`NO`
outcome to the positive class. That is an *any readmission* endpoint despite the
benchmark prose referring to 30-day readmission. HeartShift therefore retains a
separate `readmitted == "<30"` sensitivity label and never conflates the two.
