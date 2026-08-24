# Protocol-v3 code dry run

This is a preserved plumbing test, not scientific acceptance evidence. It used
one seed, 80 records per environment, five training epochs, 32 RFF dimensions,
19 bootstrap repetitions, and deliberately permissive gate thresholds. With 19
bootstrap repetitions the minimum corrected p-value is 0.05, so this run cannot
demonstrate rejection at the configured `p >= 0.05` acceptance boundary.

Its purpose was limited to verifying the CUDA training loop, unique-patient
source artifact, target artifact, multiview diagnostic schema, persisted-summary
gate reconstruction, and complete output writing before the registered run.

