# Outer v4 failure and v5 exact-aggregation recovery

## Preserved v4 outcome

MIRRAMS v4 successfully completed all four held-out-hospital model fits and wrote
both endpoint-free and canonically labelled shard predictions. Every probability
and adaptation decision was fixed on disk before the endpoint-loading block.

The run then failed while assembling the top-level table. The pattern
`*/*outer_predictions.parquet` selected both `outer_predictions.parquet` and
`unlabelled_outer_predictions.parquet`. Concatenation therefore produced 99,360
rows instead of the 49,680 exact labelled-shard rows. Endpoint-free rows acquired
null targets, causing metric computation to stop before `outer_metrics.csv` was
written. The complete exception, row counts, and endpoint-access state are
preserved in `artifacts/runs/mirrams-outer-v4/failure.json`.

This failure differs from v3: the v4 target endpoint was loaded for the canonical
join. The result is therefore not described as unopened. However, the failure
occurred after model fitting, probability generation, and every adaptation
decision were complete, and no valid aggregate metric artifact was produced.

PS-MaskDRO v4 was not launched after this deterministic shared defect became
known.

## v5 recovery constraints

The v5 recovery makes one aggregation change: shard discovery requires the exact
basename `outer_predictions.parquet` (and the corresponding exact seed basename).
A regression test places labelled and `unlabelled_` files side by side and proves
only the labelled file is selected.

MIRRAMS is **not retrained**. The frozen finalizer:

1. verifies the preserved v4 failure and its complete artifact tree;
2. verifies each labelled shard is exactly its endpoint-free predecessor plus the
   canonical target column;
3. copies the fixed shards into a new, versioned evidence directory;
4. aggregates only exact labelled filenames and computes metrics for the first
   time;
5. records hashes of every source artifact and states that no model, probability,
   or adaptation decision was recomputed.

PS-MaskDRO remains unopened and is run once under the exact-filename fix. Its
features, selected configurations, epochs, seeds, policies, adaptable variants,
diagnostic, and bootstrap plan remain byte-for-byte equivalent in meaning to v4.
No MIRRAMS performance value is used to alter PS-MaskDRO.

## Versioned commands and outputs

- MIRRAMS finalization config: `configs/benchmark/mirrams_outer_v5_finalize.yaml`
- MIRRAMS finalized run: `artifacts/runs/mirrams-outer-v5-finalized`
- PS-MaskDRO config: `configs/benchmark/psmask_outer_v5.yaml`
- PS-MaskDRO run: `artifacts/runs/psmask-outer-v5`
- v5 lock: `artifacts/locks/heartshift_candidate_v5_aggregation_recovery.json`
- combined report: `artifacts/reports/heart-outer-v5`

The final report has mixed provenance: v3 baseline predictions, no-refit v5
finalization of fixed v4 MIRRAMS shards, and a once-run v5 PS-MaskDRO evaluation.
Both failed MIRRAMS attempts remain mandatory evidence.
