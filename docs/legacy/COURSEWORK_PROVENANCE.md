# Coursework provenance and recovered trials

The December 2025 CS5079 group project is the origin of the legacy benchmark.
The retained report credits Abdulla Huseyinli, Ibrahim Alkali, Raghuraman TM,
Kaixin Zhu, and Khalid Suliman. This attribution is distinct from the authorship
of later HeartShift research. The report's historical title and clinical-use
aspirations are not claims made by the current benchmark.

In September 2026, 152 local coursework files were compared with the candidate
repository. Twenty-seven had byte-identical counterparts; 121 additional output
files are now preserved in [the coursework archive](../../results/legacy_coursework_2025/).
The three original notebooks and the group PDF remain local and are identified
by [size and SHA-256](../audit/2026-09-20/coursework_manifest.json).

## Trial-to-evidence map

| Trial or analysis | Retained evidence | Interpretation |
| --- | --- | --- |
| Model-family cross-validation | [Per-fold and repeated-CV tables](../../results/legacy_coursework_2025/Main_Experiment/metrics/cv/), [existing summary](../../results/main/metrics/cv_models_summary.csv) | Historical model-development comparisons; repeated folds are not independent test patients. |
| Nested-CV exploration | [Fold scores and selected hyperparameters](../../results/legacy_coursework_2025/Main_Experiment/metrics/nested_cv/), [hyperparameter plots](../../results/legacy_coursework_2025/Main_Experiment/figures/nested_cv/) | Preserve actual search outputs; do not convert a repeatedly inspected holdout into a locked result. |
| Training dynamics | [LightGBM and XGBoost training curves](../../results/legacy_coursework_2025/Main_Experiment/metrics/model_performance/) | Diagnostic train/validation trajectories, distinct from outer hospital testing. |
| Holdout and uncertainty | [Existing holdout values](../../results/main/metrics/test/holdout_models.csv), [bootstrap intervals](../../results/main/metrics/test/holdout_bootstrap_ci.csv), [additional reports](../../results/legacy_coursework_2025/Main_Experiment/metrics/test/) | Consumed development evidence; the archived model intervals overlap. |
| Calibration and thresholds | [Existing Brier decomposition](../../results/main/metrics/test/brier_decomposition_and_calibration.csv), [threshold table](../../results/main/metrics/test/lgbm_threshold_analysis.csv), [recovered plots](../../results/legacy_coursework_2025/Main_Experiment/figures/model_performance/) | Descriptive calibration and assumed-cost sensitivity, not measured clinical utility. |
| Explanation and stability | [Recovered SHAP plots](../../results/legacy_coursework_2025/Main_Experiment/figures/shap/), [interaction importance](../../results/legacy_coursework_2025/Main_Experiment/metrics/test/lgbm_shap_interaction_importance.csv), [bootstrap frequencies](../../results/legacy_coursework_2025/Main_Experiment/metrics/test/lgbm_shap_bootstrap_frequency.csv) | Model explanations, not causal mechanisms or evidence of clinical safety. |
| Recorded-sex subgroups | [Existing subgroup summary](../../results/main/metrics/test/subgroup_sex_metrics_and_fairness.csv), [LightGBM detail](../../results/legacy_coursework_2025/Main_Experiment/metrics/test/lgbm_subgroup_sex.csv) | Small-sample descriptive subgroup evidence; not a fairness certification. |
| Dask distributed training | [Original summary](<../../results/legacy_coursework_2025/Dask_Distributed/results/metrics/lightgbm_distributed_summary_colab.csv>), [all comparison plots](../../results/legacy_coursework_2025/Dask_Distributed/results/figures/model_performance/) | Failed comparison because of row/label alignment; retain the low AUC and execution cost. |
| Colab HTTP inference latency | [Original samples](<../../results/legacy_coursework_2025/Embedded Device Simulation/metrics/latency_samples_colab.csv>), [original summary](<../../results/legacy_coursework_2025/Embedded Device Simulation/metrics/latency_summary_colab.json>), [existing figures](../../results/edge/figures/) | A loopback HTTP experiment on one recorded environment; not a physical embedded-device measurement or service-level guarantee. |

The complete original text tables are preserved in
[tables_ascii](../../results/legacy_coursework_2025/Main_Experiment/tables_ascii/).
The [source-file manifest](../audit/2026-09-20/coursework_manifest.json) records
duplicate/renamed images, so filename count is not treated as experiment count.

## Notebook source representation

| Notebook | Code cells | Cells with stored execution counts | Reviewable source |
| --- | ---: | ---: | --- |
| Main experiment | 41 | 5 | [Main source extract](notebook_sources/main_experiment.py.txt) |
| Dask experiment | 1 | 1 | [Dask source extract](notebook_sources/dask_experiment.py.txt) |
| Embedded device simulation | 26 | 26 | [Edge source extract](notebook_sources/edge_simulation.py.txt) |

All 68 cells are indexed with original source hashes in the
[notebook source index](../audit/2026-09-20/notebook_source_index.json).
The extracts retain cell order, notebook magics, and original execution counts.
Outputs and notebook metadata are omitted. The index records workstation-path
redaction flags; none were needed in these source cells. These are archival text,
not executable reproduction scripts. The main
notebook has only five stored execution counts despite substantial saved output,
and the edge counts are out of order. This is a limitation of the retained
notebook state, not proof that every cell was executed in sequence.

The current historical scripts remain
[main benchmark](../../scripts/heart_disease_model_benchmark.py),
[Dask result visualizer](../../scripts/dask_lightgbm_results_visuals.py),
[edge inference service](../../scripts/edge_inference_service.py), and
[latency benchmark](../../scripts/edge_latency_benchmark.py). The source extracts
also preserve the notebook-specific upload, server-start, example-request,
measurement, and download sequence that the standalone scripts do not capture.

## Latency field interpretation

The edge notebook measures 2,760 sequential loopback requests: three repetitions
over 920 rows. It calls those rows `X_test`, but they are loaded from the processed
table; the variable name is not proof of a held-out evaluation. Its summary gives
mean latency 14.610649 ms and p95 33.251787 ms.

The historical `throughput_rps` field, 91.057378, is calculated as
`1000 / median(latency_ms)`. It is a latency-derived rate proxy, not observed
requests completed per wall-clock second under a service workload. The field
name and value remain unchanged in preserved evidence. No GPU/CPU comparison,
actual embedded deployment, concurrency capacity, or clinical inference validity
can be inferred from that single latency trial.

## Preservation and verification

Recovered outputs are copied byte-for-byte and excluded from Git text
normalization. Existing results keep their original paths. Run:

```powershell
uv run python tools/build_research_atlas.py --check
```

This verifies every mapped output's size and SHA-256 and checks the notebook
extract hashes. It does not claim to verify the unavailable original documents
on another machine. The original evidence limitations remain in the
[legacy validity audit](VALIDITY_AUDIT.md) and [archived benchmark](LEGACY_BENCHMARK.md).

For existing text files, the manifest separately records original Windows
checkout bytes and canonical Git-blob bytes. Only the declared CRLF-to-LF
conversion is permitted for those existing mappings; numeric changes fail the
check. Newly recovered outputs and source extracts retain their exact bytes on
both Windows and Linux.
