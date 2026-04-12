# Heart Disease Risk Model Benchmark

Benchmarking heart disease risk models with calibration, fairness, distributed training, interpretability, and edge-latency analysis.

## Overview

This repository presents a cleaned portfolio version of a heart disease prediction project built on tabular clinical data. The work compares Logistic Regression, LightGBM, XGBoost, and TabNet, and extends the comparison with calibration analysis, fairness evaluation, SHAP-based interpretability, distributed LightGBM experiments, and a lightweight edge-latency study.

## Key results

- On the hold-out set, TabNet achieved the strongest AUC at 0.930 and the lowest Brier score at 0.107.
- XGBoost achieved the strongest hold-out F1 at 0.877.
- Logistic Regression achieved a hold-out AUC of 0.908 and remained highly competitive as a transparent baseline.
- LightGBM offered a strong balance between predictive quality, calibration, efficiency, and deployment practicality.
- In the distributed comparison, Dask-based LightGBM underperformed on this small dataset, with AUC around 0.533 versus 0.896 for the single-node model.
- In the latency simulation, the exported LightGBM pipeline showed a mean latency of 14.61 ms, p95 latency of 33.25 ms, and throughput of about 91 requests per second.
- SHAP rankings for LightGBM and Logistic Regression remained strongly aligned, with Spearman rho = 0.771.

## Methods

- Logistic Regression
- LightGBM
- XGBoost
- TabNet
- Dask-based LightGBM comparison
- Edge-latency simulation

## Repository structure

```text
heart-disease-risk-model-benchmark/
├── scripts/
├── data/
├── results/
├── requirements.txt
├── requirements-optional.txt
└── README.md
