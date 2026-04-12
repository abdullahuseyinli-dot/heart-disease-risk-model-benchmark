# Heart Disease Risk Model Benchmark

Benchmarking heart disease risk models with calibration, fairness analysis, distributed training comparisons, interpretability, and edge inference latency evaluation.

## Overview

This repository presents a cleaned portfolio version of a heart disease prediction project built on tabular clinical data. The work compares Logistic Regression, LightGBM, XGBoost, and TabNet, then extends the analysis with calibration diagnostics, fairness checks, SHAP-based interpretability, a distributed LightGBM comparison, and a lightweight edge inference simulation.

## Key results

- On the hold-out set, TabNet achieved the strongest AUC at **0.930** and the lowest Brier score at **0.107**.
- XGBoost achieved the strongest hold-out **F1 score of 0.877**.
- Logistic Regression achieved a hold-out **AUC of 0.908** and remained a strong transparent baseline.
- LightGBM provided a strong balance between predictive performance, calibration, efficiency, and deployment suitability.
- In the distributed comparison, Dask-based LightGBM underperformed on this dataset, with **AUC ≈ 0.533** versus **0.896** for the single-node model.
- In the edge inference simulation, the exported LightGBM pipeline showed **mean latency of 14.61 ms**, **p95 latency of 33.25 ms**, and **throughput of about 91 requests per second**.
- SHAP rankings for LightGBM and Logistic Regression remained strongly aligned, with **Spearman rho = 0.771**.

## Methods

- Logistic Regression
- LightGBM
- XGBoost
- TabNet
- Dask-based LightGBM comparison
- Edge inference latency simulation

## Repository structure

```text
heart-disease-risk-model-benchmark/
├── scripts/
├── data/
├── results/
│   ├── main/
│   ├── dask/
│   └── edge/
├── deployment_bundle/
├── requirements.txt
├── requirements-optional.txt
├── requirements-edge.txt
└── README.md
