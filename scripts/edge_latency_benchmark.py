from pathlib import Path
import argparse
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


def to_python_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def main():
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Benchmark latency of the edge inference service."
    )
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--data",
        default=str(repo_root / "data" / "heart_disease_processed.parquet"),
    )
    parser.add_argument(
        "--bundle",
        default=str(repo_root / "deployment_bundle" / "lgbm_deployment_bundle.joblib"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--n-requests", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    metrics_dir = repo_root / "results" / "edge" / "metrics"
    figures_dir = repo_root / "results" / "edge" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)

    bundle_path = Path(args.bundle)
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        feature_names = bundle.get("feature_names")
        if not feature_names:
            raise ValueError("Deployment bundle is missing 'feature_names'.")
        X = df[feature_names].copy()
    else:
        X = df.drop(columns=["num"], errors="ignore").copy()

    if X.empty:
        raise ValueError("No feature rows available for benchmarking.")

    service_url = args.host.rstrip("/") + "/predict"
    rows = []
    request_idx = 0

    def run_request(phase, row):
        nonlocal request_idx
        payload = {"features": {k: to_python_scalar(v) for k, v in row.to_dict().items()}}
        t0 = time.perf_counter()
        response = requests.post(service_url, json=payload, timeout=args.timeout)
        response.raise_for_status()
        t1 = time.perf_counter()
        rows.append(
            {
                "request_idx": request_idx,
                "phase": phase,
                "stage": "predict",
                "latency_ms": (t1 - t0) * 1000.0,
            }
        )
        request_idx += 1

    for i in range(min(args.warmup, len(X))):
        run_request("warmup", X.iloc[i % len(X)])

    for i in range(args.n_requests):
        run_request("steady", X.iloc[i % len(X)])

    df_lat = pd.DataFrame(rows)
    df_lat["delta_ms"] = df_lat["latency_ms"].diff().abs().fillna(0.0)

    ref = df_lat.loc[df_lat["phase"] == "steady", "latency_ms"]
    if ref.empty:
        ref = df_lat["latency_ms"]

    summary = {
        "n_requests": int(len(df_lat)),
        "mean_ms": float(df_lat["latency_ms"].mean()),
        "median_ms": float(df_lat["latency_ms"].median()),
        "std_ms": float(df_lat["latency_ms"].std(ddof=1) if len(df_lat) > 1 else 0.0),
        "p90_ms": float(np.percentile(df_lat["latency_ms"], 90)),
        "p95_ms": float(np.percentile(df_lat["latency_ms"], 95)),
        "p99_ms": float(np.percentile(df_lat["latency_ms"], 99)),
        "max_ms": float(df_lat["latency_ms"].max()),
        "throughput_rps": float(1000.0 / ref.median()),
    }

    df_lat.to_csv(metrics_dir / "latency_samples.csv", index=False)
    with open(metrics_dir / "latency_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(6, 4))
    plt.hist(df_lat["latency_ms"], bins=30)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Latency distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "latency_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df_lat["request_idx"], df_lat["latency_ms"], marker=".", linestyle="-", alpha=0.7)
    plt.xlabel("Request index")
    plt.ylabel("Latency (ms)")
    plt.title("Latency over time")
    plt.tight_layout()
    plt.savefig(figures_dir / "latency_over_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df_lat["request_idx"], df_lat["delta_ms"], marker=".", linestyle="-", alpha=0.7)
    plt.xlabel("Request index")
    plt.ylabel("Absolute change in latency (ms)")
    plt.title("Latency jitter")
    plt.tight_layout()
    plt.savefig(figures_dir / "latency_jitter.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    df_lat.boxplot(column="latency_ms", by="phase")
    plt.suptitle("")
    plt.title("Warmup vs steady latency")
    plt.ylabel("Latency (ms)")
    plt.tight_layout()
    plt.savefig(figures_dir / "latency_warmup_vs_steady.png", dpi=150)
    plt.close()

    print("Saved metrics to", metrics_dir)
    print("Saved figures to", figures_dir)


if __name__ == "__main__":
    main()
