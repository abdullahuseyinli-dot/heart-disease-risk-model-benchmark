"""No-refit finalization of fixed neural shards from a preserved outer failure."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from heartshift.config import config_hash
from heartshift.data.uci import sha256_file
from heartshift.evaluation.classical_benchmark import write_run_manifest
from heartshift.evaluation.neural_outer import aggregate_neural_outer_shards
from heartshift.research.gates import validate_failed_outer_evidence


def finalize_fixed_neural_shards(
    repo_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Path]:
    """Copy and aggregate already-fixed labelled shards without any model execution."""
    source_run = repo_root / str(config["source_failed_run"])
    failure_evidence = validate_failed_outer_evidence(source_run)
    failure = json.loads((source_run / "failure.json").read_text(encoding="utf-8"))
    if failure.get("target_endpoint_loaded") is not True:
        raise AssertionError("No-refit finalization requires the disclosed post-endpoint failure")
    if failure.get("model_predictions_fixed_before_endpoint_loaded") is not True:
        raise AssertionError("Source failure does not prove probabilities were fixed pre-endpoint")
    if failure.get("adaptation_decisions_fixed_before_endpoint_loaded") is not True:
        raise AssertionError("Source failure does not prove adaptation was fixed pre-endpoint")
    source_shards = source_run / "shards"
    labelled_paths = sorted(source_shards.glob("*/outer_predictions.parquet"))
    unlabelled_paths = sorted(source_shards.glob("*/unlabelled_outer_predictions.parquet"))
    expected_shards = int(config["expected_complete_shards"])
    if len(labelled_paths) != expected_shards or len(unlabelled_paths) != expected_shards:
        raise AssertionError("The preserved recovery source does not contain every expected shard")
    for labelled_path, unlabelled_path in zip(labelled_paths, unlabelled_paths, strict=True):
        if labelled_path.parent != unlabelled_path.parent:
            raise AssertionError("Labelled and endpoint-free recovery shards are mispaired")
        labelled = pd.read_parquet(labelled_path)
        unlabelled = pd.read_parquet(unlabelled_path)
        if "target" in unlabelled.columns or "target" not in labelled.columns:
            raise AssertionError("Recovery shard endpoint boundary is invalid")
        if labelled["target"].isna().any():
            raise AssertionError("A fixed labelled recovery shard contains a missing endpoint")
        assert_frame_equal(
            labelled.drop(columns="target"),
            unlabelled,
            check_dtype=False,
            check_exact=True,
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(repo_root, run_dir, config, "locked_outer_neural_finalization_no_refit")
    shutil.copytree(source_shards, run_dir / "shards")
    outputs = aggregate_neural_outer_shards(run_dir / "shards", run_dir)
    source_hashes = {
        path.relative_to(source_run).as_posix(): sha256_file(path)
        for path in sorted(source_run.rglob("*"))
        if path.is_file()
    }
    source_config = json.loads((source_run / "config.resolved.json").read_text(encoding="utf-8"))
    provenance_path = run_dir / "finalization_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "status": "completed_no_refit_shard_finalization",
                "source_failed_run": str(config["source_failed_run"]),
                "source_failure_sha256": failure_evidence["failure_sha256"],
                "source_prediction_config_sha256": config_hash(source_config),
                "finalization_config_sha256": config_hash(config),
                "source_artifacts_sha256": source_hashes,
                "model_refit": False,
                "probabilities_recomputed": False,
                "adaptation_decisions_recomputed": False,
                "metrics_first_computed_in_finalization": True,
                "target_endpoint_already_loaded_in_source_failure": True,
                "exact_labelled_shards": [
                    path.relative_to(source_run).as_posix() for path in labelled_paths
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {**outputs, "provenance": provenance_path}
