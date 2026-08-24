"""Audited comparison of versioned ShiftGuard synthetic revisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr

from heartshift.data.uci import sha256_file
from heartshift.research.shiftguard_experiment import CONCEPT_FAILURE_CONTROL


def _affected_probability(mechanism: str, prevalence: float) -> float:
    if mechanism in {
        "conditional_translation",
        "covariance_shear",
        "nonlinear_bimodal",
    }:
        return prevalence
    if mechanism == "conditional_scale":
        return 1.0 - prevalence
    if mechanism == "outcome_dependent_dropout":
        return 0.9 * prevalence
    if mechanism == "tail_contamination":
        return 0.15
    if mechanism == "support_translation":
        return 1.0
    return float("nan")


def _log_loss(target: pd.Series, score: pd.Series) -> float:
    labels = target.to_numpy(dtype=np.int8)
    probability = np.clip(score.to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))


def _sample_derived_log_losses(samples: pd.DataFrame) -> pd.DataFrame:
    records = []
    for episode_id, group in samples.groupby("episode_id", sort=False):
        records.append(
            {
                "episode_id": episode_id,
                "zero_shot_log_loss_rebuilt": _log_loss(group["target"], group["zero_shot_score"]),
                "gated_log_loss_rebuilt": _log_loss(group["target"], group["gated_score"]),
            }
        )
    return pd.DataFrame(records)


def compare_shiftguard_revisions(
    repo_root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    revision_records = []
    mechanism_records = []
    prior_power_records = []
    source_hashes = {}
    for specification in config["runs"]:
        revision = str(specification["revision"])
        run_dir = repo_root / str(specification["path"])
        episodes_path = run_dir / "episode_results.parquet"
        samples_path = run_dir / "sample_predictions.parquet"
        gates_path = run_dir / "synthetic_gates.json"
        episodes = pd.read_parquet(episodes_path)
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
        primary = str(gates["primary_representation"])
        primary_episodes = episodes.loc[episodes["representation"].eq(primary)].copy()
        rebuilt = _sample_derived_log_losses(pd.read_parquet(samples_path))
        primary_episodes = primary_episodes.merge(
            rebuilt, on="episode_id", how="left", validate="one_to_one"
        )
        if primary_episodes["zero_shot_log_loss_rebuilt"].isna().any():
            raise AssertionError(f"Revision {revision} lacks sample-derived episode scores")
        if "zero_shot_log_loss" in primary_episodes and not np.allclose(
            primary_episodes["zero_shot_log_loss"],
            primary_episodes["zero_shot_log_loss_rebuilt"],
            atol=1e-12,
        ):
            raise AssertionError(f"Revision {revision} log-loss aggregates do not rebuild")
        valid = primary_episodes["mechanism"].eq("pure_label_shift")
        concept = primary_episodes["mechanism"].eq(CONCEPT_FAILURE_CONTROL)
        invalid = ~valid & ~concept
        revision_records.append(
            {
                "revision": revision,
                "protocol_version": specification["protocol_version"],
                "primary_representation": primary,
                "valid_acceptance": float(primary_episodes.loc[valid, "accepted"].mean()),
                "observable_invalid_acceptance": float(
                    primary_episodes.loc[invalid, "accepted"].mean()
                ),
                "valid_gated_minus_zero_log_loss": float(
                    (
                        primary_episodes.loc[valid, "gated_log_loss_rebuilt"]
                        - primary_episodes.loc[valid, "zero_shot_log_loss_rebuilt"]
                    ).mean()
                ),
                "invalid_gated_minus_zero_log_loss": float(
                    (
                        primary_episodes.loc[invalid, "gated_log_loss_rebuilt"]
                        - primary_episodes.loc[invalid, "zero_shot_log_loss_rebuilt"]
                    ).mean()
                ),
                "concept_acceptance": float(primary_episodes.loc[concept, "accepted"].mean()),
                "concept_gated_minus_zero_log_loss": float(
                    (
                        primary_episodes.loc[concept, "gated_log_loss_rebuilt"]
                        - primary_episodes.loc[concept, "zero_shot_log_loss_rebuilt"]
                    ).mean()
                ),
            }
        )
        grouped = primary_episodes.groupby(["mechanism", "batch_size"], as_index=False).agg(
            acceptance_rate=("accepted", "mean"),
            episodes=("episode_id", "size"),
        )
        grouped.insert(0, "revision", revision)
        mechanism_records.append(grouped)
        prior_power = primary_episodes.groupby(
            ["mechanism", "batch_size", "nominal_latent_prevalence"],
            as_index=False,
        ).agg(
            acceptance_rate=("accepted", "mean"),
            realized_latent_prevalence=("realized_latent_prevalence", "mean"),
            episodes=("episode_id", "size"),
        )
        prior_power.insert(0, "revision", revision)
        prior_power_records.append(prior_power)
        source_hashes[revision] = {
            episodes_path.name: sha256_file(episodes_path),
            samples_path.name: sha256_file(samples_path),
            gates_path.name: sha256_file(gates_path),
        }
    comparison_path = output_dir / "revision_comparison.csv"
    mechanism_path = output_dir / "mechanism_batch_acceptance.csv"
    prior_power_path = output_dir / "mechanism_batch_prior_acceptance.csv"
    manifest_path = output_dir / "report_manifest.json"
    audit_path = output_dir / "evidence_audit.json"
    pd.DataFrame(revision_records).to_csv(comparison_path, index=False)
    pd.concat(mechanism_records, ignore_index=True).to_csv(mechanism_path, index=False)
    prior_power = pd.concat(prior_power_records, ignore_index=True)
    prior_power.to_csv(prior_power_path, index=False)
    power = prior_power.loc[
        ~prior_power["mechanism"].isin(["pure_label_shift", CONCEPT_FAILURE_CONTROL])
    ].copy()
    power["affected_probability"] = [
        _affected_probability(str(mechanism), float(prevalence))
        for mechanism, prevalence in zip(
            power["mechanism"], power["nominal_latent_prevalence"], strict=True
        )
    ]
    power["expected_affected_rows"] = power["batch_size"] * power["affected_probability"]
    power["probability_fewer_than_five_affected"] = [
        float(binom.cdf(4, int(size), float(probability)))
        for size, probability in zip(
            power["batch_size"], power["affected_probability"], strict=True
        )
    ]
    power_path = output_dir / "mechanism_power_feasibility.csv"
    power.to_csv(power_path, index=False)
    power_summary = {}
    for revision, group in power.groupby("revision", sort=False):
        correlation = spearmanr(
            group["expected_affected_rows"],
            group["acceptance_rate"],
            nan_policy="omit",
        )
        power_summary[str(revision)] = {
            "cells": len(group),
            "spearman_expected_affected_vs_acceptance": float(correlation.statistic),
            "acceptance_expected_affected_below_five": float(
                group.loc[group["expected_affected_rows"] < 5.0, "acceptance_rate"].mean()
            ),
            "acceptance_expected_affected_at_least_twenty": float(
                group.loc[group["expected_affected_rows"] >= 20.0, "acceptance_rate"].mean()
            ),
        }
    power_summary_path = output_dir / "power_feasibility_summary.json"
    power_summary_path.write_text(
        json.dumps(power_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "status": "sample_reconstructed_synthetic_revision_comparison",
                "selection_status": "iterative_mechanism_development_not_confirmation",
                "ordinary_log_loss_rebuilt_for_every_revision": True,
                "balanced_log_loss_not_used_as_prevalence_adaptation_success_metric": True,
                "source_sha256": source_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "sha256": {
                    path.name: sha256_file(path)
                    for path in (
                        comparison_path,
                        mechanism_path,
                        prior_power_path,
                        power_path,
                        power_summary_path,
                        manifest_path,
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "comparison": comparison_path,
        "mechanisms": mechanism_path,
        "prior_power": prior_power_path,
        "power_feasibility": power_path,
        "power_summary": power_summary_path,
        "manifest": manifest_path,
        "audit": audit_path,
    }
