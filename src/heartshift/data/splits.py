"""Immutable hospital-level split construction and validation."""

from __future__ import annotations

import pandas as pd


def build_outer_manifest(data: pd.DataFrame) -> pd.DataFrame:
    """Return one explicit source/target assignment per sample and outer site."""
    sites = tuple(data["site"].drop_duplicates())
    records: list[pd.DataFrame] = []
    for outer_target in sites:
        assignment = data.loc[:, ["sample_id", "site", "record_sha256"]].copy()
        assignment.insert(0, "outer_target", outer_target)
        assignment["outer_role"] = (
            assignment["site"]
            .eq(outer_target)
            .map({True: "target_test", False: "source_development"})
        )
        records.append(assignment)
    result = pd.concat(records, ignore_index=True)
    validate_outer_manifest(data, result)
    return result


def build_inner_manifest(data: pd.DataFrame) -> pd.DataFrame:
    """Return every inner source-only train/validation assignment explicitly."""
    sites = tuple(data["site"].drop_duplicates())
    records: list[pd.DataFrame] = []
    for outer_target in sites:
        source = data.loc[data["site"].ne(outer_target), ["sample_id", "site", "record_sha256"]]
        for inner_validation in tuple(site for site in sites if site != outer_target):
            assignment = source.copy()
            assignment.insert(0, "inner_validation", inner_validation)
            assignment.insert(0, "outer_target", outer_target)
            assignment["inner_role"] = (
                assignment["site"].eq(inner_validation).map({True: "validation", False: "train"})
            )
            records.append(assignment)
    result = pd.concat(records, ignore_index=True)
    validate_inner_manifest(data, result)
    return result


def validate_outer_manifest(data: pd.DataFrame, manifest: pd.DataFrame) -> None:
    sites = set(data["site"])
    for outer_target, fold in manifest.groupby("outer_target"):
        if set(fold["sample_id"]) != set(data["sample_id"]):
            raise AssertionError(
                f"Outer fold {outer_target} does not contain every sample exactly once"
            )
        if len(fold) != len(data) or not fold["sample_id"].is_unique:
            raise AssertionError(f"Outer fold {outer_target} duplicates or omits sample IDs")
        target_sites = set(fold.loc[fold["outer_role"].eq("target_test"), "site"])
        if target_sites != {outer_target}:
            raise AssertionError(f"Outer target contamination for {outer_target}: {target_sites}")
    if set(manifest["outer_target"]) != sites:
        raise AssertionError("Outer manifest does not enumerate every hospital")


def validate_inner_manifest(data: pd.DataFrame, manifest: pd.DataFrame) -> None:
    sites = set(data["site"])
    for (outer_target, inner_validation), fold in manifest.groupby(
        ["outer_target", "inner_validation"]
    ):
        if outer_target in set(fold["site"]):
            raise AssertionError(f"Outer target {outer_target} leaked into inner fold")
        validation_sites = set(fold.loc[fold["inner_role"].eq("validation"), "site"])
        if validation_sites != {inner_validation}:
            raise AssertionError("Inner validation assignment is not hospital-disjoint")
        expected_ids = set(data.loc[data["site"].ne(outer_target), "sample_id"])
        if set(fold["sample_id"]) != expected_ids or not fold["sample_id"].is_unique:
            raise AssertionError("Inner fold does not contain each source sample exactly once")
    if set(manifest["outer_target"]) != sites:
        raise AssertionError("Inner manifest does not enumerate every outer target")
