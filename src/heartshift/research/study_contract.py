"""Machine-checkable claim and data-use guardrails for post-v5 research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from heartshift.config import load_yaml
from heartshift.data.external import ExternalDatasetContract

ALLOWED_TRACKS = {"dg_zero_shot", "uda_unlabelled", "few_shot_labelled_future_only"}
PROHIBITED_CONSUMED_PHASES = {
    "confirmatory_architecture_selection",
    "locked_confirmation",
    "new_superiority_claim",
}
HISTORICAL_PSMASK_EXPERIMENTS = {
    "v0_pooled_erm",
    "v1_site_balanced",
    "v2_prior_separated",
    "v3_mcar_augmentation",
    "v4_structured_policy_bank",
    "v5_site_only_dro",
    "v5_mask_only_dro",
    "v5_site_mask_dro",
    "v5_site_mask_dro_brier",
    "v7_ane",
}


def validate_shiftguard_study_config(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate the protocol before any external endpoint can be opened."""
    if str(config.get("protocol_version")) != "shiftguard-v1-development":
        raise ValueError("Unexpected ShiftGuard protocol version")
    consumed = tuple(str(value) for value in config.get("consumed_evidence", []))
    if not consumed or len(set(consumed)) != len(consumed):
        raise ValueError("Consumed evidence must be explicitly and uniquely listed")
    prohibited = {str(value) for value in config.get("consumed_evidence_prohibited_uses", [])}
    if not {"confirmatory_architecture_selection", "new_superiority_claim"} <= prohibited:
        raise ValueError("Consumed-target claim prohibitions are incomplete")
    tracks = tuple(str(value) for value in config.get("tracks", []))
    if set(tracks) != ALLOWED_TRACKS or len(tracks) != len(set(tracks)):
        raise ValueError("DG, UDA, and future few-shot tracks must remain explicit and separate")
    seeds = tuple(int(value) for value in config.get("seeds", []))
    if len(seeds) < 10 or len(set(seeds)) != len(seeds):
        raise ValueError("ShiftGuard development requires at least ten unique seeds")
    evaluation = config["evaluation"]
    if int(evaluation["development_seed_count"]) != len(seeds):
        raise ValueError("Declared development seed count does not match the seed list")
    if float(evaluation["natural_auroc_noninferiority_margin"]) > 0.01:
        raise ValueError("Natural AUROC non-inferiority margin cannot exceed 0.01")
    shiftguard = config["shiftguard"]
    if not 0.0 < float(shiftguard["alpha"]) < 1.0:
        raise ValueError("ShiftGuard alpha is invalid")
    if int(shiftguard["calibration_repetitions_per_prior"]) < 100:
        raise ValueError("ShiftGuard calibration is underpowered by protocol")
    external_path = repo_root / str(config["external_data_config"])
    if not external_path.is_file():
        raise FileNotFoundError(f"External data contract is missing: {external_path}")
    external = ExternalDatasetContract.from_mapping(load_yaml(external_path))
    registry_path = repo_root / str(config["method_registry"])
    if not registry_path.is_file():
        raise FileNotFoundError(f"Comparator method registry is missing: {registry_path}")
    registry = load_yaml(registry_path)
    omissions = {str(value["method"]) for value in registry.get("named_omissions", [])}
    required_omissions = {"neumiss_neumise", "caustab", "distpfn"}
    if not required_omissions <= omissions:
        raise ValueError("Comparator registry silently drops a declared unavailable method")
    evidence = {str(value) for value in config.get("required_evidence", [])}
    required_evidence = {
        "exact_mask_hashes",
        "patient_disjoint_manifest",
        "hospital_disjoint_manifest",
        "source_oof_predictions",
        "individual_seed_predictions",
        "locked_external_predictions",
        "independent_report_reconstruction",
    }
    if not required_evidence <= evidence:
        raise ValueError(
            f"ShiftGuard evidence contract misses: {sorted(required_evidence - evidence)}"
        )
    return {
        "status": "passed_shiftguard_study_contract",
        "protocol_version": str(config["protocol_version"]),
        "consumed_datasets": list(consumed),
        "tracks": list(tracks),
        "seed_count": len(seeds),
        "external_dataset": external.name,
        "named_omission_count": len(omissions),
        "confirmation_fraction": external.confirmation_fraction,
    }


def assert_dataset_use_allowed(
    config: dict[str, Any],
    *,
    dataset_id: str,
    phase: str,
) -> None:
    """Fail closed when a consumed dataset is assigned a confirmatory role."""
    consumed = {str(value) for value in config.get("consumed_evidence", [])}
    if dataset_id in consumed and phase in PROHIBITED_CONSUMED_PHASES:
        raise PermissionError(
            f"Dataset {dataset_id} is consumed and cannot be used for phase {phase}"
        )


def validate_router_development_contract(
    control: dict[str, Any],
    neural_inner: dict[str, Any],
    neural_outer: dict[str, Any],
    router: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Validate seed, mask, comparator, and claim alignment across router stages."""
    if str(router.get("status")) != "consumed_uci_development_only":
        raise ValueError("The full router contract must remain consumed-data development")
    if str(report.get("status")) != "post_outcome_consumed_uci_development_only":
        raise ValueError("The router report cannot be relabelled as confirmation")
    seed_sets = {
        tuple(int(value) for value in control["seeds"]),
        tuple(int(value) for value in neural_inner["seeds"]),
        tuple(int(value) for value in neural_outer["seeds"]),
        tuple(int(value) for value in router["prediction_seeds"]),
        tuple(int(value) for value in router["router_seeds"]),
        tuple(int(value) for value in report["seeds"]),
    }
    if len(seed_sets) != 1:
        raise ValueError("Control, neural, router, and report seeds must match exactly")
    seeds = next(iter(seed_sets))
    if len(seeds) < 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Full router development requires at least ten unique seeds")
    if int(control["outer_mask_replicates"]) != int(neural_outer["outer_mask_replicates"]):
        raise ValueError("Control and neural outer mask replicate counts differ")
    if int(control["outer_evaluation_seed"]) != int(neural_inner["inner_evaluation_seed"]):
        raise ValueError("Control and neural evaluation-mask seeds differ")
    configured_experiments = {
        str(experiment["name"]) for experiment in neural_inner["experiments"]
    }
    grouped_experiments = {
        str(experiment)
        for values in router["neural_experiment_groups"].values()
        for experiment in values
    }
    if grouped_experiments != configured_experiments:
        raise ValueError("Router neural groups do not cover the equal-budget experiment set")
    feature_modes = {
        str(candidate["feature_mode"]) for candidate in router["router_candidates"]
    }
    required_modes = {
        "support_only",
        "expert_logit_context",
        "anchor_relative_context",
    }
    if feature_modes != required_modes:
        raise ValueError("Router context ablations are incomplete or changed")
    inference = router["inference"]
    if str(inference["reference_method"]) != "support_aware_router":
        raise ValueError("Router uncertainty must use the router as the paired reference")
    comparisons = {str(value) for value in inference["comparison_methods"]}
    required_comparisons = {
        "equal_logit_blend",
        "selected_best_expert",
        "selected_convex_blend",
    }
    if not required_comparisons <= comparisons:
        raise ValueError("Strong fixed router comparisons are incomplete")
    if int(report["bootstrap_repetitions"]) < 2000:
        raise ValueError("Full research report bootstrap is underpowered by protocol")
    return {
        "status": "passed_router_development_contract",
        "seed_count": len(seeds),
        "outer_mask_replicates": int(control["outer_mask_replicates"]),
        "neural_experiment_count": len(configured_experiments),
        "router_feature_mode_count": len(feature_modes),
    }


def validate_historical_seed_extension_contract(
    control: dict[str, Any],
    extension: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Validate the explicitly post-outcome ten-seed stability experiment."""
    if str(extension.get("protocol_version")) != (
        "historical-psmask-ten-seed-post-outcome-sensitivity-v1"
    ):
        raise ValueError("Unexpected historical PS-MaskDRO extension protocol")
    if str(extension.get("status")) != "consumed_uci_development_only":
        raise ValueError("The seed extension must remain consumed-data development")
    if str(extension.get("manifest_stage")) != (
        "consumed_uci_post_outcome_seed_extension_sensitivity"
    ):
        raise ValueError("The seed extension must disclose its post-outcome timing")
    if str(report.get("status")) != (
        "post_outcome_consumed_uci_seed_sensitivity_only"
    ):
        raise ValueError("The seed-extension report cannot be relabelled as confirmation")

    control_seeds = tuple(int(value) for value in control["seeds"])
    extension_seeds = tuple(int(value) for value in extension["seeds"])
    report_seeds = tuple(int(value) for value in report["seeds"])
    if not control_seeds == extension_seeds == report_seeds:
        raise ValueError("Control, extension, and report seeds must match exactly")
    if len(extension_seeds) < 10 or len(set(extension_seeds)) != len(extension_seeds):
        raise ValueError("The historical stability extension requires ten unique seeds")

    historical_seeds = tuple(int(value) for value in report["historical_seeds"])
    if len(historical_seeds) != 3 or historical_seeds != extension_seeds[:3]:
        raise ValueError("Historical seeds must be the exact three-seed prefix")
    if int(extension["outer_mask_replicates"]) != int(
        control["outer_mask_replicates"]
    ):
        raise ValueError("Control and extension outer mask replicate counts differ")
    if list(extension.get("adaptable_variants", [])):
        raise ValueError("Unlabelled-target adaptation is prohibited in this DG sensitivity")
    if str(extension["data"]["canonical_path"]) != str(
        control["data"]["canonical_path"]
    ):
        raise ValueError("Control and extension canonical datasets differ")

    experiments = tuple(str(value) for value in report["experiments"])
    if len(experiments) != len(set(experiments)) or set(experiments) != (
        HISTORICAL_PSMASK_EXPERIMENTS
    ):
        raise ValueError("The historical PS-MaskDRO experiment family is incomplete")
    controls = {str(value) for value in report["control_methods"]}
    required_controls = {
        "control:random_forest_natural",
        "control:logistic_natural",
    }
    if not required_controls <= controls:
        raise ValueError("The seed extension omits required classical controls")
    if int(report["bootstrap"]["repetitions"]) < 2000:
        raise ValueError("The seed-extension bootstrap is underpowered by protocol")
    if str(report.get("historical_run")) != "artifacts/runs/psmask-outer-v5":
        raise ValueError("The seed extension no longer points to the locked historical run")
    if str(report.get("extension_run")) != (
        "artifacts/runs/historical-psmask-ten-seed-sensitivity-v1"
    ):
        raise ValueError("The report no longer points to the exact ten-seed run")

    return {
        "status": "passed_historical_seed_extension_contract",
        "seed_count": len(extension_seeds),
        "historical_seed_count": len(historical_seeds),
        "outer_mask_replicates": int(extension["outer_mask_replicates"]),
        "experiment_count": len(experiments),
        "bootstrap_repetitions": int(report["bootstrap"]["repetitions"]),
    }
