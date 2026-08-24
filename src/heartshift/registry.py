"""Typed, release-facing registry for benchmark methods and provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from heartshift.config import ConfigurationError, load_yaml
from heartshift.contracts import validate_document

REGISTRY_PATH = Path("configs/research/method_registry_v3.yaml")
TOP_LEVEL_KEYS = {"schema_version", "record_kind", "status", "methods"}
METHOD_KEYS = {
    "method_id",
    "version",
    "family",
    "factory",
    "dependency_profile",
    "devices",
    "capabilities",
    "configuration_schema",
    "output_schema_version",
    "provenance",
    "evidence_eligibility",
}
CAPABILITY_KEYS = {
    "missing_values",
    "sample_weight",
    "unlabelled_target",
    "labelled_target",
}
PROVENANCE_KEYS = {
    "paper_url",
    "repository_url",
    "audited_revision",
    "license",
    "implementation_fidelity",
}


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{name} must be a string-keyed mapping")
    return {str(key): child for key, child in value.items()}


def _exact_keys(value: dict[str, Any], expected: set[str], *, name: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ConfigurationError(f"{name} key mismatch; missing={missing}, unexpected={unexpected}")


@dataclass(frozen=True)
class MethodCapabilities:
    missing_values: bool
    sample_weight: bool
    unlabelled_target: bool
    labelled_target: bool

    @classmethod
    def from_mapping(cls, value: object, *, name: str) -> MethodCapabilities:
        payload = _mapping(value, name=name)
        _exact_keys(payload, CAPABILITY_KEYS, name=name)
        if not all(isinstance(payload[key], bool) for key in CAPABILITY_KEYS):
            raise ConfigurationError(f"{name} capability flags must be booleans")
        return cls(**{key: payload[key] for key in CAPABILITY_KEYS})


@dataclass(frozen=True)
class MethodProvenance:
    paper_url: str | None
    repository_url: str | None
    audited_revision: str
    license: str
    implementation_fidelity: str

    @classmethod
    def from_mapping(cls, value: object, *, name: str) -> MethodProvenance:
        payload = _mapping(value, name=name)
        _exact_keys(payload, PROVENANCE_KEYS, name=name)
        for key in ("audited_revision", "license", "implementation_fidelity"):
            if not isinstance(payload[key], str) or not payload[key]:
                raise ConfigurationError(f"{name}.{key} must be a non-empty string")
        for key in ("paper_url", "repository_url"):
            if payload[key] is not None and not isinstance(payload[key], str):
                raise ConfigurationError(f"{name}.{key} must be a string or null")
        return cls(
            paper_url=payload["paper_url"],
            repository_url=payload["repository_url"],
            audited_revision=payload["audited_revision"],
            license=payload["license"],
            implementation_fidelity=payload["implementation_fidelity"],
        )


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    version: str
    family: str
    factory: str
    dependency_profile: str
    devices: tuple[str, ...]
    capabilities: MethodCapabilities
    configuration_schema: str | None
    output_schema_version: str
    provenance: MethodProvenance
    evidence_eligibility: str

    @classmethod
    def from_mapping(cls, value: object, *, index: int) -> MethodSpec:
        payload = _mapping(value, name=f"methods[{index}]")
        _exact_keys(payload, METHOD_KEYS, name=f"methods[{index}]")
        scalar_keys = (
            "method_id",
            "version",
            "family",
            "factory",
            "dependency_profile",
            "output_schema_version",
            "evidence_eligibility",
        )
        for key in scalar_keys:
            if not isinstance(payload[key], str) or not payload[key]:
                raise ConfigurationError(f"methods[{index}].{key} must be a non-empty string")
        devices = payload["devices"]
        if (
            not isinstance(devices, list)
            or not devices
            or not all(isinstance(device, str) for device in devices)
            or len(set(devices)) != len(devices)
        ):
            raise ConfigurationError(f"methods[{index}].devices must be unique strings")
        configuration_schema = payload["configuration_schema"]
        if configuration_schema is not None and not isinstance(configuration_schema, str):
            raise ConfigurationError(
                f"methods[{index}].configuration_schema must be a string or null"
            )
        return cls(
            method_id=payload["method_id"],
            version=payload["version"],
            family=payload["family"],
            factory=payload["factory"],
            dependency_profile=payload["dependency_profile"],
            devices=tuple(devices),
            capabilities=MethodCapabilities.from_mapping(
                payload["capabilities"], name=f"methods[{index}].capabilities"
            ),
            configuration_schema=configuration_schema,
            output_schema_version=payload["output_schema_version"],
            provenance=MethodProvenance.from_mapping(
                payload["provenance"], name=f"methods[{index}].provenance"
            ),
            evidence_eligibility=payload["evidence_eligibility"],
        )


@dataclass(frozen=True)
class MethodRegistry:
    schema_version: str
    status: str
    methods: tuple[MethodSpec, ...]

    def by_id(self) -> dict[str, MethodSpec]:
        return {method.method_id: method for method in self.methods}


def load_method_registry(repo_root: Path, path: Path | None = None) -> MethodRegistry:
    registry_path = path or repo_root / REGISTRY_PATH
    payload = load_yaml(
        registry_path,
        required_keys=TOP_LEVEL_KEYS,
        allowed_keys=TOP_LEVEL_KEYS,
    )
    validate_document(payload, repo_root / "configs/schema/method_registry.schema.json")
    values = payload["methods"]
    if not isinstance(values, list):
        raise ConfigurationError("method registry methods must be a list")
    methods = tuple(
        MethodSpec.from_mapping(value, index=index) for index, value in enumerate(values)
    )
    identifiers = [method.method_id for method in methods]
    if len(identifiers) != len(set(identifiers)):
        raise ConfigurationError("method registry contains duplicate method_id values")
    return MethodRegistry(
        schema_version=str(payload["schema_version"]),
        status=str(payload["status"]),
        methods=methods,
    )
