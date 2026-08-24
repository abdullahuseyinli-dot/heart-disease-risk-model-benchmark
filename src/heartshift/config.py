"""Configuration loading, canonicalization, and immutable experiment hashes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Hashable
from pathlib import Path
from typing import Any

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode


class ConfigurationError(ValueError):
    """Raised when a YAML configuration is ambiguous or structurally invalid."""


class StrictSafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    """Safe YAML loader that rejects aliases, merges, and duplicate keys."""

    def compose_node(self, parent: MappingNode | None, index: int | None) -> yaml.Node:
        if self.check_event(AliasEvent):
            event = self.peek_event()
            raise ConfigurationError(
                f"YAML aliases are prohibited at line {event.start_mark.line + 1}"
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        if not isinstance(node, MappingNode):
            raise ConfigurationError("expected a YAML mapping node")
        result: dict[Hashable, Any] = {}
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise ConfigurationError(
                    f"YAML merge keys are prohibited at line {key_node.start_mark.line + 1}"
                )
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ConfigurationError(
                    f"unhashable YAML key at line {key_node.start_mark.line + 1}"
                )
            if key in result:
                raise ConfigurationError(
                    f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_yaml(
    path: Path,
    *,
    required_keys: Collection[str] = (),
    allowed_keys: Collection[str] | None = None,
) -> dict[str, Any]:
    """Load unambiguous YAML and optionally enforce an exact top-level key set."""
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=StrictSafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    if not all(isinstance(key, str) for key in payload):
        raise ConfigurationError(f"configuration keys must be strings: {path}")
    typed_payload = {str(key): value for key, value in payload.items()}
    missing = sorted(set(required_keys) - set(typed_payload))
    if missing:
        raise ConfigurationError(f"missing required keys in {path}: {missing}")
    if allowed_keys is not None:
        unexpected = sorted(set(typed_payload) - set(allowed_keys))
        if unexpected:
            raise ConfigurationError(f"unexpected keys in {path}: {unexpected}")
    return typed_payload


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
