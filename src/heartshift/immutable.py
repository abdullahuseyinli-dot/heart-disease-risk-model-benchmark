"""Create-only publication helpers for derived data and evidence records."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from heartshift.contracts import canonical_json_bytes


class ImmutableWriteError(FileExistsError):
    """Raised when a create-only destination already exists."""


class ParquetWritable(Protocol):
    def to_parquet(self, path: Path, *, index: bool) -> None: ...


def preflight_create_only(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Resolve and reject every existing destination before any write starts."""
    targets = tuple(Path(path) for path in paths)
    duplicates = sorted(str(path) for path in targets if targets.count(path) > 1)
    if duplicates:
        raise ImmutableWriteError(f"duplicate create-only destinations: {duplicates}")
    existing = [path for path in targets if os.path.lexists(path)]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise ImmutableWriteError(f"refusing to overwrite existing evidence: {joined}")
    return targets


def _staging_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.partial.{uuid.uuid4().hex}")


def publish_staged_create_only(staged: Path, target: Path) -> None:
    """Publish a completed same-directory staging file without replacement."""
    if staged.parent.resolve() != target.parent.resolve():
        raise ValueError("staging and target paths must share a directory")
    try:
        os.link(staged, target)
    except FileExistsError as exc:
        raise ImmutableWriteError(f"refusing to overwrite existing evidence: {target}") from exc
    else:
        staged.unlink()


def write_bytes_create_only(target: Path, payload: bytes, *, read_only: bool = False) -> Path:
    preflight_create_only((target,))
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = _staging_path(target)
    with staged.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    publish_staged_create_only(staged, target)
    if read_only:
        target.chmod(0o444)
    return target


def write_json_create_only(target: Path, payload: Mapping[str, Any]) -> Path:
    pretty = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return write_bytes_create_only(target, pretty.encode("utf-8"))


def write_parquet_create_only(target: Path, frame: ParquetWritable) -> Path:
    preflight_create_only((target,))
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = _staging_path(target)
    frame.to_parquet(staged, index=False)
    publish_staged_create_only(staged, target)
    return target


def validate_canonical_payload(payload: Mapping[str, Any]) -> None:
    """Expose canonical validation to callers before any output path is touched."""
    canonical_json_bytes(payload)
