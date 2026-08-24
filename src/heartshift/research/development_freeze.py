"""Hash a development candidate before endpoint-isolated descriptive inference."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from heartshift.config import config_hash
from heartshift.data.uci import sha256_file


def _git_value(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def freeze_development_candidate(
    repo_root: Path,
    config: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write an immutable hash inventory while preserving consumed-data boundaries."""
    repo_root = repo_root.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Development freeze already exists: {output_path}")
    try:
        output_path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("Development freeze must be inside the repository") from error
    records = []
    for relative_value in config["files"]:
        relative = Path(str(relative_value))
        if relative.is_absolute():
            raise ValueError(f"Development-freeze path must be relative: {relative}")
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as error:
            raise ValueError(f"Development-freeze path escapes repository: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"Development-freeze input is missing: {path}")
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "protocol_version": str(config["protocol_version"]),
                "status": str(config["status"]),
                "created_utc": datetime.now(UTC).isoformat(),
                "new_confirmatory_claim_allowed": False,
                "outer_outcomes_historically_consumed": True,
                "purpose": (
                    "Freeze exact development bytes before endpoint-isolated "
                    "descriptive outer inference."
                ),
                "config_sha256": config_hash(config),
                "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
                "git_status_porcelain": _git_value(repo_root, "status", "--porcelain"),
                "files": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path
