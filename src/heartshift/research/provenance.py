"""Immutable post-run source snapshots for experiments started from a dirty tree."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from heartshift.data.uci import sha256_file


def write_post_run_source_snapshot(
    repo_root: Path,
    run_dir: Path,
    source_files: list[Path],
) -> Path:
    """Hash exact run metadata and declared source files without rewriting evidence."""
    repo_root = repo_root.resolve()
    run_dir = run_dir.resolve()
    try:
        run_dir.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("Run directory must be inside the repository") from error
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    output = run_dir / "post_run_source_snapshot.json"
    if output.exists():
        raise FileExistsError(f"Post-run source snapshot already exists: {output}")
    declared = [run_dir / "run_manifest.json", run_dir / "config.resolved.json"]
    declared.extend(path if path.is_absolute() else repo_root / path for path in source_files)
    records = []
    for path in declared:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(repo_root)
        except ValueError as error:
            raise ValueError(f"Snapshot source is outside the repository: {path}") from error
        if not resolved.is_file():
            raise FileNotFoundError(f"Snapshot source is missing: {resolved}")
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(resolved),
                "bytes": resolved.stat().st_size,
            }
        )
    output.write_text(
        json.dumps(
            {
                "status": "post_run_exact_source_snapshot",
                "created_utc": datetime.now(UTC).isoformat(),
                "limitation": (
                    "This snapshot records post-run bytes and does not retroactively "
                    "turn a dirty working tree into a committed pre-run state."
                ),
                "files": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
