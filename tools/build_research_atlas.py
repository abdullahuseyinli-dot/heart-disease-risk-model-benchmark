"""Build and verify the research index without fitting models or changing evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = "docs/research/trial_classification.json"
AUDIT = "docs/audit/2026-09-20"
STATUS_FIELDS = (
    "status",
    "passed",
    "gate_passed",
    "phase",
    "evidence_status",
    "evidence_class",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {relative}")
    return path


def load(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key, value in pairs:
            if key in data:
                raise ValueError(f"Duplicate key {key} in {path}")
            data[key] = value
        return data

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def inspect_trials(root: Path, catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Require explicit classification of every retained run, report and failure."""
    actual = {
        path.relative_to(root).as_posix()
        for kind in ("runs", "reports", "failures")
        for path in (root / "artifacts" / kind).iterdir()
        if path.is_dir()
    }
    entries = catalog["entries"]
    paths = [entry["path"] for entry in entries]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate trial classification")
    if set(paths) != actual:
        raise ValueError(
            f"Trial coverage mismatch: unclassified={sorted(actual - set(paths))}, "
            f"missing={sorted(set(paths) - actual)}"
        )
    result = []
    for entry in sorted(entries, key=lambda item: item["path"]):
        path = local_path(root, entry["path"])
        if not local_path(root, entry["reference"]).is_file():
            raise ValueError(f"Missing interpretation reference: {entry['reference']}")
        records = []
        signals = []
        for source in sorted(path.glob("*.json")):
            data = load(source)
            if not isinstance(data, dict):
                continue
            fields = {key: data[key] for key in STATUS_FIELDS if key in data}
            if fields:
                records.append(
                    {
                        "path": source.relative_to(root).as_posix(),
                        "sha256": digest(source),
                        "fields": fields,
                    }
                )
                if "gate" in source.name or "failure" in source.name:
                    signals.append(f"{source.name}: {json.dumps(fields, sort_keys=True)}")
        manifest = path / "run_manifest.json"
        metadata = load(manifest) if manifest.exists() else {}
        files = [item for item in path.rglob("*") if item.is_file()]
        result.append(
            {
                **entry,
                "kind": entry["path"].split("/")[1],
                "created_utc": metadata.get("created_utc", "not_recorded"),
                "run_commit": metadata.get("git_commit", "not_recorded"),
                "retained_file_count": len(files),
                "recorded_signals": signals,
                "status_records": records,
            }
        )
    return result


def validate_recovery(root: Path) -> dict[str, int]:
    """Verify recovered bytes, duplicate mappings, and transformed notebook extracts."""
    manifest = load(root / AUDIT / "coursework_manifest.json")
    seen = set()
    dispositions: Counter[str] = Counter()
    for row in manifest["files"]:
        if row["path"] in seen:
            raise ValueError(f"Duplicate coursework source: {row['path']}")
        seen.add(row["path"])
        dispositions[row["disposition"]] += 1
        relative = row["repository_path"]
        if relative is None:
            if row["disposition"] != "original_retained_locally_indexed":
                raise ValueError(f"Unrepresented coursework output: {row['path']}")
            continue
        target = local_path(root, relative)
        if not target.is_file():
            raise ValueError(f"Recovered evidence missing: {relative}")
        content = target.read_bytes()
        if row.get("repository_normalization") == "git_text_lf":
            content = content.replace(b"\r\n", b"\n")
        expected_size = row.get("repository_bytes", row["bytes"])
        expected_hash = row.get("repository_sha256", row["sha256"])
        if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"Recovered evidence hash mismatch: {relative}")
    if dict(dispositions) != manifest["counts"]:
        raise ValueError("Coursework disposition counts do not match the inventory")
    notebooks = load(root / AUDIT / "notebook_source_index.json")
    for notebook in notebooks["notebooks"]:
        path = local_path(root, notebook["extract_path"])
        if digest(path) != notebook["extract_sha256"]:
            raise ValueError(f"Notebook extract hash mismatch: {notebook['extract_path']}")
    return dict(dispositions)


def serialize_csv(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build_outputs(root: Path) -> dict[str, str]:
    catalog = load(root / CATALOG)
    entries = inspect_trials(root, catalog)
    recovery = validate_recovery(root)
    counts = Counter(entry["kind"] for entry in entries)
    document = [
        "# Experiment ledger",
        "",
        "Generated by `python tools/build_research_atlas.py`. This is an index of retained",
        "records, not a new scientific evaluation or a declaration that every run passed.",
        "",
        f"Coverage: **{counts['runs']} runs, {counts['reports']} reports, "
        f"{counts['failures']} separate failure package**. "
        "Failures also live inside run directories.",
        "",
        "Evidence scope is explicitly assigned in "
        "[trial_classification.json](trial_classification.json).",
        "A smoke gate remains a smoke result even when its stored gate says passed.",
        "Missing status is unknown; an audit-complete flag is not a scientific acceptance gate.",
        "Exact recorded status fields and their source hashes are in "
        "[experiment_inventory.json](experiment_inventory.json).",
        "The [CSV index](experiment_inventory.csv) supports filtering. "
        "See the [research atlas](../RESEARCH_ATLAS.md) for interpretation.",
        "",
    ]
    for family in sorted({entry["family"] for entry in entries}):
        document.extend(
            [
                f"## {family}",
                "",
                "| Artifact | Evidence scope | Recorded gate / failure | Interpretation |",
                "| --- | --- | --- | --- |",
            ]
        )
        for entry in entries:
            if entry["family"] != family:
                continue
            signals = (
                "; ".join(entry["recorded_signals"]) or "No explicit top-level gate / failure field"
            )
            signals = signals.replace("|", "\\|")
            document.append(
                f"| [{Path(entry['path']).name}](../../{entry['path']}) "
                f"({entry['kind']}) | {entry['scope']} | {signals} | "
                f"[{entry['note']}](../../{entry['reference']}) |"
            )
        document.append("")
    rows = [
        {
            "path": item["path"],
            "kind": item["kind"],
            "family": item["family"],
            "scope": item["scope"],
            "note": item["note"],
            "reference": item["reference"],
            "created_utc": item["created_utc"],
            "run_commit": item["run_commit"],
            "retained_file_count": item["retained_file_count"],
            "recorded_gate_or_failure": "; ".join(item["recorded_signals"]) or "not_recorded",
        }
        for item in entries
    ]
    payload = {
        "schema_version": "1.0.0",
        "interpretation": "inventory_not_new_evaluation",
        "counts": dict(counts),
        "coursework_recovery": recovery,
        "entries": entries,
    }
    return {
        "docs/research/EXPERIMENT_LEDGER.md": "\n".join(document),
        "docs/research/experiment_inventory.csv": serialize_csv(rows),
        "docs/research/experiment_inventory.json": json.dumps(payload, indent=2) + "\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    outputs = build_outputs(root)
    for relative, content in outputs.items():
        path = root / relative
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Stale generated research index: {relative}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    print("Research inventory, recovery hashes, and notebook extracts verified.")


if __name__ == "__main__":
    main()
