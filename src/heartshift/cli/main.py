"""Unified command-line interface for HeartShift contracts and release operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from heartshift import __version__
from heartshift.cli.validate import validate_repository
from heartshift.contracts import validate_contract_file, validate_contract_repository
from heartshift.data.acquisition import (
    acquire_dataset_manifest,
    verification_as_dict,
    verify_dataset_manifest,
)
from heartshift.registry import load_method_registry
from heartshift.release import (
    assemble_release_gate_report,
    create_release_inventory,
    dump_json,
    record_gate_status,
    validate_release_policy,
    write_candidate_scan,
)


def _path(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def _repo_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heartshift", description=__doc__)
    parser.add_argument("--version", action="version", version=f"HeartShift {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate data, studies, and contracts")
    _repo_argument(validate)

    contracts = commands.add_parser("contracts", help="inspect versioned JSON contracts")
    contract_commands = contracts.add_subparsers(dest="contract_command", required=True)
    contract_validate = contract_commands.add_parser("validate", help="validate contract records")
    _repo_argument(contract_validate)
    contract_validate.add_argument("--path", type=Path)

    methods = commands.add_parser("methods", help="inspect the typed method registry")
    method_commands = methods.add_subparsers(dest="method_command", required=True)
    for name in ("list", "validate"):
        method_parser = method_commands.add_parser(name)
        _repo_argument(method_parser)
    method_commands.choices["list"].add_argument("--json", action="store_true")

    data = commands.add_parser("data", help="verify or acquire manifest-bound datasets")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    for name in ("verify", "acquire"):
        data_parser = data_commands.add_parser(name)
        _repo_argument(data_parser)
        data_parser.add_argument("--manifest", type=Path, required=True)
    data_commands.choices["acquire"].add_argument("--allow-network", action="store_true")
    data_commands.choices["acquire"].add_argument("--receipt", type=Path)

    release = commands.add_parser("release", help="build exact-candidate release evidence")
    release_commands = release.add_subparsers(dest="release_command", required=True)
    scan = release_commands.add_parser("scan", help="scan an exact Git candidate tree")
    _repo_argument(scan)
    scan.add_argument("--candidate", required=True)
    scan.add_argument(
        "--policy", type=Path, default=Path("configs/release/release_gate_policy_v1.json")
    )
    scan.add_argument("--output", type=Path, required=True)

    record = release_commands.add_parser("record-gate", help="hash-bind one gate result")
    _repo_argument(record)
    record.add_argument("--candidate", required=True)
    record.add_argument("--gate", required=True)
    record.add_argument("--status", choices=("pass", "fail", "not_run"), required=True)
    record.add_argument("--evidence", type=Path)
    record.add_argument("--output", type=Path, required=True)

    assemble = release_commands.add_parser("assemble", help="assemble a release gate report")
    _repo_argument(assemble)
    assemble.add_argument("--candidate", required=True)
    assemble.add_argument("--version", required=True)
    assemble.add_argument(
        "--policy", type=Path, default=Path("configs/release/release_gate_policy_v1.json")
    )
    assemble.add_argument("--status-directory", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument(
        "--mode",
        choices=("candidate_attestation", "completed_remote_attestation"),
        default="candidate_attestation",
    )
    assemble.add_argument("--workflow-url")

    inventory = release_commands.add_parser("inventory", help="inventory a gated candidate")
    _repo_argument(inventory)
    inventory.add_argument("--candidate", required=True)
    inventory.add_argument("--version", required=True)
    inventory.add_argument("--tag", required=True)
    inventory.add_argument(
        "--policy", type=Path, default=Path("configs/release/release_gate_policy_v1.json")
    )
    inventory.add_argument("--gate-report", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    return parser


def _validate(repo_root: Path) -> None:
    validate_repository(repo_root)
    validate_contract_repository(repo_root)
    load_method_registry(repo_root)
    validate_release_policy(repo_root, repo_root / "configs/release/release_gate_policy_v1.json")
    for manifest in sorted((repo_root / "manifests/datasets").glob("*.json")):
        verify_dataset_manifest(repo_root, manifest)


def _run_contracts(args: argparse.Namespace, repo_root: Path) -> None:
    if args.path is None:
        paths = validate_contract_repository(repo_root)
    else:
        path = _path(repo_root, args.path)
        validate_contract_file(repo_root, path)
        paths = [path]
    print(json.dumps({"status": "pass", "validated": [str(path) for path in paths]}))


def _run_methods(args: argparse.Namespace, repo_root: Path) -> None:
    registry = load_method_registry(repo_root)
    if args.method_command == "validate":
        print(f"Method registry valid: {len(registry.methods)} entries")
        return
    rows = [
        {
            "method_id": method.method_id,
            "family": method.family,
            "version": method.version,
            "dependency_profile": method.dependency_profile,
            "evidence_eligibility": method.evidence_eligibility,
        }
        for method in registry.methods
    ]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(
                f"{row['method_id']:<34} {row['version']:<24} "
                f"{row['dependency_profile']:<18} {row['evidence_eligibility']}"
            )


def _run_data(args: argparse.Namespace, repo_root: Path) -> None:
    manifest = _path(repo_root, args.manifest)
    if args.data_command == "verify":
        results = verify_dataset_manifest(repo_root, manifest)
    else:
        receipt = _path(repo_root, args.receipt) if args.receipt is not None else None
        results = acquire_dataset_manifest(
            repo_root,
            manifest,
            allow_network=bool(args.allow_network),
            receipt_path=receipt,
        )
    print(json.dumps([verification_as_dict(result) for result in results], indent=2))


def _run_release(args: argparse.Namespace, repo_root: Path) -> None:
    if args.release_command == "scan":
        report = write_candidate_scan(
            repo_root,
            args.candidate,
            _path(repo_root, args.policy),
            _path(repo_root, args.output),
        )
    elif args.release_command == "record-gate":
        evidence = _path(repo_root, args.evidence) if args.evidence is not None else None
        report = record_gate_status(
            repo_root,
            candidate=args.candidate,
            gate=args.gate,
            status=args.status,
            evidence_path=evidence,
            output_path=_path(repo_root, args.output),
        )
    elif args.release_command == "assemble":
        report = assemble_release_gate_report(
            repo_root,
            candidate=args.candidate,
            version=args.version,
            policy_path=_path(repo_root, args.policy),
            status_directory=_path(repo_root, args.status_directory),
            output_path=_path(repo_root, args.output),
            mode=args.mode,
            workflow_url=args.workflow_url,
        )
    else:
        report = create_release_inventory(
            repo_root,
            candidate=args.candidate,
            version=args.version,
            tag=args.tag,
            policy_path=_path(repo_root, args.policy),
            gate_report_path=_path(repo_root, args.gate_report),
            output_path=_path(repo_root, args.output),
        )
    print(dump_json(report))


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    if args.command == "validate":
        _validate(repo_root)
        print("HeartShift data, study, and contract validation passed.")
    elif args.command == "contracts":
        _run_contracts(args, repo_root)
    elif args.command == "methods":
        _run_methods(args, repo_root)
    elif args.command == "data":
        _run_data(args, repo_root)
    else:
        _run_release(args, repo_root)


if __name__ == "__main__":
    main()
