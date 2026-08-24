#!/usr/bin/env python3
"""Validate that package archives contain software rather than research evidence."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

MAX_ARCHIVE_BYTES = 5 * 1024 * 1024
FORBIDDEN_TOP_LEVEL = {
    "artifacts",
    "data",
    "deployment_bundle",
    "results",
    "rtdl_checkpoints",
}
LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"distribution validation failed: {message}")


def validate_member_names(names: list[str], archive: Path) -> None:
    require(names, f"{archive.name} is empty")
    for name in names:
        path = PurePosixPath(name)
        parts = path.parts
        if len(parts) > 1 and parts[0].startswith("heartshift-"):
            parts = parts[1:]
        top_level = parts[0]
        require(
            top_level not in FORBIDDEN_TOP_LEVEL,
            f"{archive.name} contains research or local path: {name}",
        )
        require(
            not {".git", ".venv"}.intersection(parts),
            f"{archive.name} contains local tooling path: {name}",
        )
        require(top_level != "AGENTS.md", f"{archive.name} contains repository policy input")


def validate_wheel(path: Path) -> None:
    require(path.stat().st_size <= MAX_ARCHIVE_BYTES, f"{path.name} exceeds 5 MiB")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        validate_member_names(names, path)
        require("heartshift/py.typed" in names, "wheel is missing heartshift/py.typed")
        for info in archive.infolist():
            if info.is_dir():
                continue
            require(
                not archive.read(info).startswith(LFS_POINTER_HEADER),
                f"wheel contains a Git LFS pointer: {info.filename}",
            )


def validate_sdist(path: Path) -> None:
    require(path.stat().st_size <= MAX_ARCHIVE_BYTES, f"{path.name} exceeds 5 MiB")
    with tarfile.open(path, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        names = [member.name for member in members]
        validate_member_names(names, path)
        basenames = {PurePosixPath(name).name for name in names}
        for required in ("LICENSE", "README.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml"):
            require(required in basenames, f"sdist is missing {required}")
        for member in members:
            extracted = archive.extractfile(member)
            require(extracted is not None, f"could not read {member.name}")
            require(
                not extracted.read(len(LFS_POINTER_HEADER)).startswith(LFS_POINTER_HEADER),
                f"sdist contains a Git LFS pointer: {member.name}",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    wheels = sorted(args.dist_dir.glob("heartshift-*.whl"))
    sdists = sorted(args.dist_dir.glob("heartshift-*.tar.gz"))
    require(len(wheels) == 1, f"expected one wheel, found {len(wheels)}")
    require(len(sdists) == 1, f"expected one sdist, found {len(sdists)}")
    validate_wheel(wheels[0])
    validate_sdist(sdists[0])
    print("Distribution validation passed.")


if __name__ == "__main__":
    main()
