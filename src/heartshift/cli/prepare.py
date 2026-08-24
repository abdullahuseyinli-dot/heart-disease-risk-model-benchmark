"""Prepare canonical UCI Heart data and immutable split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from heartshift.data.uci import prepare_uci_heart


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = prepare_uci_heart(args.repo_root.resolve())
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
