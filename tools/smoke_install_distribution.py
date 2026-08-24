#!/usr/bin/env python3
"""Install the built wheel into an empty environment and inspect its package contract."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"distribution smoke test failed: {message}")


def run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=environment)


def interpreter_path(environment_root: Path) -> Path:
    if os.name == "nt":
        return environment_root / "Scripts" / "python.exe"
    return environment_root / "bin" / "python"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    wheels = sorted(args.dist_dir.resolve().glob("heartshift-*.whl"))
    require(len(wheels) == 1, f"expected one wheel, found {len(wheels)}")

    offline_environment = os.environ.copy()
    offline_environment.update(
        {
            "HEARTSHIFT_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "WANDB_DISABLED": "true",
        }
    )

    with tempfile.TemporaryDirectory(prefix="heartshift-wheel-") as temporary:
        environment_root = Path(temporary) / "environment"
        run(
            ["uv", "venv", "--python", sys.executable, str(environment_root)],
            environment=offline_environment,
        )
        interpreter = interpreter_path(environment_root)
        run(
            [
                "uv",
                "pip",
                "install",
                "--offline",
                "--python",
                str(interpreter),
                "--no-deps",
                str(wheels[0]),
            ],
            environment=offline_environment,
        )
        run(
            [
                str(interpreter),
                "-c",
                (
                    "import heartshift; "
                    "from importlib.metadata import entry_points, version; "
                    "assert heartshift.__version__ == version('heartshift'); "
                    "scripts = {point.name for point in entry_points(group='console_scripts')}; "
                    "assert 'heartshift-validate' in scripts; "
                    "assert 'heartshift-benchmark' in scripts"
                ),
            ],
            environment=offline_environment,
        )

    print(f"Clean-wheel smoke test passed for {wheels[0].name}.")


if __name__ == "__main__":
    main()
