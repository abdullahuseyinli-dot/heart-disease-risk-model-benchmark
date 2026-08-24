from __future__ import annotations

import hashlib
import io
import json
import shutil
import urllib.request
from pathlib import Path

import pytest

from heartshift.contracts import bind_record_hash, validate_contract_file
from heartshift.data import acquisition
from heartshift.data.acquisition import (
    AcquisitionError,
    acquire_dataset_manifest,
    verify_dataset_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeResponse:
    status = 200

    def __init__(
        self,
        payload: bytes,
        *,
        final_url: str = "https://example.org/fixture.bin",
        headers: dict[str, str] | None = None,
    ) -> None:
        self._stream = io.BytesIO(payload)
        self._final_url = final_url
        self.headers = headers or {}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return self._final_url

    def read(self, size: int) -> bytes:
        return self._stream.read(size)


def _manifest_repository(tmp_path: Path, payload: bytes = b"verified bytes\n") -> tuple[Path, Path]:
    root = tmp_path / "repository"
    shutil.copytree(REPO_ROOT / "configs/schema", root / "configs/schema")
    manifest_path = root / "manifests/datasets/test_v1.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = bind_record_hash(
        {
            "schema_version": "1.0.0",
            "record_kind": "dataset_manifest",
            "dataset_id": "test-public-data",
            "title": "Test public data",
            "status": "available",
            "release": {
                "version": "1",
                "doi": "not-applicable",
                "landing_url": "https://example.org/dataset",
                "published_date": None,
            },
            "license": {
                "identifier": "CC0-1.0",
                "name": "CC0 1.0",
                "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "redistribution_policy": "redistributable_with_attribution",
            },
            "provenance": {
                "official_repository_url": "https://example.org/dataset",
                "retrieved_date": None,
                "hash_provenance": "Fixed synthetic fixture bytes.",
            },
            "artifacts": [
                {
                    "artifact_id": "fixture.bin",
                    "role": "test_fixture",
                    "source_url": "https://example.org/fixture.bin",
                    "storage_path": "data/raw/fixture.bin",
                    "expected_size_bytes": len(payload),
                    "expected_sha256": hashlib.sha256(payload).hexdigest(),
                    "hash_provenance": "Test fixture.",
                    "git_policy": "exclude_from_git",
                    "acquisition": "download",
                }
            ],
            "claim_scope": "Unit-test fixture only.",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest_path


def test_verify_reports_missing_and_rejects_mismatch(tmp_path: Path) -> None:
    root, manifest_path = _manifest_repository(tmp_path)
    missing = verify_dataset_manifest(root, manifest_path, require_present=False)
    assert missing[0].status == "missing"
    with pytest.raises(AcquisitionError, match="missing"):
        verify_dataset_manifest(root, manifest_path)

    raw = root / "data/raw/fixture.bin"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"wrong")
    with pytest.raises(AcquisitionError, match="mismatch"):
        verify_dataset_manifest(root, manifest_path)


def test_acquire_is_explicit_verified_and_create_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"verified bytes\n"
    root, manifest_path = _manifest_repository(tmp_path, payload)
    with pytest.raises(AcquisitionError, match="disabled"):
        acquire_dataset_manifest(root, manifest_path, allow_network=False)

    def fake_download(
        url: str,
        staged: Path,
        *,
        timeout_seconds: float,
        expected_size_bytes: int,
    ) -> None:
        assert url == "https://example.org/fixture.bin"
        assert timeout_seconds == 120.0
        assert expected_size_bytes == len(payload)
        staged.write_bytes(payload)

    monkeypatch.setattr(acquisition, "_download_to_staging", fake_download)
    receipt = root / ".audit/acquisition/receipt.json"
    results = acquire_dataset_manifest(
        root,
        manifest_path,
        allow_network=True,
        receipt_path=receipt,
    )
    assert results[0].status == "verified"
    assert (root / "data/raw/fixture.bin").read_bytes() == payload
    receipt_payload = validate_contract_file(root, receipt)
    assert receipt_payload["status"] == "complete"
    assert receipt_payload["network_explicitly_enabled"] is True

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        acquire_dataset_manifest(
            root,
            manifest_path,
            allow_network=False,
            receipt_path=receipt,
        )


def test_acquire_preserves_bad_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path = _manifest_repository(tmp_path)

    def corrupt_download(
        _url: str,
        staged: Path,
        *,
        timeout_seconds: float,
        expected_size_bytes: int,
    ) -> None:
        assert timeout_seconds > 0
        assert expected_size_bytes > 0
        staged.write_bytes(b"corrupt")

    monkeypatch.setattr(acquisition, "_download_to_staging", corrupt_download)
    receipt = root / ".audit/acquisition/failed.json"
    with pytest.raises(AcquisitionError, match="retained"):
        acquire_dataset_manifest(
            root,
            manifest_path,
            allow_network=True,
            receipt_path=receipt,
        )
    partials = list((root / "data/raw").glob(".fixture.bin.partial.*"))
    assert len(partials) == 1
    assert not (root / "data/raw/fixture.bin").exists()
    failure = validate_contract_file(root, receipt)
    assert failure["status"] == "failed"
    assert failure["failure"]["preserved_partial_path"].endswith(partials[0].name)


def test_existing_receipt_is_rejected_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path = _manifest_repository(tmp_path)
    receipt = root / ".audit/acquisition/existing.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("preserved\n", encoding="utf-8")
    called = False

    def unexpected_download(
        _url: str,
        _staged: Path,
        *,
        timeout_seconds: float,
        expected_size_bytes: int,
    ) -> None:
        del timeout_seconds, expected_size_bytes
        nonlocal called
        called = True

    monkeypatch.setattr(acquisition, "_download_to_staging", unexpected_download)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        acquire_dataset_manifest(
            root,
            manifest_path,
            allow_network=True,
            receipt_path=receipt,
        )
    assert called is False
    assert not (root / "data/raw/fixture.bin").exists()
    assert receipt.read_text(encoding="utf-8") == "preserved\n"


def test_download_stream_enforces_https_and_manifest_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(b"verified")

    def fake_urlopen(_request: object, *, timeout: float) -> _FakeResponse:
        assert timeout == 5.0
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    successful = tmp_path / "successful.partial"
    acquisition._download_to_staging(
        "https://example.org/fixture.bin",
        successful,
        timeout_seconds=5.0,
        expected_size_bytes=8,
    )
    assert successful.read_bytes() == b"verified"

    response = _FakeResponse(b"four")
    oversized = tmp_path / "oversized.partial"
    with pytest.raises(AcquisitionError, match="exceeds manifest size"):
        acquisition._download_to_staging(
            "https://example.org/fixture.bin",
            oversized,
            timeout_seconds=5.0,
            expected_size_bytes=3,
        )
    assert oversized.read_bytes() == b"four"

    response = _FakeResponse(b"data", final_url="http://example.org/fixture.bin")
    with pytest.raises(AcquisitionError, match="redirected outside HTTPS"):
        acquisition._download_to_staging(
            "https://example.org/fixture.bin",
            tmp_path / "redirect.partial",
            timeout_seconds=5.0,
            expected_size_bytes=4,
        )
