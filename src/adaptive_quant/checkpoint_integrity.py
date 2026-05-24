"""SHA-256 integrity tags for checkpoint sidecars (tamper detection on resume)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

INTEGRITY_FIELD = "integrity_sha256"
_SKIP_ENV = "ADAPTIVE_RL_SKIP_CHECKPOINT_INTEGRITY"


def skip_checkpoint_integrity_verification() -> bool:
    raw = os.environ.get(_SKIP_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    body = {key: value for key, value in payload.items() if key != INTEGRITY_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def attach_dict_integrity(payload: dict[str, Any]) -> dict[str, Any]:
    stamped = dict(payload)
    stamped[INTEGRITY_FIELD] = _sha256_bytes(canonical_json_bytes(stamped))
    return stamped


def verify_dict_integrity(payload: dict[str, Any], *, label: str) -> None:
    if skip_checkpoint_integrity_verification():
        return
    expected = payload.get(INTEGRITY_FIELD)
    if not expected:
        return
    actual = _sha256_bytes(canonical_json_bytes(payload))
    if str(expected) != actual:
        raise ValueError(
            f"{label}: checkpoint integrity mismatch (expected {expected!r}, computed {actual!r}). "
            "Refusing to load a tampered checkpoint."
        )


def attach_torch_sidecar_integrity(meta: dict[str, Any], pt_path: str | Path) -> dict[str, Any]:
    stamped = dict(meta)
    meta_digest = _sha256_bytes(canonical_json_bytes(stamped))
    tensor_digest = sha256_file(pt_path)
    combined = f"{meta_digest}:{tensor_digest}".encode("utf-8")
    stamped[INTEGRITY_FIELD] = _sha256_bytes(combined)
    return stamped


def verify_torch_sidecar_integrity(meta: dict[str, Any], pt_path: str | Path, *, label: str) -> None:
    if skip_checkpoint_integrity_verification():
        return
    expected = meta.get(INTEGRITY_FIELD)
    if not expected:
        return
    meta_digest = _sha256_bytes(canonical_json_bytes(meta))
    tensor_digest = sha256_file(pt_path)
    combined = f"{meta_digest}:{tensor_digest}".encode("utf-8")
    actual = _sha256_bytes(combined)
    if str(expected) != actual:
        raise ValueError(
            f"{label}: checkpoint sidecar/tensor integrity mismatch. "
            "Refusing to load a tampered checkpoint."
        )


__all__ = [
    "INTEGRITY_FIELD",
    "attach_dict_integrity",
    "attach_torch_sidecar_integrity",
    "canonical_json_bytes",
    "sha256_file",
    "skip_checkpoint_integrity_verification",
    "verify_dict_integrity",
    "verify_torch_sidecar_integrity",
]
