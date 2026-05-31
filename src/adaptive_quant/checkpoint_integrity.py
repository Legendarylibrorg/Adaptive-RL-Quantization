"""SHA-256 integrity tags for checkpoint sidecars (tamper detection on resume)."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from adaptive_quant.logging_utils import to_jsonable

INTEGRITY_FIELD = "integrity_sha256"
_SKIP_ENV = "ADAPTIVE_RL_SKIP_CHECKPOINT_INTEGRITY"
_REQUIRE_ENV = "ADAPTIVE_RL_REQUIRE_CHECKPOINT_INTEGRITY"


def skip_checkpoint_integrity_verification() -> bool:
    raw = os.environ.get(_SKIP_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def require_checkpoint_integrity_verification() -> bool:
    raw = os.environ.get(_REQUIRE_ENV, "").strip().lower()
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


def canonical_json_bytes(
    payload: Any,
    *,
    exclude_keys: Iterable[str] = (INTEGRITY_FIELD,),
) -> bytes:
    excluded = set(exclude_keys)
    safe = to_jsonable(payload)
    if isinstance(safe, Mapping):
        safe = {key: value for key, value in safe.items() if key not in excluded}
    return json.dumps(safe, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_canonical(
    payload: Any,
    *,
    exclude_keys: Iterable[str] = (INTEGRITY_FIELD,),
) -> str:
    return _sha256_bytes(canonical_json_bytes(payload, exclude_keys=exclude_keys))


def attach_dict_integrity(payload: dict[str, Any]) -> dict[str, Any]:
    stamped = dict(payload)
    stamped[INTEGRITY_FIELD] = sha256_canonical(stamped)
    return stamped


def verify_dict_integrity(payload: dict[str, Any], *, label: str) -> None:
    if skip_checkpoint_integrity_verification():
        return
    expected = payload.get(INTEGRITY_FIELD)
    if not expected:
        if require_checkpoint_integrity_verification():
            raise ValueError(
                f"{label}: missing {INTEGRITY_FIELD}; refusing to load checkpoint without "
                f"integrity tag (set {_REQUIRE_ENV}=0 or unset to allow legacy sidecars)."
            )
        return
    actual = sha256_canonical(payload)
    if str(expected) != actual:
        raise ValueError(
            f"{label}: checkpoint integrity mismatch (expected {expected!r}, computed {actual!r}). "
            "Refusing to load a tampered checkpoint."
        )


def attach_torch_sidecar_integrity(meta: dict[str, Any], pt_path: str | Path) -> dict[str, Any]:
    stamped = dict(meta)
    meta_digest = sha256_canonical(stamped)
    tensor_digest = sha256_file(pt_path)
    combined = f"{meta_digest}:{tensor_digest}".encode()
    stamped[INTEGRITY_FIELD] = _sha256_bytes(combined)
    return stamped


def verify_torch_sidecar_integrity(
    meta: dict[str, Any], pt_path: str | Path, *, label: str
) -> None:
    if skip_checkpoint_integrity_verification():
        return
    expected = meta.get(INTEGRITY_FIELD)
    if not expected:
        if require_checkpoint_integrity_verification():
            raise ValueError(
                f"{label}: missing {INTEGRITY_FIELD}; refusing to load checkpoint without "
                f"integrity tag (set {_REQUIRE_ENV}=0 or unset to allow legacy sidecars)."
            )
        return
    meta_digest = sha256_canonical(meta)
    tensor_digest = sha256_file(pt_path)
    combined = f"{meta_digest}:{tensor_digest}".encode()
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
    "require_checkpoint_integrity_verification",
    "sha256_canonical",
    "sha256_file",
    "skip_checkpoint_integrity_verification",
    "verify_dict_integrity",
    "verify_torch_sidecar_integrity",
]
