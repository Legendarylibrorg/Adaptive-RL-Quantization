"""Structured security provenance for pipeline summaries and online telemetry."""

from __future__ import annotations

import os
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.configuration.validation import (
    _HF_ALLOWED_REPOS_ENV,
    _LLAMA_CPP_BINARY_PREFIXES_ENV,
    hf_allow_unlisted_from_env,
    hf_allowed_repos_from_env,
)


def build_security_audit_record(config: FrameworkConfig) -> dict[str, Any]:
    record: dict[str, Any] = {
        "backend": config.backend,
        "training_backend": config.training_backend,
        "hf_allow_unlisted": hf_allow_unlisted_from_env(),
        "hf_allowed_repos_env_set": bool(os.environ.get(_HF_ALLOWED_REPOS_ENV, "").strip()),
        "hf_repo_allowlist": sorted(
            hf_allowed_repos_from_env() | frozenset(config.route_hf_allowed_repos)
        ),
        "router_hf_allowed_models": list(config.router_hf_allowed_models),
        "router_hf_embedding_revision": config.router_hf_embedding_revision,
        "llama_cpp_binary_prefixes_env_set": bool(
            os.environ.get(_LLAMA_CPP_BINARY_PREFIXES_ENV, "").strip()
        ),
    }
    binary = config.llama_cpp_binary
    if binary:
        try:
            record["llama_cpp_binary"] = os.path.realpath(str(binary))
        except OSError:
            record["llama_cpp_binary"] = str(binary)
    model = config.llama_cpp_model
    if model:
        try:
            record["llama_cpp_model"] = os.path.realpath(str(model))
        except OSError:
            record["llama_cpp_model"] = str(model)
    if config.router_enabled:
        record["router_routes"] = list(config.router_routes)
        record["router_feature_backend"] = config.router_feature_backend
    return record


__all__ = ["build_security_audit_record"]
