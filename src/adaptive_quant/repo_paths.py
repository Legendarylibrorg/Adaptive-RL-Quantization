"""Locate the repository root for optional native tooling (Rust CLI, scripts)."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_MARKERS = ("pyproject.toml", "rust/Cargo.toml")


def find_repo_root(*, start: Path | None = None) -> Path | None:
    """Return repo root when ``pyproject.toml`` and ``rust/Cargo.toml`` both exist."""
    env_root = os.environ.get("ADAPTIVE_RL_REPO_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _is_repo_root(candidate):
            return candidate

    seeds = [start, Path.cwd(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for seed in seeds:
        if seed is None:
            continue
        resolved = seed.expanduser().resolve()
        for parent in (resolved, *resolved.parents):
            if parent in seen:
                continue
            seen.add(parent)
            if _is_repo_root(parent):
                return parent
    return None


def _is_repo_root(path: Path) -> bool:
    return all((path / marker).is_file() for marker in _REPO_MARKERS)


def default_rust_binary_paths(repo_root: Path) -> tuple[Path, ...]:
    """Release-first search order under ``rust/target/``."""
    return (
        repo_root / "rust/target/release/adaptive-rl-quant-rust",
        repo_root / "rust/target/release/adaptive-rl-quant-rust.exe",
        repo_root / "rust/target/debug/adaptive-rl-quant-rust",
        repo_root / "rust/target/debug/adaptive-rl-quant-rust.exe",
    )


__all__ = ["default_rust_binary_paths", "find_repo_root"]
