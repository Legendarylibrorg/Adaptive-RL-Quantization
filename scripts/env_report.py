#!/usr/bin/env python3
"""Environment summary for local debugging (`make doctor`)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _git(repo: Path, *args: str) -> str | None:
    try:
        p = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return p.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _output_stats(repo: Path) -> None:
    base = repo / "outputs"
    if not base.is_dir():
        print("  outputs/:     (directory missing — no local runs yet)")
        return
    for name in ("benchmarks", "logs", "analysis", "checkpoints", "reports"):
        d = base / name
        if not d.is_dir():
            print(f"  outputs/{name:<12} —")
            continue
        n = sum(1 for p in d.rglob("*") if p.is_file())
        print(f"  outputs/{name:<12} {n} file(s)")


def main() -> int:
    repo = _repo_root()
    print("== Python ==")
    print(f"  executable:  {sys.executable}")
    print(f"  version:     {sys.version.split()[0]}")

    print("== adaptive_quant ==")
    import_ok = False
    try:
        import adaptive_quant  # noqa: F401

        print("  import:      OK")
        import_ok = True
    except ImportError as e:
        print(f"  import:      FAILED ({e})")
        print("  hint:        run from repo root, or: pip install -e .")

    try:
        import importlib.metadata as md

        print(f"  pkg version: {md.version('adaptive-rl-quant')}")
    except Exception:
        print("  pkg version: (editable install without metadata — OK)")

    print("== PyTorch (optional) ==")
    try:
        import torch

        print(f"  torch:       {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"  cuda:        {cuda}")
        if cuda:
            print(f"  device[0]:   {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  torch:       not installed → pip install -e \".[torch]\" (CUDA wheel from pytorch.org)")

    print("== Ruff (optional) ==")
    try:
        import importlib.metadata as md

        print(f"  ruff:        {md.version('ruff')}")
    except Exception:
        print("  ruff:        not installed → pip install -e \".[dev]\"")

    print("== Git ==")
    head = _git(repo, "rev-parse", "--short", "HEAD")
    if head:
        dirty = _git(repo, "status", "--porcelain")
        state = "dirty" if (dirty or "").strip() else "clean"
        print(f"  HEAD:        {head} ({state})")
    else:
        print("  (not a git checkout or git unavailable)")

    print("== Run artifacts ==")
    _output_stats(repo)

    print("== Next steps ==")
    if not import_ok:
        print("  Fix import: pip install -e .")
    print("  Simulator:  make reproduce   (or make run)")
    print("  CUDA:       make install-torch && make pytorch")
    print("  Quality:    make test && make check")
    return 0 if import_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
