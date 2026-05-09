#!/usr/bin/env python3
"""Environment summary for local debugging (`make doctor`)."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_src_on_path() -> None:
    src = _repo_root() / "src"
    for path in (src, _repo_root()):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def _git(repo: Path, *args: str) -> str | None:
    try:
        p = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
        return p.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
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


def _detect_wsl2() -> bool:
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl2" in version


def _platform_report(repo: Path) -> tuple[bool, bool]:
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    is_wsl2 = _detect_wsl2()
    on_windows_mount = repo.resolve().as_posix().startswith("/mnt/")

    print("== Platform ==")
    print(f"  system:      {system}")
    print(f"  release:     {release}")
    print(f"  machine:     {machine}")
    if is_wsl2:
        print("  wsl2:        yes")
        distro = os.environ.get("WSL_DISTRO_NAME")
        if distro:
            print(f"  distro:      {distro}")
        print("  guidance:    prefer keeping the repo inside the Linux filesystem (for example ~/src/...)")
        if on_windows_mount:
            print("  warning:     repo appears to be under /mnt/... which is slower for Python/env workloads")
    else:
        print("  wsl2:        no")
    return is_wsl2, on_windows_mount


def main() -> int:
    repo = _repo_root()
    is_wsl2, on_windows_mount = _platform_report(repo)
    print("== Python ==")
    print(f"  executable:  {sys.executable}")
    print(f"  version:     {sys.version.split()[0]}")

    print("== adaptive_quant ==")
    import_ok = False
    try:
        _ensure_src_on_path()
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
    if is_wsl2 and on_windows_mount:
        print("  WSL2:       move the repo under ~/... for better I/O and venv performance")
    print("  Simulator:  make reproduce   (or make run)")
    print("  CUDA:       make install-torch && make pytorch")
    print("  Quality:    make test && make check")
    return 0 if import_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
