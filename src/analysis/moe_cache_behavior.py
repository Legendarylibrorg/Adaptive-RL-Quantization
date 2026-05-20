"""Post-hoc CLI for MoE cache behavior analysis (prefer ``python -m analysis``)."""

from __future__ import annotations

if __name__ == "__main__":
    import importlib.util
    from pathlib import Path

    _legacy = Path(__file__).resolve().parent / "_legacy_shim.py"
    _spec = importlib.util.spec_from_file_location("_analysis_legacy_shim", _legacy)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"Cannot load {_legacy}")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod.run_as_main(__file__)
