# Locked Python dependencies

| File | Purpose |
| --- | --- |
| `ci.txt` + `../security/dependency_hashes.json` | Bootstrap (`setuptools`) for CI/Docker; verified by `scripts/verify_hashes.py` |
| `dev.txt` | Hash-pinned dev/CI tools (`ruff`, `mypy`, `coverage`, `pip-audit`, …) via `pip-compile` |
| `pytorch-cpu.txt` | Hash-pinned CPU `torch` stack for `INSTALL_EXTRAS=torch` Docker (Dependabot; not scanned in CI `pip-audit`) |

Regenerate lockfiles after changing pins in `pyproject.toml` or `*.in`:

```bash
# pip-tools path (Linux CI images) or uv path (macOS-friendly for pytorch-cpu.txt)
pip install pip-tools   # omit if you use uv for compile_locked_requirements.py
python scripts/compile_locked_requirements.py
python scripts/verify_lockfiles.py
python scripts/verify_hashes.py
```

On macOS, `scripts/compile_locked_requirements.py` uses `uv` when available so `pytorch-cpu.txt` resolves Linux CPU wheels; the lockfile keeps `--extra-index-url https://download.pytorch.org/whl/cpu` for `pip --require-hashes` installs.

Optional extras (`torch`, `router`) in `pyproject.toml` remain **minimum versions** for local `pip install -e ".[torch,router]"`; CI and Docker use the locked files above where possible.
