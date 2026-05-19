# Locked Python dependencies

| File | Purpose |
| --- | --- |
| `ci.txt` + `../security/dependency_hashes.json` | Bootstrap (`setuptools`) for CI/Docker; verified by `scripts/verify_hashes.py` |
| `dev.txt` | Hash-pinned dev/CI tools (`ruff`, `mypy`, `coverage`, `pip-audit`, …) via `pip-compile` |
| `pytorch-cpu.txt` | Hash-pinned CPU `torch` stack for `pip-audit`, scheduled audit, and `INSTALL_EXTRAS=torch` Docker |

Regenerate lockfiles after changing pins in `pyproject.toml` or `*.in`:

```bash
pip install pip-tools
python scripts/compile_locked_requirements.py
python scripts/verify_lockfiles.py
```

Optional extras (`torch`, `router`) in `pyproject.toml` remain **minimum versions** for local `pip install -e ".[torch,router]"`; CI and Docker use the locked files above where possible.
