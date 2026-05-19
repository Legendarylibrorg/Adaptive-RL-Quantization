# Locked Python dependencies

| File | Purpose |
| --- | --- |
| `ci.txt` + `../security/dependency_hashes.json` | Bootstrap (`setuptools`) for CI/Docker; verified by `scripts/verify_hashes.py` |
| `dev.txt` | Hash-pinned dev/CI tools (`ruff`, `mypy`, `coverage`, …) via `pip-compile` |
| `audit.txt` | Hash-pinned `pip-audit` for the CI audit job |
| `pytorch-cpu.txt` | Hash-pinned CPU `torch` stack for `pytorch-smoke` CI |

Regenerate lockfiles after changing pins in `pyproject.toml` or `*.in`:

```bash
pip install pip-tools
python scripts/compile_locked_requirements.py
```

Optional extras (`torch`, `router`) in `pyproject.toml` remain **minimum versions** for local `pip install -e ".[torch,router]"`; CI uses the locked files above for reproducible installs.
