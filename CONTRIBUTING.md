# Contributing

## Before you open a PR

1. **Branch off `main`** with a short descriptive name.
2. From the repo root, run **`bash scripts/pre_commit_check.sh`** (whitespace, Python syntax, bash `shell` scripts, full `unittest` suite).
3. **Simulator-only** changes must pass without PyTorch. GPU paths are tested manually or on hosts with CUDA.

## Style

- Match existing imports, typing, and dataclass patterns in `adaptive_quant/`.
- Prefer focused diffs; avoid drive-by refactors outside the request.
- No secrets: use `.env` (gitignored) or local JSON/TOML; never commit credentials (see **Security** in [README.md](README.md)).

## Checks in CI

GitHub Actions runs the same **pre-commit** script plus a short **E2E smoke** run (`.github/workflows/ci.yml`). PRs should stay green on **Python 3.11 and 3.12**.
