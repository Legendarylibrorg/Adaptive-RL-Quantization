# Releasing

## Pre-release checklist (do in order)

1. **Branch** — work on a PR or local branch; merge to `main` when ready (no open blockers).
2. **Python ≥ 3.11** — from repo root, with a venv:
   - `python3 -m pip install -e ".[dev]"`
   - `ruff check adaptive_quant analysis tests scripts "run_*.py" "config*.py"`
   - `python3 -m unittest discover -s tests -q`
3. **Optional smoke** — `python3 run_research.py --config config.e2e_smoke.json` (matches CI’s last step).
4. **Version files** — for `vX.Y.Z`, keep these aligned:
   - [`pyproject.toml`](pyproject.toml) → `version = "X.Y.Z"`
   - [`CITATION.cff`](CITATION.cff) → `version` and `date-released`
   - [`CHANGELOG.md`](CHANGELOG.md) → move bullets from `[Unreleased]` into a new `## [X.Y.Z] - YYYY-MM-DD` section; leave `[Unreleased]` empty or with “Nothing yet.”
5. **Commit** — one commit (or a small series) on `main` with the version + changelog.
6. **Tag** — from a **clean** tree (`git status` clean on `main`):
   - `git pull origin main`
   - `git tag -a vX.Y.Z -m "vX.Y.Z"` (add `-s` if you use GPG signing)
   - `git push origin vX.Y.Z`
7. **GitHub Release** — **Releases → Draft a new release**:
   - Choose the tag `vX.Y.Z`.
   - **Title:** `vX.Y.Z` (or `Release vX.Y.Z`).
   - **Description:** paste the section for that version from [`CHANGELOG.md`](CHANGELOG.md), or use the template below for **0.1.0**.
   - Publish the release.
8. **Repository settings** — **Settings → Security** → enable **private vulnerability reporting** if it is not already (see [`SECURITY.md`](SECURITY.md)).
9. **PyPI (optional)** — only if you publish the package:
   - `python3 -m pip install build twine`
   - `python3 -m build`
   - `python3 -m twine upload dist/*` (trusted publisher or API token; never commit secrets).

## GitHub release notes (v0.1.0) — copy below the line

--- COPY FROM NEXT LINE ---

## What’s in 0.1.0

Initial public release.

- Simulator-first RL quantization research loop (`adaptive-rl-quant` and related CLIs).
- Optional PyTorch/CUDA training, MoE, online loop, multiseed runs, and llama.cpp calibration.
- JSON/TOML `FrameworkConfig` loading, hash-verified CI bootstrap, and dependency review on pull requests.

**Full history:** [CHANGELOG.md](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/blob/v0.1.0/CHANGELOG.md)

--- END COPY ---

## Fast path (after checklist is green)

1. **Version** — bump `version` in [`pyproject.toml`](pyproject.toml), match [`CITATION.cff`](CITATION.cff), update [`CHANGELOG.md`](CHANGELOG.md).
2. **Tag** — `git tag -a vX.Y.Z -m "vX.Y.Z"` and `git push origin vX.Y.Z` (or create the tag from the GitHub release UI).
3. **PyPI (optional)** — `python3 -m build` then `twine upload dist/*` with a **trusted publisher** or token with least privilege; do not commit credentials.
4. **GitHub** — private vulnerability reporting under **Settings → Security** so [`SECURITY.md`](SECURITY.md) matches the UI.
