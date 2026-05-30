# Contributing policy

Thank you for helping improve this project. This document is the **contributing policy**: what we expect before you open a pull request, how we handle security and conduct, and how to make reviews straightforward.

**Quick links:** [Changelog](CHANGELOG.md) · [Releasing](RELEASING.md) · [Code of Conduct](CODE_OF_CONDUCT.md) · [Security disclosure](SECURITY.md) · [Support / how to ask for help](SUPPORT.md) · [README](README.md) · [CI workflow](.github/workflows/ci.yml)

---

## Code of conduct

Read **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)**. Report problems to maintainers in private. **Do not** use public issues for security—use [SECURITY.md](SECURITY.md).

---

## License and copyrights

By contributing, you agree that your contributions are licensed under the same terms as the project. This repository uses the **MIT License** ([LICENSE](LICENSE)). You retain your copyright; the license grants others permission to use your contribution under those terms.

---

## Security

- **Vulnerabilities:** Do **not** open a public issue. Use the private [GitHub Security Advisory form](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/security/advisories/new) and follow [SECURITY.md](SECURITY.md) (scope, response SLAs, safe harbor, coordinated disclosure).
- **Secrets:** Never commit API keys, tokens, passwords, or private model paths. Use `.env` (gitignored) or local JSON/TOML configs. See README **Security** notes.
- **Checkpoints:** Treat downloaded `.pt` / legacy pickle checkpoints as **untrusted** unless you created them locally.
- **Supply chain:** Keep Python dependency changes reviewable. CI bootstrap packages live in **[`requirements/ci.txt`](requirements/ci.txt)**, their sha256 values live separately in **[`security/dependency_hashes.json`](security/dependency_hashes.json)**, and **[`scripts/verify_hashes.py`](scripts/verify_hashes.py)** should stay green when those pins change.

---

## Development setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
python3 -m pip install -e .
python3 -m unittest discover -s tests -t . -q
```

**Without** `pip install -e .`, set `PYTHONPATH=src` (or `src` on `sys.path`) so imports resolve to `src/adaptive_quant` and `src/analysis`. CI always uses the editable install above.

Optional GPU work: `pip install -e ".[torch]"` or a CUDA-matched PyTorch wheel, then see [docs/INSTALL.md](docs/INSTALL.md).

**Simulator-only changes** must pass **without** PyTorch. Do not add imports that force `torch` on the default test path unless guarded (existing patterns in the codebase).

### High-grade local workflow (optional)

For a tighter loop on your machine:

1. `pip install -e ".[dev]"` — installs **[Ruff](https://docs.astral.sh/ruff/)** for lint + format (`pyproject.toml` → `[project.optional-dependencies] dev`).
2. **`make help`** on Linux/macOS — see [Makefile](Makefile): quality (`lint`, `format`, `check` = Ruff + `pre_commit_check.py`); experiments (`run`, `reproduce` / `smoke`, `multiseed`, `sweep`, `pytorch`, …). On Windows, use the Python scripts under `scripts/`.
3. [.vscode/extensions.json](.vscode/extensions.json) recommends the Python and Ruff extensions; [.editorconfig](.editorconfig) keeps basic spacing consistent.

CI installs hash-pinned dev tools from [`requirements/dev.txt`](requirements/dev.txt) (see [`requirements/README.md`](requirements/README.md)); locally, `pip install -e ".[dev]"` is still fine for Ruff and mypy. Mypy covers configuration, logging, easy_config, backends, `route_pipeline`, CLI, and `torch_trainer` (see `scripts/pre_commit_check.py`).

---

## Local quality gate (required before a PR)

From the repo root:

```bash
python3 scripts/pre_commit_check.py
```

This runs:

- `git diff --check` / staged check (whitespace, conflict markers)
- **[`scripts/secret_scan.py`](scripts/secret_scan.py)** — heuristic secret patterns on tracked files (git only, no third-party scanner)
- **[`scripts/verify_hashes.py`](scripts/verify_hashes.py)** — validates pinned CI dependency hashes against separate storage
- Python syntax (`compileall` on `adaptive_quant`, `analysis`; `py_compile` on root `*.py`)
- `bash -n` on `scripts/*.sh`
- Full **`unittest`** suite

If **`.venv`** exists and **`PYTHON_BIN`** is unset, the script uses the repo venv interpreter automatically (`.venv/bin/python` on Unix, `.venv\Scripts\python.exe` on Windows).

This script is **not** the third-party [pre-commit](https://pre-commit.com/) framework; it is the repository’s canonical gate and matches what CI runs.

**Optional Git hooks:** `pip install pre-commit && pre-commit install` using [`.pre-commit-config.yaml`](.pre-commit-config.yaml) (wraps the same `pre_commit_check.py` flow).

---

## Branching and pull requests

1. **Branch** from the default branch (`main`, or `master` on older forks). Use a short, descriptive branch name (e.g. `fix-jsonl-parse`, `docs-install-typo`).
2. **Keep PRs focused.** One coherent change is easier to review than unrelated refactors. Avoid “cleanup” mixed with feature work unless requested.
3. **Describe the PR.** Use the [pull request template](.github/PULL_REQUEST_TEMPLATE.md): problem, solution, testing, risk, linked issues.
4. **Green CI.** PRs should pass GitHub Actions on **Linux, macOS, and Windows** (see [.github/workflows/ci.yml](.github/workflows/ci.yml)): secret pattern scan, dependency hash verification, hash-verified bootstrap install, editable install, `pre_commit_check.py`, and a short E2E RL smoke (`config.e2e_smoke.json`). Pull requests also run dependency review via [.github/workflows/dependency-review.yml](.github/workflows/dependency-review.yml).
5. **Respond to review feedback.** Maintainers may request tests, docs, or scope changes before merge.

Force-pushing during review is fine once agreed; avoid rewriting history after merge.

---

## Code and architecture expectations

- **Match existing style** in `src/adaptive_quant/`: imports, typing, dataclasses, naming.
- **No unnecessary dependencies** for the simulator path. Core library and analysis should remain usable with the stdlib where that is already the design.
- **Prefer small, testable units**; add or extend **`tests/`** when behavior is non-trivial or regression-prone.
- **GPU / PyTorch paths:** guard heavy imports; follow patterns in `src/adaptive_quant/torch_*.py` and existing tests (skips when PyTorch is absent).

---

## Documentation policy

- **User-visible behavior changes** (CLI flags, config fields, defaults, install steps) should update **README** and/or **`docs/`** in the same PR when practical.
- **Purely internal refactors** do not require doc churn.
- **Docs should call out OS-specific steps** when behavior differs (for example simulator vs CUDA workflows).

---

## Issues

Use [GitHub Issues](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/issues) with the provided templates when possible:

- **Bug report** — reproduction, environment, expected vs actual.
- **Feature / research idea** — motivation, proposed direction, constraints.

For **upstream** alignment, note the repo’s relationship to the upstream project listed in [README.md](README.md).

---

## Maintainer notes (informational)

- **First / routine releases:** follow [`RELEASING.md`](RELEASING.md) (pre-release checklist, tag, GitHub Release body, security settings).
- **CI** uses `permissions: contents: read`, workflow concurrency, a Python version matrix, and hash-verified bootstrap dependencies from `requirements/ci.txt`.
- **E2E smoke** is intentionally short; full research budgets live in `config*.py` / JSON presets.
- **GPU pipelines** (e.g. `run_pytorch.py`, `scripts/run_4090_pipeline.sh`) are validated on appropriate hardware, not in default CI.

---

## Questions

If something in this policy is unclear, open a **documentation** issue or a PR that proposes a clarification. For behavior disputes unrelated to code, follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
