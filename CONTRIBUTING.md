# Contributing policy

Thank you for helping improve this project. This document is the **contributing policy**: what we expect before you open a pull request, how we handle security and conduct, and how to make reviews straightforward.

**Quick links:** [Code of Conduct](CODE_OF_CONDUCT.md) · [Security disclosure](SECURITY.md) · [README](README.md) · [CI workflow](.github/workflows/ci.yml)

---

## Code of conduct

All contributors are expected to follow the **[Code of Conduct](CODE_OF_CONDUCT.md)**. Unacceptable behavior may be reported to repository maintainers through appropriate **private** channels. **Do not** use public issues for security reports—use [SECURITY.md](SECURITY.md).

---

## License and copyrights

By contributing, you agree that your contributions are licensed under the same terms as the project. This repository uses the **MIT License** ([LICENSE](LICENSE)). You retain your copyright; the license grants others permission to use your contribution under those terms.

---

## Security

- **Vulnerabilities:** Do **not** open a public issue. Follow [SECURITY.md](SECURITY.md).
- **Secrets:** Never commit API keys, tokens, passwords, or private model paths. Use `.env` (gitignored) or local JSON/TOML configs. See README **Security** notes.
- **Checkpoints:** Treat downloaded `.pt` / legacy pickle checkpoints as **untrusted** unless you created them locally.
- **Supply chain:** Prefer minimal, reviewable changes to dependencies (`pyproject.toml`, GitHub Actions). When upgrading Actions or adding packages, prefer pins reviewers can verify (exact versions or commit SHAs for actions where practical).

---

## Development setup

From the repository root (Linux/macOS; Windows analogous):

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
python3 -m pip install -U pip
python3 -m pip install -e .
python3 -m unittest discover -s tests -q
```

Optional GPU work: `pip install -e ".[torch]"` or a CUDA-matched PyTorch wheel, then see [docs/INSTALL.md](docs/INSTALL.md).

**Simulator-only changes** must pass **without** PyTorch. Do not add imports that force `torch` on the default test path unless guarded (existing patterns in the codebase).

---

## Local quality gate (required before a PR)

From the repo root:

```bash
bash scripts/pre_commit_check.sh
```

This runs:

- `git diff --check` / staged check (whitespace, conflict markers)
- Python syntax (`compileall` on `adaptive_quant`, `analysis`; `py_compile` on root `*.py`)
- `bash -n` on `scripts/*.sh`
- Full **`unittest`** suite

If **`.venv`** exists and **`PYTHON_BIN`** is unset, the script uses **`.venv/bin/python`** (see [scripts/_resolve_venv_python.sh](scripts/_resolve_venv_python.sh)). In CI, **`PYTHON_BIN=python`** is set explicitly.

This script is **not** the third-party [pre-commit](https://pre-commit.com/) framework; it is the repository’s canonical gate and matches what CI runs.

**Optional Git hooks:** `pip install pre-commit && pre-commit install` using [`.pre-commit-config.yaml`](.pre-commit-config.yaml) (wraps the same `pre_commit_check.sh`).

---

## Branching and pull requests

1. **Branch** from the default branch (`main`, or `master` on older forks). Use a short, descriptive branch name (e.g. `fix-jsonl-parse`, `docs-install-typo`).
2. **Keep PRs focused.** One coherent change is easier to review than unrelated refactors. Avoid “cleanup” mixed with feature work unless requested.
3. **Describe the PR.** Use the [pull request template](.github/PULL_REQUEST_TEMPLATE.md): problem, solution, testing, risk, linked issues.
4. **Green CI.** PRs should pass GitHub Actions on **Python 3.11 and 3.12** (see [.github/workflows/ci.yml](.github/workflows/ci.yml)): editable install, `pre_commit_check.sh`, and a short E2E RL smoke (`config.e2e_smoke.json`).
5. **Respond to review feedback.** Maintainers may request tests, docs, or scope changes before merge.

Force-pushing during review is fine once agreed; avoid rewriting history after merge.

---

## Code and architecture expectations

- **Match existing style** in `adaptive_quant/`: imports, typing, dataclasses, naming.
- **No unnecessary dependencies** for the simulator path. Core library and analysis should remain usable with the stdlib where that is already the design.
- **Prefer small, testable units**; add or extend **`tests/`** when behavior is non-trivial or regression-prone.
- **GPU / PyTorch paths:** guard heavy imports; follow patterns in `adaptive_quant/torch_*.py` and existing tests (skips when PyTorch is absent).

---

## Documentation policy

- **User-visible behavior changes** (CLI flags, config fields, defaults, install steps) should update **README** and/or **`docs/`** in the same PR when practical.
- **Purely internal refactors** do not require doc churn.
- **Docs are Linux-first** unless the change is explicitly cross-platform.

---

## Issues

Use [GitHub Issues](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/issues) with the provided templates when possible:

- **Bug report** — reproduction, environment, expected vs actual.
- **Feature / research idea** — motivation, proposed direction, constraints.

For **upstream** alignment, note the repo’s relationship to the upstream project listed in [README.md](README.md).

---

## Maintainer notes (informational)

- **CI** uses `permissions: contents: read`, workflow concurrency, and a Python version matrix.
- **E2E smoke** is intentionally short; full research budgets live in `config*.py` / JSON presets.
- **GPU pipelines** (e.g. `run_pytorch.py`, `scripts/run_4090_pipeline.sh`) are validated on appropriate hardware, not in default CI.

---

## Questions

If something in this policy is unclear, open a **documentation** issue or a PR that proposes a clarification. For behavior disputes unrelated to code, follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
