# Adaptive RL Quantization with llama.cpp

[![CI](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/actions/workflows/ci.yml/badge.svg)](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/actions/workflows/ci.yml) **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md) · **Support:** [SUPPORT.md](SUPPORT.md) · **Changelog:** [CHANGELOG.md](CHANGELOG.md) · **Code of Conduct:** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) · **Security:** [SECURITY.md](SECURITY.md) · **Report a vulnerability:** [private advisory](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/security/advisories/new)

### What this pushes

Most deployments still treat quantization as a **one-time export**: pick a preset, ship it, hope it holds on every device and every prompt. This repo treats it as a **closed-loop control problem**: an agent observes **where the model runs** and **what it is asked to do**, then **acts**—bit widths, grouping, dynamic schedules, and (if you turn it on) **learned continuous controls** over scale, clip, and effective precision. The goal is not a single blessed `.gguf`; it is a **policy** you can train, evaluate, ablate, and (optionally) ground against a real **`llama.cpp`-class** binary.

**What you can stress-test here**

- **Universal vs narrow policies** — train across **GPU / CPU / low-resource** profiles so one controller sees more than a single silicon story.
- **Input adaptation** — prompt features drive different quantization behavior when complexity or sensitivity change.
- **Beyond discrete menus** — **learned** modes map into safe ranges instead of freezing behavior to a small set of named presets.
- **MoE serving realism** — packed expert **variants**, swap/cache/churn penalties: the action space looks more like **systems constraints** than a pure MAP benchmark.
- **Reward engineering** — latency, throughput, memory, perplexity, instability probes, MoE-specific terms: compose the objective, don’t hard-code one “score.”
- **Reproducibility knobs** — JSON/TOML configs, named presets, seeds, sequential sampling, deterministic train/eval modes when you need an audit trail.

**How it runs (one stack)**

Train → evaluate → benchmark suite → analysis artifacts under **`outputs/`** (JSON, JSONL, inline SVG, optional Markdown reports). **/stdlib simulator** is the default path (CI-friendly, no PyTorch). **Optional:** drive the same loop with a local **llama.cpp** binary + GGUF, or scale policy learning with **PyTorch + CUDA** (`run_pytorch.py` presets). Same `FrameworkConfig` surface either way.

Runs now also **detect the host hardware** (safe fallback to static defaults when probing is unavailable) and emit an **RL-backed quantization recommendation** for the detected target class under the benchmark artifacts.

**Where the boundary is**

Headline quantitative stories in **[docs/PAPER.md](docs/PAPER.md)** are written for the **simulator-first** evidence base (honest scope, reproducible budgets). The CUDA path is for real compute and host-grounded training; it does not, by itself, replace careful multi-machine measurement if you want deployment claims. This is **research infrastructure**, not a hosted inference API—but the **ambition** is production-shaped: multi-objective rewards, hardware-aware state, and analysis hooks you can extend rather than a one-off script.

| Mode | What you need |
| --- | --- |
| **Simulator** | **Python ≥ 3.11**, `python3 -m pip install -e .` — **no** PyPI runtime deps, **no** CUDA |
| **PyTorch / CUDA** | Same repo + **CUDA-enabled PyTorch** on **Linux + NVIDIA** (recommended for that path) |

**Install (after activating a venv):** **`python3 -m pip install -e .`**. GPU training: **`python3 -m pip install -e ".[torch]"`** or install a matching [torch](https://pytorch.org/get-started/locally/) wheel first, then **`python3 -m pip install -e .`**. On Windows, substitute `py -3.11 -m pip` or `python -m pip`.
Editable installs expose console commands: `adaptive-rl-quant`, `adaptive-rl-quant-moe`, `adaptive-rl-quant-pytorch`, `adaptive-rl-quant-online`, `adaptive-rl-quant-multiseed`, `adaptive-rl-quant-calibrate`, and `adaptive-rl-quant-route`.

The simulator path is supported on **Linux, macOS, and Windows**. GPU workflows still target **Linux + NVIDIA** unless noted, and **WSL2 is the recommended Windows path** when you want Linux-parity tooling and layout.

**Daily dev (optional):** `python3 -m pip install -e ".[dev]"` then **`make help`** on Linux/macOS, or run **`python3 scripts/pre_commit_check.py`** directly on Unix-like hosts (`py -3.11` / `python` on Windows). See [CONTRIBUTING.md](CONTRIBUTING.md).

**Dependency hardening:** CI bootstrap packages now live in **`requirements/ci.txt`** and are installed with **`pip --require-hashes`** after verification against the separate manifest in **`security/dependency_hashes.json`**. **Dependabot** watches both **`pyproject.toml`** and **`requirements/`**.

---

## Linux Quick Start

**Requirements on PATH:** `git` and **Python ≥ 3.11**. On Linux, `curl` is still useful for manual fallback bootstrapping on minimal systems.

From the **repository root** after `git clone`:

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization

python3 scripts/setup_from_clone.py
```

That creates **`.venv`**, installs the package in editable mode, runs tests, and completes a **short reproducible end-to-end RL pipeline** (train → eval → benchmarks → analysis) using **`config.e2e_smoke.json`** (edit that file to change `training_episodes`, `seed`, `run_name`, etc.). On Linux/macOS, `bash scripts/setup_from_clone.sh` remains a wrapper around the same Python flow.

**Manual equivalent:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m unittest discover -s tests -q
adaptive-rl-quant --config config.e2e_smoke.json   # fast smoke
adaptive-rl-quant                                 # full run from config.py
```

Installed console commands are the public interface shown throughout this README. Source-checkout equivalents remain `python3 run_research.py ...` if you prefer calling the repo files directly.

Artifacts appear under `outputs/` (see below).

**GPU (Linux, after installing PyTorch for your driver):**

```bash
python3 -m pip install -e ".[torch]"   # or install a CUDA wheel from https://pytorch.org/get-started/locally/
adaptive-rl-quant-pytorch --preset gpu
```

**RTX 4090 validation + run (bash on Linux):**

```bash
bash scripts/run_4090_pipeline.sh
```

Uses **`.venv/bin/python`** when that venv exists and **`PYTHON_BIN`** is unset (same idea as **`setup_from_clone.sh`**).

Detailed install (distro packages, **WSL2**, SSH clone, llama.cpp): **[docs/INSTALL.md](docs/INSTALL.md)**.

**Windows:** use `py -3.11 scripts/setup_from_clone.py` or `python scripts/setup_from_clone.py`, and activate with `\.venv\Scripts\activate`. Then prefer the same installed commands (`adaptive-rl-quant`, `adaptive-rl-quant-online`, and so on). GPU support is still oriented to Linux + NVIDIA.

**Architecture:** see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the repo layering, reproducibility contract, and Linux-first / WSL2 guidance.

---

## Repository layout

| Path | Role |
| --- | --- |
| `adaptive_quant/` | Core library: env, trainers, policies, backends, **`easy_config.py`** (JSON/TOML), **`runner_cli.py`** (shared `--config`) |
| `config.py`, `config_*.py` | Python experiment presets (imported by `run_*.py`) |
| `config.example.json` | Example **JSON** config (`preset` + overrides) |
| `config.e2e_smoke.json` | **Short reproducible RL run** (train+eval+benchmarks+analysis) for CI and quick tuning |
| `config.example.pytorch.toml` | Example **TOML** for `run_pytorch.py --config` (needs CUDA PyTorch) |
| `run_*.py` | Source-checkout equivalents to the installed console commands — run from repo root |
| `Makefile` | **Research** targets: `make help` — `run` / `reproduce` (`smoke`) / `multiseed` / `pytorch`; quality: `lint` / `format` / `check` (Ruff needs `pip install -e ".[dev]"`) |
| `scripts/` | Cross-platform **`setup_from_clone.py`**, **`pre_commit_check.py`**, **`secret_scan.py`** plus Unix wrappers (`*.sh`), **`run_4090_pipeline.sh`**, **`_resolve_venv_python.sh`** |
| `requirements/ci.txt` + `security/dependency_hashes.json` | Pinned CI bootstrap dependencies plus the separate sha256 manifest used to render a `--require-hashes` install file |
| `analysis/` | Post-hoc analysis CLIs |
| `docs/` | Install, running, config reference, troubleshooting |
| `CONTRIBUTING.md` | Contributing policy, PR expectations, local quality gate |
| `CHANGELOG.md`, `RELEASING.md` | Version history and release process |
| `CITATION.cff` | Software citation (for papers and “Cite this repository”) |
| `CODE_OF_CONDUCT.md` | Short rules for issues and pull requests |
| `SECURITY.md` | Vulnerability reporting (private disclosure, scope, SLAs, safe harbor) |
| `SUPPORT.md` | Where to ask for help and how to file a useful bug |
| `.well-known/security.txt` | Machine-readable disclosure metadata (RFC 9116) |
| `.github/workflows/` | CI (Linux on Python 3.11/3.12/3.13; macOS and Windows on 3.12; E2E smoke) |
| `.github/ISSUE_TEMPLATE/` | Bug report and feature issue forms |
| `.github/PULL_REQUEST_TEMPLATE.md` | Default PR checklist |
| `.github/CODEOWNERS` | Reviewer routing for security-sensitive paths |
| `tests/` | `unittest` suite (no GPU required) |

---

## Configuration

**1. Python presets** — Edit or copy `config.py`, `config_gpu.py`, `config_moe.py`, etc. This is the default path used when you do **not** pass `--config`.

**2. JSON / TOML** — Copy **`config.example.json`**, or write a `.toml` file with the same keys. Optional top-level **`preset`**: `default`, `minimal`, `pytorch`, `reproducible`. Load from the installed CLI:

```bash
adaptive-rl-quant --config my_settings.json
adaptive-rl-quant -c my_settings.toml
adaptive-rl-quant-pytorch --config cuda_run.toml    # replaces --preset
```

Source-checkout equivalents remain `python3 run_research.py --config ...` and `python3 run_pytorch.py --config ...`.

Programmatically: `FrameworkConfig.from_file("path.json")`, `load_config()` from `adaptive_quant`, or `FrameworkConfig.from_mapping({...})`. See **[docs/CONFIG.md](docs/CONFIG.md)**.

**3. Reproducible research preset** — `FrameworkConfig.reproducible_research(seed=...)` or JSON `"preset": "reproducible"` turns on sequential env sampling, deterministic train policy, deterministic stability probes, and PyTorch deterministic mode when applicable.

---

## Public Commands

| Goal | Command |
| --- | --- |
| Default offline run (simulator, no PyTorch) | `adaptive-rl-quant` |
| Fast E2E RL smoke (edit `config.e2e_smoke.json`) | `adaptive-rl-quant --config config.e2e_smoke.json` |
| Same with your own file | `adaptive-rl-quant --config path.json` |
| MoE preset | `adaptive-rl-quant-moe` |
| NVIDIA GPU (auto VRAM profile) | `adaptive-rl-quant-pytorch --preset gpu` |
| RTX 3090 preset | `adaptive-rl-quant-pytorch --preset 3090` (or `make 3090`) |
| RTX 4090 preset | `adaptive-rl-quant-pytorch --preset 4090` |
| 4090 checks + unittest + run | `bash scripts/run_4090_pipeline.sh` |
| Multi-seed aggregation | `adaptive-rl-quant-multiseed --preset dense --seeds 13,17,23` |
| Calibrate simulator from llama.cpp | `adaptive-rl-quant-calibrate` (binary + model in config) |
| GGUF route catalog + contextual bandit | `adaptive-rl-quant-route --catalog outputs/routes/catalog.json seed` |
| Online / continual experiment | `adaptive-rl-quant-online` |

Full descriptions: **[docs/RUNNING.md](docs/RUNNING.md)**. Pass **`--help`** on any installed command for `-c` / `--config`. Source-checkout equivalents remain available as `python3 run_*.py`.

**Verify the install:**

```bash
adaptive-rl-quant --help
adaptive-rl-quant-online --help
adaptive-rl-quant-pytorch --help
adaptive-rl-quant-multiseed --help
adaptive-rl-quant-calibrate --help
adaptive-rl-quant-route --help
python3 -m unittest discover -s tests -q
```

---

## Outputs

Under **`outputs/`**:

- `logs/` — JSONL episodes
- `benchmarks/` — summaries, optional `*_preflight.json` (GPU)
- `benchmarks/*_recommendation.json` — detected hardware + adaptive-policy summary + best fixed quant candidate sourced from RL rollouts
- `analysis/<run_name>/` — JSON + figures
- `checkpoints/` — policy checkpoints (PyTorch)
- `reports/` — Markdown reports
- `paper_bundles/<run_name>/` — manifest, metric CSV/JSON, flattened telemetry, appendix, and claims validation for citation/review

Paths are driven by `run_name` and directory fields in config.

---

## Security

- **Vulnerability reporting:** **Do not** open a public issue for security problems. Use the private [GitHub Security Advisories form](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/security/advisories/new) and follow [SECURITY.md](SECURITY.md). Machine-readable disclosure metadata lives at [`.well-known/security.txt`](.well-known/security.txt) (RFC 9116).
- **Secrets:** Do not commit API keys or `.env` files. `.gitignore` excludes `.env`, `.env.*`, `*.pem`, `*.key`, and `secrets/`. Use a local env file or your shell; optionally commit a redacted **`.env.example`** only.
- **Checkpoints:** Treat downloaded or third-party **`.pt` / pickle checkpoints** as **untrusted code**. The loader accepts current split checkpoints with **`weights_only=True`** when supported and refuses legacy pickle-only `.pt` checkpoints; convert old checkpoints only in a separate trusted environment.
- **Scans:** CI and **`pre_commit_check.py`** run **`scripts/secret_scan.py`** (high-signal tracked-file scan — lightweight, not exhaustive). Enable **GitHub secret scanning** on the org/repo if available; for deeper audits you can additionally run tools like [gitleaks](https://github.com/gitleaks/gitleaks) locally. **`SECURITY.md`** covers private reporting.
- **Dependency integrity:** CI verifies **`requirements/ci.txt`** against **`security/dependency_hashes.json`**, renders a temporary `--require-hashes` file, and installs only from that verified manifest. Run **`python3 scripts/verify_hashes.py`** locally when you change pinned CI packages.

---

## Documentation index

| Doc | Contents |
| --- | --- |
| [docs/INSTALL.md](docs/INSTALL.md) | Cross-platform venv setup, optional `[torch]`, llama.cpp |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Repo layering, artifact contract, Linux-first / WSL2 guidance |
| [docs/RUNNING.md](docs/RUNNING.md) | Every entrypoint, examples, OS notes |
| [docs/CONFIG.md](docs/CONFIG.md) | All settings + **JSON/TOML** + reproducibility |
| [docs/USAGE.md](docs/USAGE.md) | Artifacts, API, re-running analysis |
| [docs/GPU_PROFILES.md](docs/GPU_PROFILES.md) | VRAM / preset table |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | CUDA / preflight |
| [docs/ONLINE.md](docs/ONLINE.md) | Online loop |
| [docs/ROUTES.md](docs/ROUTES.md) | GGUF route catalogs and contextual route bandits |
| [docs/LOCAL_RESEARCH.md](docs/LOCAL_RESEARCH.md) | Local `llama.cpp` evidence and paper bundles |
| [docs/PAPER.md](docs/PAPER.md) | Research summary |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [RELEASING.md](RELEASING.md) | Tags and optional PyPI release |
| [CITATION.cff](CITATION.cff) | Citation metadata (GitHub “Cite this repository”) |
