# Adaptive RL Quantization with llama.cpp

**Upstream:** https://github.com/Legendarylibrorg/Adaptive-RL-Quantization

[![CI](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/actions/workflows/ci.yml/badge.svg)](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/actions/workflows/ci.yml) **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md) · **Code of Conduct:** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) · **Security:** [SECURITY.md](SECURITY.md)

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

**Where the boundary is**

Headline quantitative stories in **[docs/PAPER.md](docs/PAPER.md)** are written for the **simulator-first** evidence base (honest scope, reproducible budgets). The CUDA path is for real compute and host-grounded training; it does not, by itself, replace careful multi-machine measurement if you want deployment claims. This is **research infrastructure**, not a hosted inference API—but the **ambition** is production-shaped: multi-objective rewards, hardware-aware state, and analysis hooks you can extend rather than a one-off script.

| Mode | What you need |
| --- | --- |
| **Simulator** | **Python ≥ 3.11**, `pip install -e .` — **no** PyPI runtime deps, **no** CUDA |
| **PyTorch / CUDA** | Same repo + **CUDA-enabled PyTorch** on **Linux + NVIDIA** (recommended for that path) |

**Install:** `pip install -U pip` then **`pip install -e .`**. GPU training: **`pip install -e ".[torch]"`** or install a matching [torch](https://pytorch.org/get-started/locally/) wheel first, then **`pip install -e .`**.

Docs assume **Linux** (`bash`, `python3`, `venv`). The simulator runs on **macOS** too; GPU workflows target **Linux + NVIDIA** unless noted.

**Daily dev (optional):** `pip install -e ".[dev]"` then **`make help`** — Ruff lint/format + **`make check`** runs Ruff and the same script as CI (`pre_commit_check.sh`). See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Quick start (Linux)

**Requirements on PATH:** `git`, `curl`, and **Python ≥ 3.11** (Debian/Ubuntu: `sudo apt install -y git curl python3 python3-venv`). `curl` bootstraps `pip` inside a fresh venv when needed.

From the **repository root** after `git clone`:

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization

bash scripts/setup_from_clone.sh
```

That creates **`.venv`**, installs the package in editable mode, runs tests, and completes a **short reproducible end-to-end RL pipeline** (train → eval → benchmarks → analysis) using **`config.e2e_smoke.json`** (edit that file to change `training_episodes`, `seed`, `run_name`, etc.). For later commands in the same shell, activate the venv: `source .venv/bin/activate`.

**Manual equivalent:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
python3 -m unittest discover -s tests -q
python3 run_research.py --config config.e2e_smoke.json   # fast smoke
python3 run_research.py                                 # full run from config.py
```

Artifacts appear under `outputs/` (see below).

**GPU (Linux, after installing PyTorch for your driver):**

```bash
python3 -m pip install -e ".[torch]"   # or install a CUDA wheel from https://pytorch.org/get-started/locally/
python3 run_pytorch.py --preset gpu
```

**RTX 4090 validation + run (bash on Linux):**

```bash
bash scripts/run_4090_pipeline.sh
```

Uses **`.venv/bin/python`** when that venv exists and **`PYTHON_BIN`** is unset (same idea as **`setup_from_clone.sh`**).

Detailed install (distro packages, SSH clone, llama.cpp): **[docs/INSTALL.md](docs/INSTALL.md)**.

**Other OS:** On Windows, use `py -3.11` / `python` instead of `python3`, and `\.venv\Scripts\activate` instead of `source .venv/bin/activate`. GPU support is still oriented to Linux + NVIDIA.

---

## Repository layout

| Path | Role |
| --- | --- |
| `adaptive_quant/` | Core library: env, trainers, policies, backends, **`easy_config.py`** (JSON/TOML), **`runner_cli.py`** (shared `--config`) |
| `config.py`, `config_*.py` | Python experiment presets (imported by `run_*.py`) |
| `config.example.json` | Example **JSON** config (`preset` + overrides) |
| `config.e2e_smoke.json` | **Short reproducible RL run** (train+eval+benchmarks+analysis) for CI and quick tuning |
| `config.example.pytorch.toml` | Example **TOML** for `run_pytorch.py --config` (needs CUDA PyTorch) |
| `run_*.py` | CLI entrypoints — run from repo root |
| `Makefile` | **Research** targets: `make help` — `run` / `reproduce` (`smoke`) / `multiseed` / `pytorch`; quality: `lint` / `format` / `check` (Ruff needs `pip install -e ".[dev]"`) |
| `scripts/` | **`setup_from_clone.sh`**, **`pre_commit_check.sh`**, **`secret_scan.sh`** (heuristic grep, no extra deps), **`run_4090_pipeline.sh`**, **`_resolve_venv_python.sh`** (shared `.venv` Python pick when `PYTHON_BIN` unset) |
| `analysis/` | Post-hoc analysis CLIs |
| `docs/` | Install, running, config reference, troubleshooting |
| `CONTRIBUTING.md` | Contributing policy, PR expectations, local quality gate |
| `CODE_OF_CONDUCT.md` | Community standards ([Contributor Covenant](https://www.contributor-covenant.org/) 2.1) |
| `SECURITY.md` | Vulnerability reporting (private disclosure) |
| `.github/workflows/` | CI (Python 3.11/3.12, tests, E2E smoke) |
| `.github/ISSUE_TEMPLATE/` | Bug report and feature issue forms |
| `.github/PULL_REQUEST_TEMPLATE.md` | Default PR checklist |
| `tests/` | `unittest` suite (no GPU required) |

---

## Configuration

**1. Python presets** — Edit or copy `config.py`, `config_gpu.py`, `config_moe.py`, etc. This is the default path used when you do **not** pass `--config`.

**2. JSON / TOML** — Copy **`config.example.json`**, or write a `.toml` file with the same keys. Optional top-level **`preset`**: `default`, `minimal`, `pytorch`, `reproducible`. Load from the CLI:

```bash
python3 run_research.py --config my_settings.json
python3 run_research.py -c my_settings.toml
python3 run_pytorch.py --config cuda_run.toml    # replaces --preset
```

Programmatically: `FrameworkConfig.from_file("path.json")`, `load_config()` from `adaptive_quant`, or `FrameworkConfig.from_mapping({...})`. See **[docs/CONFIG.md](docs/CONFIG.md)**.

**3. Reproducible research preset** — `FrameworkConfig.reproducible_research(seed=...)` or JSON `"preset": "reproducible"` turns on sequential env sampling, deterministic train policy, deterministic stability probes, and PyTorch deterministic mode when applicable.

---

## Commands (run from repo root)

| Goal | Command |
| --- | --- |
| Default offline run (simulator, no PyTorch) | `python3 run_research.py` |
| Fast E2E RL smoke (edit `config.e2e_smoke.json`) | `python3 run_research.py --config config.e2e_smoke.json` |
| Same with your own file | `python3 run_research.py --config path.json` |
| MoE preset | `python3 run_moe_research.py` |
| NVIDIA GPU (auto VRAM profile) | `python3 run_pytorch.py --preset gpu` |
| RTX 4090 preset | `python3 run_pytorch.py --preset 4090` |
| 4090 checks + unittest + run | `bash scripts/run_4090_pipeline.sh` |
| Multi-seed aggregation | `python3 run_multiseed.py --preset dense --seeds 13,17,23` |
| Calibrate simulator from llama.cpp | `python3 run_calibrate_llama_cpp.py` (binary + model in config) |
| Online / continual experiment | `python3 run_online_learning.py` |

Full descriptions: **[docs/RUNNING.md](docs/RUNNING.md)**. Pass **`--help`** on any updated runner for `-c` / `--config`.

**Verify the install:**

```bash
python3 run_research.py --help
python3 run_pytorch.py --help
python3 -m unittest discover -s tests -q
```

---

## Outputs

Under **`outputs/`**:

- `logs/` — JSONL episodes
- `benchmarks/` — summaries, optional `*_preflight.json` (GPU)
- `analysis/<run_name>/` — JSON + figures
- `checkpoints/` — policy checkpoints (PyTorch)
- `reports/` — Markdown reports

Paths are driven by `run_name` and directory fields in config.

---

## Security

- **Secrets:** Do not commit API keys or `.env` files. `.gitignore` excludes `.env`, `.env.*`, `*.pem`, `*.key`, and `secrets/`. Use a local env file or your shell; optionally commit a redacted **`.env.example`** only.
- **Checkpoints:** Treat downloaded or third-party **`.pt` / pickle checkpoints** as **untrusted code** unless you saved them yourself. The default loader prefers **split checkpoints** with **`weights_only=True`**; legacy single-file loads require an explicit opt-in (`allow_legacy_checkpoint_load`).
- **Scans:** CI and **`pre_commit_check.sh`** run **`scripts/secret_scan.sh`** (high-signal `git grep` patterns on tracked files — lightweight, not exhaustive). Enable **GitHub secret scanning** on the org/repo if available; for deeper audits you can additionally run tools like [gitleaks](https://github.com/gitleaks/gitleaks) locally. **`SECURITY.md`** covers private reporting.

---

## Documentation index

| Doc | Contents |
| --- | --- |
| [docs/INSTALL.md](docs/INSTALL.md) | Linux packages, venv, optional `[torch]`, llama.cpp |
| [docs/RUNNING.md](docs/RUNNING.md) | Every entrypoint, examples, **Linux-first** |
| [docs/CONFIG.md](docs/CONFIG.md) | All settings + **JSON/TOML** + reproducibility |
| [docs/USAGE.md](docs/USAGE.md) | Artifacts, API, re-running analysis |
| [docs/GPU_PROFILES.md](docs/GPU_PROFILES.md) | VRAM / preset table |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | CUDA / preflight |
| [docs/ONLINE.md](docs/ONLINE.md) | Online loop |
| [docs/PAPER.md](docs/PAPER.md) | Research summary |
