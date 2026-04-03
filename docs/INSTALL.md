# Installation Guide

This project supports two installation targets:

- simulator mode: no external ML libraries required
- CUDA GPU mode: CUDA-enabled PyTorch required

## OS support

This repo is designed to be **Linux-first** (especially for NVIDIA CUDA training), while the **simulator research path works on macOS** as well.

- **Linux**: recommended for GPU runs and for `scripts/run_4090_pipeline.sh`
- **macOS**: supported for simulator runs; CUDA/NVIDIA paths are not the primary target

Follow the sections below in order on a fresh machine: **get the code** → **Linux packages** (if on Linux) → **venv + editable install** → optional GPU / llama.cpp.

## 0. Prerequisites (before `git clone`)

On Linux you need **`git`**, **`curl`**, and **Python 3.11+** on your `PATH`. `curl` is required for the official **`get-pip.py`** bootstrap when a new venv has no `pip` (common on minimal Debian images).

Sanity check:

```bash
command -v git
command -v curl
python3 --version   # expect 3.11 or newer
```

**One-command bootstrap (Linux):** after cloning the repo, from the repo root run:

```bash
bash scripts/setup_from_clone.sh
```

This creates **`.venv`**, upgrades **`pip`** (using **`curl`** if the venv has no pip), runs **`pip install -e .`**, **`unittest`**, and a **short reproducible end-to-end RL run** (train → eval → benchmarks → analysis) via **`config.e2e_smoke.json`**. Install, tests, and smoke use **`.venv/bin/python` explicitly** (no reliance on activating the venv first). Edit that JSON to tune episode counts, `seed`, and `run_name` without touching Python.

Override paths if needed:

```bash
PYTHON_BIN=python3.12 VENV_DIR="$PWD/.venv" bash scripts/setup_from_clone.sh
```

**Quality gate (contributors):** run **`bash scripts/pre_commit_check.sh`** from the repo root before pushing (see **[CONTRIBUTING.md](../CONTRIBUTING.md)**). If **`.venv`** exists and **`PYTHON_BIN`** is unset, checks use **`.venv/bin/python`**; otherwise they use **`python3`**. Override either way with **`PYTHON_BIN`**. In CI, the workflow sets job-level **`PYTHON_BIN=python`** so install, pre-commit, and E2E smoke share the same **`setup-python`** interpreter.

## Get the code

Use HTTPS (works everywhere; no SSH keys required):

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization
```

SSH (if you use GitHub with SSH keys):

```bash
git clone git@github.com:Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization
```

Forks and mirrors: replace the URL with your `git remote` as needed; the rest of this guide stays the same.

## Linux: system packages

On a minimal Linux install, install `git`, `curl`, and a recent Python before creating the venv. `curl` is used by many install docs (e.g. bootstrap scripts) and is assumed available for the copy-paste flows below.

**Debian / Ubuntu:**

```bash
sudo apt update
sudo apt install -y git curl python3 python3-venv
```

**Fedora:**

```bash
sudo dnf install -y git curl python3 python3-venv
```

Verify:

```bash
git --version
curl --version
python3 --version
```

Python must report **3.11 or newer**.

## 1. Base setup

Requirements:

- Python 3.11+
- `git` (to clone this repository)
- `curl` (recommended on Linux; used below if `pip` is missing)

The package declares **`dependencies = []`** in **[`pyproject.toml`](../pyproject.toml)** (stdlib only on Python 3.11+). The only declared optional extra is **`torch`** (`pip install -e ".[torch]"`); see GPU section.

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
```

If `python3 -m pip` fails because `pip` is not installed in the venv, bootstrap it (official PyPA script, fetched with `curl`):

```bash
curl -sS https://bootstrap.pypa.io/get-pip.py | python3
python3 -m pip install -U pip
```

Install the package (editable):

```bash
python3 -m pip install -e .
```

This is enough for:

- `python3 run_research.py`
- `python3 -m unittest discover -s tests -v`

## 2. Simulator-only setup

No extra dependencies are required beyond Python 3.11+ and the editable install above.

Recommended verification:

```bash
python3 run_research.py
python3 -m unittest discover -s tests -v
```

## 3. GPU setup

Optional: install the declared PyTorch extra (still pick a CUDA wheel that matches your driver):

```bash
python3 -m pip install -e ".[torch]"
```

Or install CUDA-enabled PyTorch manually on the target GPU host. The exact command depends on:

- your driver version
- your CUDA runtime
- the PyTorch build you want to use

Linux NVIDIA sanity checks:

```bash
nvidia-smi
python3 -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

After installation, verify:

```bash
python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
python3 -c "import torch; print('device_count', torch.cuda.device_count())"
```

Optional deeper verification:

```bash
python3 -c "import torch; print('bf16', getattr(torch.cuda, 'is_bf16_supported', lambda: False)())"
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

Then run (unified CLI or legacy wrappers):

```bash
python3 run_pytorch.py --preset gpu
# or: python3 run_pytorch_gpu.py
```

4090 preset:

```bash
python3 run_pytorch.py --preset 4090
# or: python3 run_pytorch_4090.py
```

For a one-command 4090 validation and run:

```bash
bash scripts/run_4090_pipeline.sh
```

If **`.venv`** already exists (for example after **`setup_from_clone.sh`**), the script picks **`.venv/bin/python`** automatically unless you set **`PYTHON_BIN`**. Override: `PYTHON_BIN=/usr/bin/python3 bash scripts/run_4090_pipeline.sh`.

## 4. llama.cpp setup

If you want real `llama.cpp`-backed measurements instead of the simulator backend, you need:

- a built `llama.cpp` CLI binary
- a local model file

Then set these config values in [`config.py`](../config.py), [`config_gpu.py`](../config_gpu.py), or [`config_4090.py`](../config_4090.py):

- `backend="llama_cpp"`
- `llama_cpp_binary="/absolute/path/to/llama-cli-or-equivalent"`
- `llama_cpp_model="/absolute/path/to/model.gguf"`

Useful fields:

- `llama_cpp_threads`
- `llama_cpp_context`

## 5. What is intentionally not pinned here

This repository does not pin the CUDA PyTorch wheel directly in `pyproject.toml` because the correct wheel depends on the machine where you run it.

That means:

- simulator mode is easy to install everywhere
- CUDA mode stays explicit and machine-specific

## 6. Quick install matrix

**Linux — simulator (full copy-paste from clone):**

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization
sudo apt update && sudo apt install -y git curl python3 python3-venv   # Debian/Ubuntu; skip if already installed
bash scripts/setup_from_clone.sh
source .venv/bin/activate
python3 run_research.py   # full baseline from config.py (smoke already ran during the script)
```

**Linux — CUDA GPU (after editable install; install PyTorch for your driver):**

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
# Install CUDA-enabled PyTorch: use https://pytorch.org/get-started/locally/ and copy the `pip` line, or:
# python3 -m pip install -e ".[torch]"
python3 run_pytorch_gpu.py
```

macOS — simulator only (same as Linux clone path; no `apt` step):

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
python3 run_research.py
```

Optional experimental online extension:

```bash
python3 run_online_learning.py
```

RTX 4090 preset (Linux NVIDIA host, after PyTorch is installed):

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
python3 run_pytorch_4090.py
```

RTX 4090 preset with smoke tests:

```bash
bash scripts/run_4090_pipeline.sh
```

## Next steps

Repository overview and command table: [README.md](../README.md). Usage and config files: [USAGE.md](USAGE.md) · [RUNNING.md](RUNNING.md) · [CONFIG.md](CONFIG.md).
