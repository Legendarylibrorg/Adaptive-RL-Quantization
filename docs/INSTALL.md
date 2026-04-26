# Installation Guide

This project supports two installation targets:

- simulator mode: no external ML libraries required
- CUDA GPU mode: CUDA-enabled PyTorch required

## OS support

This repo is designed to be **cross-platform for the simulator path**, while **Linux + NVIDIA** remains the primary target for CUDA training.

- **Linux**: supported for simulator and GPU runs; required for `scripts/run_4090_pipeline.sh`
- **macOS**: supported for simulator runs
- **Windows**: supported for simulator runs through the installed console commands plus the Python setup/check scripts
- **WSL2**: recommended Windows path for Linux-parity development and the preferred route for Windows-hosted GPU workflows

Follow the sections below in order on a fresh machine: **get the code** → **platform packages** (if needed) → **venv + editable install** → optional GPU / llama.cpp.

## Recommended path (Linux / WSL2 first)

After cloning the repo, the shortest supported setup is:

```bash
python3 scripts/setup_from_clone.py
source .venv/bin/activate
adaptive-rl-quant --config config.e2e_smoke.json
adaptive-rl-quant
```

That path is the primary workflow in this guide. macOS and Windows notes follow after the Linux-first sections.

## 0. Prerequisites (before `git clone`)

You need **`git`** and **Python 3.11+** on your `PATH`. On Linux, `curl` is still helpful for manual fallback bootstrapping on minimal images.

Sanity check:

```bash
command -v git
command -v curl
python3 --version   # expect 3.11 or newer
```

**One-command bootstrap:** after cloning the repo, from the repo root run:

```bash
python3 scripts/setup_from_clone.py
```

On Windows, use `py -3.11 scripts/setup_from_clone.py` or `python scripts/setup_from_clone.py`.

This creates **`.venv`**, upgrades **`pip`** (using `ensurepip` first and falling back to `get-pip.py` only if needed), runs **`pip install -e .`**, **`unittest`**, and a **short reproducible end-to-end RL run** (train → eval → benchmarks → analysis) via **`config.e2e_smoke.json`**. Install, tests, and smoke use the venv interpreter directly (no reliance on activating the venv first). Edit that JSON to tune episode counts, `seed`, and `run_name` without touching Python.

On Linux/macOS, `bash scripts/setup_from_clone.sh` remains available as a thin wrapper around the same Python entrypoint.

Override paths if needed:

```bash
python3 scripts/setup_from_clone.py --venv-dir .venv --config config.e2e_smoke.json
```

**Quality gate (contributors):** run **`python3 scripts/pre_commit_check.py`** from the repo root before pushing (see **[CONTRIBUTING.md](../CONTRIBUTING.md)**). On Linux/macOS, `bash scripts/pre_commit_check.sh` remains a wrapper around the same Python implementation. On Windows use `py -3.11 scripts/pre_commit_check.py` or `python scripts/pre_commit_check.py`.

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

**Fedora / RHEL / CentOS Stream / Rocky / AlmaLinux:**

```bash
sudo dnf install -y git curl python3
# `python3 -m venv` is part of the base python3 package on dnf-based distros.
```

**openSUSE / SUSE Linux Enterprise:**

```bash
sudo zypper install -y git curl python3 python3-pip
```

**Arch / Manjaro:**

```bash
sudo pacman -Syu --needed git curl python
# `python` on Arch is Python 3; venv is bundled with the python package.
```

**Alpine (musl):**

```bash
sudo apk add git curl python3 py3-pip bash
# bash is only required if you use the optional .sh wrappers under scripts/.
# The Python entry points (`python3 scripts/setup_from_clone.py`) work without bash.
```

**NixOS / nix-env:**

```bash
nix-shell -p git curl python311 python311Packages.pip --run "python3 scripts/setup_from_clone.py"
```

The Python scripts under `scripts/` (`setup_from_clone.py`, `pre_commit_check.py`, `secret_scan.py`, `verify_hashes.py`, `env_report.py`) only require **`python3 >= 3.11`** and **`git`**. The `.sh` wrappers add bash convenience for Linux/macOS shells but are not required: every `.sh` wrapper just `exec`s the matching `.py` file, so you can always invoke the Python script directly on minimal images (e.g. distroless, Alpine without bash).

Verify:

```bash
git --version
curl --version
python3 --version
```

Python must report **3.11 or newer**.

## WSL2 (recommended on Windows)

If you are on Windows and want the Linux-first workflow, use **WSL2** and follow the Linux commands from inside the Linux shell.

Recommended setup:

1. Install WSL with an Ubuntu distro from an elevated PowerShell prompt.
2. Open the Ubuntu shell and install the base Linux packages from the next section.
3. Keep the repo inside the Linux filesystem, for example `~/src/Adaptive-RL-Quantization`, not under `/mnt/c/...`.
4. Run the standard Linux bootstrap:

```bash
python3 scripts/setup_from_clone.py
```

GPU note:

- for simulator work, WSL2 behaves like the Linux path in this repo
- for GPU work, verify that `nvidia-smi` works inside WSL2 before installing PyTorch
- prefer the Linux shell, Linux venv, and Linux file paths throughout the workflow

Quick WSL2 sanity checks:

```bash
uname -a
python3 --version
git --version
```

If you expect GPU access:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

## 1. Base setup

Requirements:

- Python 3.11+
- `git` (to clone this repository)
- `curl` (optional manual fallback on Linux)

The package declares **`dependencies = []`** in **[`pyproject.toml`](../pyproject.toml)** (stdlib only on Python 3.11+). The only declared optional extra is **`torch`** (`pip install -e ".[torch]"`); see GPU section.

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
```

If `python3 -m pip` fails because `pip` is not installed in the venv, bootstrap it with `ensurepip` first:

```bash
python3 -m ensurepip --upgrade
python3 -m pip install -U pip
```

If `ensurepip` is unavailable on your platform build, `python3 scripts/setup_from_clone.py` will fall back automatically.

Install the package (editable):

```bash
python3 -m pip install -e .
```

This exposes installable console commands such as `adaptive-rl-quant`, `adaptive-rl-quant-pytorch`, `adaptive-rl-quant-online`, and `adaptive-rl-quant-multiseed`.

This is enough for:

- `adaptive-rl-quant`
- `adaptive-rl-quant --config config.e2e_smoke.json`
- `python3 -m unittest discover -s tests -v`

## 2. Simulator-only setup

No extra dependencies are required beyond Python 3.11+ and the editable install above.

Recommended verification:

```bash
adaptive-rl-quant
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

Then run:

```bash
adaptive-rl-quant-pytorch --preset gpu
```

4090 preset:

```bash
adaptive-rl-quant-pytorch --preset 4090
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
python3 scripts/setup_from_clone.py
source .venv/bin/activate
adaptive-rl-quant         # full baseline from config.py (smoke already ran during the script)
```

**WSL2 — simulator (recommended Windows workflow):**

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git ~/src/Adaptive-RL-Quantization
cd ~/src/Adaptive-RL-Quantization
sudo apt update && sudo apt install -y git curl python3 python3-venv
python3 scripts/setup_from_clone.py
source .venv/bin/activate
adaptive-rl-quant
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
adaptive-rl-quant-pytorch --preset gpu
```

**WSL2 — CUDA GPU (Windows host, Linux shell, NVIDIA passthrough required):**

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git ~/src/Adaptive-RL-Quantization
cd ~/src/Adaptive-RL-Quantization
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
nvidia-smi
# Install the CUDA-enabled PyTorch wheel that matches the WSL2 environment, then:
adaptive-rl-quant-pytorch --preset gpu
```

macOS — simulator only:

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
adaptive-rl-quant
```

**Windows — simulator only:**

```powershell
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization
py -3.11 scripts/setup_from_clone.py
.venv\Scripts\activate
adaptive-rl-quant
```

Online adaptation path:

```bash
adaptive-rl-quant-online
```

RTX 4090 preset (Linux NVIDIA host, after PyTorch is installed):

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
adaptive-rl-quant-pytorch --preset 4090
```

RTX 4090 preset with smoke tests:

```bash
bash scripts/run_4090_pipeline.sh
```

## Next steps

Repository overview and command table: [README.md](../README.md). Usage and config files: [USAGE.md](USAGE.md) · [RUNNING.md](RUNNING.md) · [CONFIG.md](CONFIG.md).
