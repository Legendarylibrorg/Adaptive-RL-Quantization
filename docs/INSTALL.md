# Installation Guide

This project supports two installation targets:

- simulator mode: no external ML libraries required
- CUDA GPU mode: CUDA-enabled PyTorch required

## 1. Base setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools
```

Install the package:

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

Install CUDA-enabled PyTorch on the target GPU host. The exact command depends on:

- your driver version
- your CUDA runtime
- the PyTorch build you want to use

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
python3 run_pytorch_gpu.py
```

Or, if you want the explicit 4090 preset:

```bash
python3 run_pytorch_4090.py
```

For a one-command 4090 validation and run:

```bash
bash scripts/run_4090_pipeline.sh
```

## 4. llama.cpp setup

If you want real `llama.cpp`-backed measurements instead of the simulator backend, you need:

- a built `llama.cpp` CLI binary
- a local model file

Then set these config values in [config.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config.py), [config_gpu.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_gpu.py), or [config_4090.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_4090.py):

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

Simulator only:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -e .
python3 run_research.py
```

Optional experimental online extension:

```bash
python3 run_online_learning.py
```

CUDA GPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -e .
# install CUDA-enabled PyTorch for your host
python3 run_pytorch_gpu.py
```

RTX 4090 preset:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -e .
# install CUDA-enabled PyTorch for your host
python3 run_pytorch_4090.py
```

RTX 4090 preset with smoke tests:

```bash
bash scripts/run_4090_pipeline.sh
```
