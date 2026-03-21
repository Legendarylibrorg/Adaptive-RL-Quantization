# Adaptive RL Quantization with llama.cpp

Adaptive RL Quantization is a research framework for:

- multi-hardware universal policy learning
- dynamic per-input quantization
- learned quantization functions
- hybrid quantization modes across `discrete`, `grouped`, `per_layer`, `dynamic`, and `learned`

The repository has two practical ways to run:

- `simulator` mode: pure-Python, no ML dependencies, works anywhere with Python 3.11+
- `pytorch` mode: optimized for CUDA GPUs, with auto-tuned profiles for cards like RTX 4070/4080/4090, RTX 3090, L4, A100, H100, and similar VRAM classes

## What you need

For the simulator path:

- Python 3.11 or newer

For the GPU PyTorch path:

- Python 3.11 or newer
- NVIDIA driver working on the host
- CUDA-enabled PyTorch installed for that machine

For real `llama.cpp` measurements:

- a built `llama.cpp` binary
- a model file to pass through the config

## Install

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Install the package in editable mode:

```bash
python3 -m pip install -e .
```

Notes:

- The default simulator workflow does not require `torch`.
- The GPU workflow does require CUDA-enabled PyTorch, but the exact install command depends on your CUDA and driver stack. Use the official PyTorch install selector for your machine, then verify with:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Fastest way to run

Run the simulator research workflow:

```bash
python3 run_research.py
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

Run the generic GPU workflow with auto-detected tuning:

```bash
python3 run_pytorch_gpu.py
```

Run the explicit RTX 4090 workflow:

```bash
python3 run_pytorch_4090.py
```

The GPU entrypoints do this automatically:

- checks that the PyTorch/CUDA path is usable
- selects a GPU tuning profile
- writes a preflight report
- trains the main policy
- runs the benchmark suite
- writes hardware, input-adaptation, and learned-parameter analyses

## Output files

Outputs are written under `outputs/`:

- `outputs/logs/*.jsonl`: episode-level traces
- `outputs/benchmarks/*.json`: run summaries, benchmark summaries, and GPU preflight reports
- `outputs/analysis/*`: JSON summaries and SVG figures

## Main files

- `config.py`: default simulator-first configuration
- `config_gpu.py`: generic CUDA/PyTorch configuration with auto GPU profile selection
- `config_4090.py`: CUDA/PyTorch configuration for RTX 4090-class hardware
- `run_research.py`: main simulator entrypoint
- `run_pytorch_gpu.py`: main CUDA entrypoint with auto-detected GPU profile
- `run_pytorch_4090.py`: main CUDA/4090 entrypoint
- `adaptive_quant/`: environment, policy, trainer, quantization, logging, benchmark, and preflight code
- `analysis/`: hardware generalization, input adaptation, and quant-function analysis modules
- `tests/`: standard-library unit tests

## Documentation

- [Installation Guide](/Users/devcomputer/Downloads/unsloth-main/rl%20quant/docs/INSTALL.md)
- [Running Guide](/Users/devcomputer/Downloads/unsloth-main/rl%20quant/docs/RUNNING.md)
- [Configuration Guide](/Users/devcomputer/Downloads/unsloth-main/rl%20quant/docs/CONFIG.md)
- [GPU Profiles Guide](/Users/devcomputer/Downloads/unsloth-main/rl%20quant/docs/GPU_PROFILES.md)
- [Paper Draft](/Users/devcomputer/Downloads/unsloth-main/rl%20quant/docs/PAPER.md)
- [Troubleshooting](/Users/devcomputer/Downloads/unsloth-main/rl%20quant/docs/TROUBLESHOOTING.md)

## Common commands

Simulator run:

```bash
python3 run_research.py
```

Generic GPU run:

```bash
python3 run_pytorch_gpu.py
```

4090-specific run:

```bash
python3 run_pytorch_4090.py
```

Tests:

```bash
python3 -m unittest discover -s tests -v
```

## Important behavior

- If `torch` is not installed, the GPU entrypoints exit immediately with a clear error.
- The GPU path uses a startup preflight benchmark before training.
- `run_pytorch_gpu.py` auto-selects a profile based on detected GPU name and memory.
- Benchmark runs use a smaller budget than the main GPU training run so they do not dominate runtime.
- Prompt feature caching and buffered logging are enabled in the GPU configs to reduce CPU and I/O overhead.
