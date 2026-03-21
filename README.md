# Adaptive RL Quantization with llama.cpp

Adaptive RL Quantization is a research framework for:

- multi-hardware universal policy learning
- dynamic per-input quantization
- learned quantization functions
- hybrid quantization modes across `discrete`, `grouped`, `per_layer`, `dynamic`, and `learned`

The repository has two main research paths:

- `simulator` mode: pure-Python, no ML dependencies, works anywhere with Python 3.11+
- `pytorch` mode: optimized for CUDA GPUs, with auto-tuned profiles for cards like RTX 4070/4080/4090, RTX 3090, L4, A100, H100, and similar VRAM classes

There is also an optional experimental extension:

- `online` mode: simulator-first continual adaptation loop with live telemetry, replay, canary checks, and rollback guards. This is useful for systems exploration, but the main research story and paper draft are centered on offline, reproducible training and evaluation.

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
python3 -m pip install --upgrade pip setuptools
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

## Linux RTX 4090 Quickstart

On a Linux 4090 host, the shortest reliable path is:

1. Create a clean virtual environment and install the repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -e .
```

2. Install a CUDA-enabled PyTorch build for that machine, then verify:

```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
python3 -c "import torch; print(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())"
```

3. Confirm the active CUDA device is really the 4090:

- `torch.cuda.is_available()` should be `True`
- the device name should include `4090`

4. Run the end-to-end 4090 pipeline:

```bash
bash scripts/run_4090_pipeline.sh
```

This script:

- checks that PyTorch is installed
- checks that CUDA is visible
- prints the active GPU name and memory
- optionally runs the test suite
- launches the full 4090 research pipeline

5. After the run, inspect the main artifacts:

- `outputs/benchmarks/adaptive_universal_policy_torch4090_preflight.json`
- `outputs/benchmarks/adaptive_universal_policy_torch4090_summary.json`
- `outputs/benchmarks/adaptive_universal_policy_torch4090_training_history.json`
- `outputs/checkpoints/adaptive_universal_policy_torch4090_final.pt`
- `outputs/reports/adaptive_universal_policy_torch4090_report.md`

If it fails:

- `PyTorch is not installed`: install a CUDA-enabled PyTorch build
- `CUDA is not available`: fix the NVIDIA driver / PyTorch CUDA mismatch
- wrong GPU detected: make sure the intended 4090 is the active CUDA device
- low free memory: close other GPU jobs and rerun

## Fastest way to run

Run the main simulator research workflow:

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

Optional experimental extension:

```bash
python3 run_online_learning.py
```

The GPU entrypoints do this automatically:

- checks that the PyTorch/CUDA path is usable
- selects a GPU tuning profile
- writes a preflight report
- trains the main policy
- runs the benchmark suite
- writes hardware, input-adaptation, and learned-parameter analyses
- writes training-history, checkpoint, and markdown report artifacts through the research pipeline

## Output files

Outputs are written under `outputs/`:

- `outputs/logs/*.jsonl`: episode-level traces
- `outputs/benchmarks/*.json`: run summaries, benchmark summaries, and GPU preflight reports
- `outputs/analysis/*`: JSON summaries and SVG figures
- `outputs/checkpoints/*`: final trainer checkpoints
- `outputs/reports/*`: experiment reports

## Main files

- `config.py`: default simulator-first configuration
- `config_gpu.py`: generic CUDA/PyTorch configuration with auto GPU profile selection
- `config_4090.py`: CUDA/PyTorch configuration for RTX 4090-class hardware
- `run_research.py`: main simulator entrypoint
- `run_online_learning.py`: experimental online adaptation entrypoint
- `run_pytorch_gpu.py`: main CUDA entrypoint with auto-detected GPU profile
- `run_pytorch_4090.py`: main CUDA/4090 entrypoint
- `scripts/run_4090_pipeline.sh`: one-command 4090 validation and pipeline runner
- `adaptive_quant/research_pipeline.py`: shared RL experiment pipeline orchestration
- `adaptive_quant/`: environment, policy, trainer, quantization, logging, benchmark, and preflight code
- `analysis/`: hardware generalization, input adaptation, and quant-function analysis modules
- `tests/`: standard-library unit tests

## Documentation

- [Installation Guide](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/docs/INSTALL.md)
- [Running Guide](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/docs/RUNNING.md)
- [Configuration Guide](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/docs/CONFIG.md)
- [GPU Profiles Guide](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/docs/GPU_PROFILES.md)
- [Paper Draft](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/docs/PAPER.md)
- [Online Adaptation Guide](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/docs/ONLINE.md)
- [Troubleshooting](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/docs/TROUBLESHOOTING.md)

## Common commands

Simulator research run:

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

One-command 4090 validation + run:

```bash
bash scripts/run_4090_pipeline.sh
```

Experimental online adaptation run:

```bash
python3 run_online_learning.py
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
- The paper draft and headline benchmark results are based on the offline research paths, not the experimental online extension.
