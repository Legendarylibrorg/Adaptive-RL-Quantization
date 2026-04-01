# Adaptive RL Quantization with llama.cpp

Project URL: `https://github.com/Legendarylibrorg/Adaptive-RL-Quantization`

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

## Which mode should I use?

Use this as the quick decision guide:

| Goal | Recommended command | Notes |
| --- | --- | --- |
| Reproduce the main offline research path | `python3 run_research.py` | Canonical baseline. No CUDA required. Best default starting point. |
| Run the canonical MoE research path | `python3 run_moe_research.py` | Enables packed expert variants, MoE benchmarks, and MoE analysis outputs. |
| Calibrate simulator against llama.cpp | `python3 run_calibrate_llama_cpp.py` | Fits simulator multipliers from measured latency/throughput (requires `backend="llama_cpp"` config fields). |
| Train on a 4090 and learn a universal policy | `python3 run_4090_universal.py` | Explicit 4090-host preset for multi-hardware policy learning. |
| Run the optimized fixed 4090 pipeline | `bash scripts/run_4090_pipeline.sh` | Best path for a Linux RTX 4090 host. Includes validation and preflight. |
| Run CUDA training on a non-4090 NVIDIA GPU | `python3 run_pytorch_gpu.py` | Auto-detects a GPU profile and uses the shared research pipeline. |
| Explore continual adaptation ideas | `python3 run_online_learning.py` | Experimental extension, not the main paper path. |

If you only run one thing first, run `python3 run_research.py`.

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

## Linux startup from a fresh host (recommended)

On a Linux machine, this is the shortest end-to-end startup sequence.

1. Confirm the basics are available:

```bash
uname -a
python3 --version
git --version
```

2. On a GPU host, confirm the NVIDIA stack is visible:

```bash
nvidia-smi
```

3. Clone the repository and enter it:

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization
```

4. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools
```

5. Install the repository:

```bash
python3 -m pip install -e .
```

6. If you are using a CUDA GPU, install a CUDA-enabled PyTorch build that matches your host, then verify:

```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.cuda.device_count())"
python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

7. Run the test suite:

```bash
python3 -m unittest discover -s tests -v
```

8. Run the default offline research path:

```bash
python3 run_research.py
```

9. On an RTX 4090 Linux host, run the full GPU pipeline:

```bash
bash scripts/run_4090_pipeline.sh
```

If you want the direct CUDA runner instead of the shell wrapper:

```bash
python3 run_pytorch_4090.py
```

10. If you want the canonical MoE research setup, run:

```bash
python3 run_moe_research.py
```

11. If you want the explicit “train on a 4090, learn a universal policy” path, run:

```bash
python3 run_4090_universal.py
```

## macOS startup from a fresh host

This repo runs fine on macOS for the simulator research path.

1. Confirm the basics are available:

```bash
uname -a
python3 --version
git --version
```

2. Clone the repository and enter it:

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization
```

3. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools
```

4. Install and run:

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
python3 run_research.py
```

Notes:

- The CUDA PyTorch paths (`run_pytorch_gpu.py`, `run_pytorch_4090.py`, `scripts/run_4090_pipeline.sh`) are intended for Linux NVIDIA hosts.
- You can still use the `llama.cpp` backend on macOS if you build `llama.cpp` locally and point the config at the binary + model.

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

## Linux command reference

Fresh clone and setup:

```bash
git clone https://github.com/Legendarylibrorg/Adaptive-RL-Quantization.git
cd Adaptive-RL-Quantization
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -e .
```

Quick simulator validation:

```bash
python3 -m unittest discover -s tests -v
python3 run_research.py
```

Generic NVIDIA GPU run:

```bash
python3 run_pytorch_gpu.py
```

RTX 4090 recommended run:

```bash
bash scripts/run_4090_pipeline.sh
```

RTX 4090 direct run:

```bash
python3 run_pytorch_4090.py
```

Canonical MoE research run:

```bash
python3 run_moe_research.py
```

4090-host universal policy run:

```bash
python3 run_4090_universal.py
```

Experimental online loop:

```bash
python3 run_online_learning.py
```

Optional real `llama.cpp` backend:

```bash
# first set backend="llama_cpp" and the binary/model paths in config.py,
# config_gpu.py, or config_4090.py, then rerun the entrypoint you want
python3 run_research.py
```

## Fastest way to run

Run the canonical offline research workflow:

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

- `config.py`: canonical offline research configuration
- `config_moe.py`: canonical MoE research configuration
- `config_gpu.py`: generic CUDA/PyTorch configuration with auto GPU profile selection
- `config_4090.py`: CUDA/PyTorch configuration for RTX 4090-class hardware
- `config_4090_universal.py`: explicit 4090-host universal-policy configuration
- `run_research.py`: main offline research entrypoint
- `run_moe_research.py`: canonical MoE research entrypoint
- `run_4090_universal.py`: explicit 4090-host universal-policy entrypoint
- `run_online_learning.py`: experimental online adaptation entrypoint
- `run_pytorch_gpu.py`: secondary CUDA entrypoint with auto-detected GPU profile
- `run_pytorch_4090.py`: primary fixed-profile CUDA/4090 entrypoint
- `scripts/run_4090_pipeline.sh`: recommended one-command 4090 validation and pipeline runner
- `adaptive_quant/research_pipeline.py`: shared RL experiment pipeline orchestration
- `adaptive_quant/`: environment, policy, trainer, quantization, logging, benchmark, and preflight code
- `analysis/`: hardware generalization, input adaptation, and quant-function analysis modules
- `analysis/moe_expert_behavior.py`: MoE expert/variant usage analysis
- `analysis/moe_cache_behavior.py`: MoE cache/swap analysis
- `tests/`: standard-library unit tests

## Documentation

- [Installation Guide](docs/INSTALL.md)
- [Running Guide](docs/RUNNING.md)
- [Configuration Guide](docs/CONFIG.md)
- [GPU Profiles Guide](docs/GPU_PROFILES.md)
- [Paper Draft](docs/PAPER.md)
- [Online Adaptation Guide](docs/ONLINE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Common commands

Canonical offline research run:

```bash
python3 run_research.py
```

Multi-seed (recommended for more meaningful research):

```bash
# runs the dense preset across multiple random seeds and aggregates mean/std
python3 run_multiseed.py --preset dense --seeds 13,17,23,29,31
```

Canonical MoE research run:

```bash
python3 run_moe_research.py
```

Multi-seed MoE run:

```bash
python3 run_multiseed.py --preset moe --seeds 13,17,23
```

## Multi-seed outputs (what to read)

`run_multiseed.py` produces:

- `outputs/benchmarks/<run_name>_multiseed_summary.json`: aggregate metrics (mean/std across seeds)
- `outputs/reports/<run_name>_multiseed_report.md`: human-readable aggregate tables + links to per-seed summaries
- `outputs/benchmarks/<run_name>_seed<seed>_summary.json`: full per-seed pipeline summary (config, train/eval, benchmarks, analysis pointers)
- `outputs/reports/<run_name>_seed<seed>_report.md`: per-seed report with benchmark details and figure links

Interpretation notes:

- The **mean ± std** tables quantify run-to-run variability from randomness (seed), not just a single lucky run.
- Simulator runs are fast and reproducible; switch to `backend="llama_cpp"` in a config preset if you want hardware-grounded measurements.

4090-host universal policy run:

```bash
python3 run_4090_universal.py
```

Generic GPU run:

```bash
python3 run_pytorch_gpu.py
```

4090-specific run:

```bash
python3 run_pytorch_4090.py
```

Recommended 4090 validation + run:

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
- The simplest stable research story is: `run_research.py` first, `run_pytorch_4090.py` or `scripts/run_4090_pipeline.sh` second, `run_online_learning.py` only if you specifically want the experimental extension.
- The MoE path adds packed expert variants, swap/cache-aware rewards, MoE benchmark baselines, and MoE-specific analysis figures through `config_moe.py` and `run_moe_research.py`.
- `run_pytorch_4090.py` and `run_4090_universal.py` both train on a 4090 host while conditioning on multiple hardware targets; the latter just makes that framing explicit in config, reporting, and output names.
