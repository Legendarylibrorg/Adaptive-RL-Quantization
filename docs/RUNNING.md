# Running Guide

This guide explains the supported entrypoints and what each one does.

## Main entrypoints

Simulator research run:

```bash
python3 run_research.py
```

Generic GPU PyTorch run:

```bash
python3 run_pytorch_gpu.py
```

RTX 4090 PyTorch run:

```bash
python3 run_pytorch_4090.py
```

Tests:

```bash
python3 -m unittest discover -s tests -v
```

## What `run_research.py` does

`run_research.py` uses [config.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config.py) and:

1. trains the default policy
2. evaluates it
3. runs the benchmark suite
4. writes analysis outputs

This is the easiest path for:

- checking that the project works
- iterating on logic without CUDA
- validating the simulator pipeline

## What `run_pytorch_gpu.py` does

`run_pytorch_gpu.py` uses [config_gpu.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_gpu.py) and:

1. detects the current CUDA device
2. selects a tuned GPU profile
3. runs a CUDA preflight
4. writes a preflight JSON report
5. trains with the PyTorch actor-critic and PPO-style updates
6. evaluates the trained policy
7. runs benchmarks using a smaller benchmark budget
8. writes analysis outputs

This is the recommended path for most NVIDIA GPUs.

## What `run_pytorch_4090.py` does

`run_pytorch_4090.py` uses [config_4090.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_4090.py) and:

1. runs a CUDA preflight
2. writes a preflight JSON report
3. trains with the PyTorch actor-critic and PPO-style updates
4. evaluates the trained policy
5. runs benchmarks using a smaller benchmark budget
6. writes analysis outputs

This is the path intended for a fixed RTX 4090 preset.

## Where outputs go

Logs:

- `outputs/logs/*.jsonl`

Benchmarks and summaries:

- `outputs/benchmarks/*_benchmarks.json`
- `outputs/benchmarks/*_summary.json`
- `outputs/benchmarks/*_preflight.json`

Analysis:

- `outputs/analysis/<run_name>/hardware`
- `outputs/analysis/<run_name>/inputs`
- `outputs/analysis/<run_name>/quant`

## Typical workflows

Validate the repository quickly:

```bash
python3 -m unittest discover -s tests -v
python3 run_research.py
```

Run the full generic GPU path:

```bash
python3 run_pytorch_gpu.py
```

After the run, inspect:

- `outputs/benchmarks/adaptive_universal_policy_torch_gpu_preflight.json`
- `outputs/benchmarks/adaptive_universal_policy_torch_gpu_summary.json`

Run the fixed 4090 path:

```bash
python3 run_pytorch_4090.py
```

After the run, inspect:

- `outputs/benchmarks/adaptive_universal_policy_torch4090_preflight.json`
- `outputs/benchmarks/adaptive_universal_policy_torch4090_summary.json`

Run with llama.cpp backend:

Edit [config.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config.py), [config_gpu.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_gpu.py), or [config_4090.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_4090.py):

- set `backend="llama_cpp"`
- set `llama_cpp_binary`
- set `llama_cpp_model`

Then rerun the same entrypoint you want.

## Runtime notes

- The simulator path is deterministic enough for development and tests.
- The GPU path is optimized for throughput, but the environment rollout itself still happens in Python because the task is simulator- and decision-heavy.
- Benchmarks intentionally use a smaller training budget than the main GPU run so the comparison suite stays practical.
