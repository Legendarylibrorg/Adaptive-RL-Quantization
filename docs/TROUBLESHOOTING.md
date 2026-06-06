# Troubleshooting

## `--config` / file path errors

- **“Config file not found”** — paths are resolved relative to the **current working directory**. From the repo root, use `./my.json` or an absolute path.
- **`adaptive-rl-quant-pytorch` exits** asking for `training_backend='pytorch'` — JSON/TOML for the GPU entrypoint must set `"training_backend": "pytorch"` (or start from the `pytorch` preset in [CONFIG.md](CONFIG.md)).

## Shell scripts use the wrong Python

- **`pre_commit_check.py`** prefers **`PYTHON_BIN`** when set, then the repo venv interpreter (`.venv/bin/python` on Unix, `.venv\Scripts\python.exe` on Windows), then the current interpreter.
- Force a specific interpreter: `PYTHON_BIN=/usr/bin/python3.12 python3 scripts/pre_commit_check.py`
- **`run_4090_pipeline.sh`** still uses **`scripts/_resolve_venv_python.sh`** on Linux GPU hosts.
- **`setup_from_clone.py`** always uses the venv interpreter it creates for install, tests, and the E2E smoke after the venv exists (see [INSTALL.md](INSTALL.md)).

## WSL2 feels slow

If you are running under **WSL2** and the repo lives under `/mnt/c/...` or another Windows-mounted path, Python startup, venv creation, and test I/O can be noticeably slower.

Preferred fix:

1. move the repo into the Linux filesystem, for example `~/src/Adaptive-RL-Quantization`
2. recreate the venv there
3. rerun `python3 scripts/setup_from_clone.py`

`python3 scripts/env_report.py` will warn if it detects a WSL2 checkout under `/mnt/...`.

## WSL2 GPU is not visible

Check these from inside the WSL2 shell:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

If `nvidia-smi` is missing or fails inside WSL2, fix the host-side WSL2/NVIDIA setup first; the repo cannot compensate for missing GPU passthrough.

## No PyTorch on the simulator path

Normal. [`src/config.py`](../src/config.py) uses `training_backend="python"` (stdlib trainer, not PyTorch). Install PyTorch only for GPU configs / `adaptive-rl-quant-pytorch`.

## `adaptive-rl-quant-pytorch` says PyTorch is required

This means CUDA-enabled PyTorch is not installed in the active environment.

Fix:

1. activate the right virtual environment
2. install the correct CUDA-enabled PyTorch build for that machine
3. rerun:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
adaptive-rl-quant-pytorch --preset gpu
```

or:

```bash
adaptive-rl-quant-pytorch --preset 4090
```

## RTX 4090 fails with a Torch CUDA architecture error

An RTX 4090 needs PyTorch CUDA kernels for compute capability `sm_89`. If the
preflight or `scripts/run_4090_pipeline.sh` says the active wheel does not
support `sm_89`, replace the active Torch install with a CUDA-enabled wheel from
the official PyTorch selector. For current CUDA 12.8-capable drivers:

```bash
python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
python3 -c "import torch; print(torch.__version__); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list())"
```

Expected: `torch.cuda.is_available()` is `True`, the device capability is
`(8, 9)`, and the architecture list includes `sm_89` or `compute_89`.

## CUDA is available but the preflight warns about low free memory

The preflight checks free VRAM before training. If it warns that free memory is low:

- close other GPU-heavy processes
- reduce `torch_batch_episodes`
- reduce `torch_minibatch_size`
- reduce `torch_hidden_dim`

## The run is slower than expected

Check these first:

- `cache_prompt_features=True`
- `torch_fused_optimizer=True`
- `torch_tf32=True`
- `torch_dtype="bfloat16"` on supported hardware
- `log_every_n_episodes` is not too small

Also remember:

- the environment rollout is still Python-driven
- switching from simulator to real `llama.cpp` usually increases runtime
- very large benchmark budgets can dominate total runtime

## The benchmark suite takes too long

Lower:

- `benchmark_training_episodes`
- `benchmark_evaluation_episodes`

Those exist specifically so the benchmark comparisons can be cheaper than the main training run.

## The GPU run exits during preflight

Read:

- `outputs/benchmarks/<run_name>_preflight.json`

That report is meant to tell you:

- whether CUDA is visible
- how much free GPU memory is available
- whether bf16 is supported
- how fast the policy forward/backward pass is

Linux NVIDIA quick sanity check:

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## llama.cpp backend does not run

Check:

- `backend="llama_cpp"`
- `llama_cpp_binary` points to a real executable
- `llama_cpp_model` points to a real model file

The backend wrapper validates those paths before trying to run.

## Tests pass but the CUDA path still fails

That can happen because the local test suite is intentionally standard-library-only and does not require `torch`.

Use the preflight plus:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```
