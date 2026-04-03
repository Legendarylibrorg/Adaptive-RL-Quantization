# Troubleshooting

## `--config` / file path errors

- **“Config file not found”** — paths are resolved relative to the **current working directory**. From the repo root, use `./my.json` or an absolute path.
- **`run_pytorch.py` exits** asking for `training_backend='pytorch'` — JSON/TOML for the GPU entrypoint must set `"training_backend": "pytorch"` (or start from the `pytorch` preset in [CONFIG.md](CONFIG.md)).

## No PyTorch on the simulator path

Normal. [`config.py`](../config.py) uses `training_backend="python"`. Install PyTorch only for GPU configs / `run_pytorch*.py`.

## `run_pytorch_gpu.py` or `run_pytorch_4090.py` says PyTorch is required

This means CUDA-enabled PyTorch is not installed in the active environment.

Fix:

1. activate the right virtual environment
2. install the correct CUDA-enabled PyTorch build for that machine
3. rerun:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python3 run_pytorch_gpu.py
```

or:

```bash
python3 run_pytorch_4090.py
```

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
