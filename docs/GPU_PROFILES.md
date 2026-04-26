# GPU Profiles Guide

The PyTorch path now supports more than one GPU target.

Use the installed GPU command from the repo root:

- `adaptive-rl-quant-pytorch --preset gpu` — auto-detect GPU → tuned profile
- `adaptive-rl-quant-pytorch --preset 3090` — fixed RTX 3090 profile ([`config_3090.py`](../config_3090.py))
- `adaptive-rl-quant-pytorch --preset 4090` — fixed RTX 4090 profile
- `adaptive-rl-quant-pytorch --preset 4090-universal` — multi-hardware universal policy on a 4090-class host ([`config_4090_universal.py`](../config_4090_universal.py))

Source-checkout equivalents remain `python3 run_pytorch.py --preset ...`.

See [Usage Guide](USAGE.md) for repository conventions and where preflight/summary JSON is written.

## Supported profiles

The profile system lives in [`adaptive_quant/gpu_profiles.py`](../adaptive_quant/gpu_profiles.py).

Current named profiles:

- `rtx4090`
- `rtx3090`
- `rtx4080`
- `rtx4070`
- `consumer_8gb`
- `l4`
- `pro_48gb`
- `a100_40gb`
- `a100_80gb`
- `h100`

## What a profile changes

Profiles tune:

- `torch_hidden_dim`
- `torch_mlp_depth`
- `torch_batch_episodes`
- `torch_minibatch_size`
- `torch_update_epochs`
- `torch_preflight_batch_size`
- `torch_preflight_min_free_memory_gb`

The goal is to keep:

- training reasonably fast
- memory pressure appropriate for the card
- preflight checks realistic for the selected GPU class

## Auto-detection behavior

The **`gpu`** preset (`torch_gpu_profile="auto"` in [`config_gpu.py`](../config_gpu.py)) detects the current CUDA device and then maps it to a profile using:

- device name matches for common cards
- total VRAM when the name is not enough

Examples:

- RTX 4090 -> `rtx4090`
- RTX 3090 -> `rtx3090`
- RTX 4080 -> `rtx4080`
- RTX 4070 / 12 GB class -> `rtx4070`
- L4 -> `l4`
- A100 40 GB -> `a100_40gb`
- A100 80 GB -> `a100_80gb`
- H100 -> `h100`

If the card is unknown, the fallback is chosen from total VRAM.

## How to force a profile

Edit [`config_gpu.py`](../config_gpu.py):

- change `torch_gpu_profile="auto"` to a named profile

For example:

```python
torch_gpu_profile="rtx4080"
```

Then run:

```bash
adaptive-rl-quant-pytorch --preset gpu
```

## Which preset to use

Use **`--preset gpu`** when:

- you want one entrypoint for multiple NVIDIA GPUs
- you want automatic tuning
- you are not sure which profile to pick

Use **`--preset 4090`** when:

- you specifically want the fixed 4090 profile
- you want reproducibility against the 4090 preset

Use **`--preset 4090-universal`** when:

- you want the multi-hardware schedule from [`config_4090_universal.py`](../config_4090_universal.py) on a 4090-class training host

## Where to inspect the selected profile

After a GPU run, inspect:

- `outputs/benchmarks/<run_name>_preflight.json`

That report includes:

- requested profile
- selected profile
- detected device name
- detected memory
- applied overrides
