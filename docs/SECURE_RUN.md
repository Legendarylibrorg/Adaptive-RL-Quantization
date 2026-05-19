# Secure run (minimal)

**Threat model:** Trust this repository, your config files, and any model or binary you point the tools at. Treat untrusted code, weights, and downloads as hazardous: use a **disposable Linux VM** when in doubt. Docker **reduces** process privilege; it is **not** a full kernel boundary.

## Preferred path by OS

| OS | Strongest default | Notes |
| --- | --- | --- |
| **Linux** | Dedicated VM → Docker Compose (hardened service in [docker-compose.yml](../docker-compose.yml)) | Add `--network none` only after the image is built and caches are baked. |
| **macOS** | Same workflow inside a **Linux VM** | Docker Desktop alone gives weaker isolation than Linux KVM/QEMU. |
| **Windows** | **WSL2 (Ubuntu)** → Docker **inside** WSL2 | Prefer over native Windows for parity with Linux tooling. Local venv-only runs are convenient but lower assurance—see [INSTALL.md](INSTALL.md). |

## Build and run

```bash
docker compose build
docker compose run --rm adaptive-rl-quant
docker compose run --rm adaptive-rl-quant python -m unittest discover -s tests -q
```

Smoke / alternate config:

```bash
docker compose run --rm adaptive-rl-quant adaptive-rl-quant --config config.e2e_smoke.json
```

Offline simulator (after trusted artifacts are in the image or mounted volumes):

```bash
docker compose run --rm --network none adaptive-rl-quant
```

GPU (trusted workloads only; exposes driver/device interfaces):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm adaptive-rl-quant
```

The GPU override requests NVIDIA device access and installs the `torch` extra, but the base image is still `python:*-slim` rather than a CUDA image. Use it only on hosts where the NVIDIA container runtime and the selected PyTorch wheel expose CUDA correctly; otherwise prefer a host venv with a CUDA-matched PyTorch install.

The default Compose service uses a non-root user, read-only root filesystem, dropped capabilities, `no-new-privileges`, limits, tmpfs `/tmp`, and an outputs volume at `/app/outputs`. Do not mount `$HOME`, `~/.ssh`, credential dirs, or `/var/run/docker.sock`.

## Model artifacts

- **`router_feature_backend="hf"`** (`transformers`): load **safetensors** weights only; set a non-empty `router_hf_allowed_models`, pin `router_hf_embedding_revision`, and use `router_hf_local_files_only=true` after caching a vetted snapshot.
- **Route / CLI downloads**: set `ADAPTIVE_RL_HF_ALLOWED_REPOS=org/model,org/other` (comma-separated) and/or `route_hf_allowed_repos` in config so only vetted Hub repos can be downloaded.
- **`backend="llama_cpp"`**: **GGUF** files and the **llama.cpp** binary are native artifacts—use builds and files you trust; mount GGUF **read-only**; pin Hub revisions when using `hf download`.

## Cleanup

```bash
docker compose down --volumes --remove-orphans
```

Revert a disposable VM snapshot after untrusted experiments.
