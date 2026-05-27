# Secure run (VM + Docker + optional NVIDIA GPU)

**Threat model:** Trust this repository, your config files, and any model or binary you point the tools at. Treat **untrusted code, weights, downloads, and native binaries** (GGUF, `llama.cpp`, PyTorch checkpoints) as hazardous. The goal is to keep a compromise inside a **disposable boundary** so your daily OS, SSH keys, and corporate network are not exposed.

## Recommended isolation (pick one tier)

| Tier | Stack | Malware / escape resistance | When to use |
| --- | --- | --- | --- |
| **1 — Best** | **Disposable Linux VM** (KVM/QEMU, VMware, or ephemeral cloud GPU) → **hardened Docker Compose** inside the VM → optional **NVIDIA GPU** only inside that VM | Strongest: revert VM snapshot after bad runs; kernel escape still trapped in guest; GPU driver attack surface kept off your host login session | Untrusted models, third-party GGUF, unknown `llama.cpp` builds, red-team experiments |
| **2 — Good** | Dedicated **Linux host** (not your laptop) → same **Docker Compose** | Docker is not a full kernel boundary, but non-root + read-only root + dropped caps limit blast radius | Trusted repo, suspicious artifacts |
| **3 — Dev parity** | **WSL2 (Ubuntu)** → Docker **inside** WSL2 → optional GPU via Windows **GPU-PV / passthrough** | Better than native Windows venv; **weaker than Tier 1** (shared kernel with Windows) | Windows developers needing Linux tooling |
| **4 — Convenience only** | Host `.venv` (`./setup.sh`) | No extra boundary — rely on code review and artifact trust | CI, trusted local iteration |

**Default recommendation:** use **Tier 1** whenever artifacts or binaries are not fully vetted. Use **Tier 2** on a lab machine when you trust the git checkout but not every download. Use Tier 3–4 only when you accept host-level risk.

```mermaid
flowchart TB
  subgraph host["Host you care about"]
    keys["SSH keys / credentials"]
  end
  subgraph vm["Disposable Linux VM (Tier 1)"]
    docker["Hardened Docker container"]
    gpu["NVIDIA GPU via passthrough or container runtime"]
    docker --> gpu
  end
  untrusted["Untrusted GGUF / llama.cpp / HF snapshot"]
  untrusted --> docker
  vm -.->|snapshot revert| host
  keys -.x vm
```

### Why VM **and** Docker (not Docker alone)

- **Docker** shrinks privilege: non-root user, read-only root filesystem, all capabilities dropped, `no-new-privileges`, PID/memory/CPU limits, tmpfs `/tmp` with `noexec`, no bind-mount of `$HOME` or `docker.sock`.
- **Docker does not** replace a separate kernel. A container breakout or malicious **kernel module / driver** path still threatens the **host kernel** unless the workload runs in a **VM** you can revert.
- **NVIDIA GPU access** widens the attack surface (driver ioctls, shared device memory). Keep GPU experiments **inside the disposable VM** (VFIO passthrough or GPU assigned only to the guest), then use the Compose GPU override **inside** that guest—not on a machine that holds production secrets.

### NVIDIA GPU: passthrough vs container runtime

| Approach | Where GPU lives | Notes |
| --- | --- | --- |
| **VM PCI passthrough (VFIO)** | GPU owned by guest kernel | Strongest separation from host OS; snapshot/revert the whole guest |
| **Cloud ephemeral GPU instance** | GPU on throwaway VM | Same idea as Tier 1; terminate instance after run |
| **WSL2 GPU passthrough** | GPU visible in WSL2 Linux | Convenient; host Windows kernel still in play (Tier 3) |
| **`nvidia-container-toolkit` in VM** | `docker compose` GPU override ([`docker-compose.gpu.yml`](../docker-compose.gpu.yml)) | Use **only inside** Tier 1–2 guests; set `NVIDIA_VISIBLE_DEVICES` to a single index |

The GPU Compose file **merges** with the base service: hardening from [`docker-compose.yml`](../docker-compose.yml) (read-only root, cap drop, etc.) is preserved. It adds `INSTALL_EXTRAS=torch`, a service-level `gpus` reservation (for `docker compose run`), and runs [`config.docker.gpu_smoke.json`](../config.docker.gpu_smoke.json) by default.

## Preferred path by OS

| OS | Strongest default | Notes |
| --- | --- | --- |
| **Linux** | Tier 1 VM → Docker Compose | Add `--network none` only **after** the image is built and caches are baked |
| **macOS** | Tier 1 **Linux VM** → Docker inside VM | Docker Desktop alone is weaker than Linux KVM/QEMU |
| **Windows** | Tier 3: **WSL2** → Docker inside WSL2 | Prefer over native Windows venv; Tier 1 VM still stronger |

## Preflight

From the repo root:

```bash
bash scripts/docker_secure_preflight.sh          # simulator / CPU container path
bash scripts/docker_secure_preflight.sh --gpu    # also checks NVIDIA container runtime
```

## Build and run (simulator)

```bash
docker compose build
docker compose run --rm adaptive-rl-quant
docker compose run --rm adaptive-rl-quant python -m unittest discover -s tests -q
```

Or via Makefile: `make docker-build`, `make docker-test`, `make docker-smoke`.

Smoke / alternate config:

```bash
docker compose run --rm adaptive-rl-quant adaptive-rl-quant --config config.e2e_smoke.json
```

Offline simulator (after trusted artifacts are in the image or mounted volumes):

```bash
docker compose run --rm --network none adaptive-rl-quant
```

(`make docker-no-network-smoke`)

## Build and run (GPU inside VM)

Prerequisites **inside the Linux VM**: NVIDIA driver, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), Docker Engine.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm adaptive-rl-quant
```

Makefile:

| Target | Purpose |
| --- | --- |
| `make docker-gpu-build` | Build GPU image (no GPU required on build host) |
| `make docker-gpu-preflight` | Validate NVIDIA runtime inside the VM |
| `make docker-gpu-smoke` | [`scripts/docker_gpu_device_probe.py`](../scripts/docker_gpu_device_probe.py) (warns if no `/dev/nvidia*`) + CPU torch smoke |
| `make docker-gpu-verify` | Same as smoke but **fails** without GPU device nodes (`ADAPTIVE_RL_REQUIRE_CONTAINER_CUDA=1`) — use inside a GPU VM |
| `make docker-gpu-pytorch` | Full `--preset gpu` (needs **CUDA-enabled PyTorch**; usually fails with the container’s CPU `torch` wheel) |
| `make docker-gpu-test` | Unit tests in the GPU image |

Restrict visible devices:

```bash
NVIDIA_VISIBLE_DEVICES=0 docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm adaptive-rl-quant
```

The GPU image installs a **hash-pinned CPU PyTorch** wheel for supply-chain consistency. **Real CUDA training** should use a CUDA-matched **venv inside the same disposable VM**, not the CPU wheel in this image. Use `docker-gpu-verify` in a GPU VM to confirm `/dev/nvidia*` inside the container; use `docker-gpu-smoke` on hosts without a GPU (probe warns, smoke still runs). Use `docker-gpu-pytorch` only after you replace the image with a CUDA-matched stack you trust.

## Container hardening contract

The default Compose service enforces:

- `user: "10001:10001"` (non-root)
- `read_only: true` root filesystem
- `cap_drop: [ALL]`, `security_opt: no-new-privileges:true`
- `pids_limit`, `mem_limit`, `cpus`
- `tmpfs` on `/tmp` with `noexec,nosuid,nodev`
- Named volume only at `/app/outputs`

**Do not mount:** `$HOME`, `~/.ssh`, credential dirs, `~/.aws`, `~/.docker`, or `/var/run/docker.sock`. Mount GGUF and `llama.cpp` **read-only** when required; prefer baking trusted artifacts into the image for untrusted runs.

Digest-pinned base image and hash-verified pip installs: see [`Dockerfile`](../Dockerfile) and [`tests/test_redteam_hardening.py`](../tests/test_redteam_hardening.py) (`DockerComposeHardeningTests`).

## Model artifacts

- **`router_feature_backend="hf"`** (`transformers`): load **safetensors** weights only; set a non-empty `router_hf_allowed_models`, pin `router_hf_embedding_revision`, and use `router_hf_local_files_only=true` after caching a vetted snapshot.
- **Route / CLI downloads**: set `ADAPTIVE_RL_HF_ALLOWED_REPOS=org/model,org/other` (comma-separated) and/or `route_hf_allowed_repos` in config so only vetted Hub repos can be downloaded.
- **`backend="llama_cpp"`**: **GGUF** files and the **llama.cpp** binary are native artifacts—use builds and files you trust; mount GGUF **read-only**; pin Hub revisions when using `hf download`. In shared or automated environments, set `ADAPTIVE_RL_LLAMA_CPP_BINARY_PREFIXES` to an absolute allowlist of directories containing vetted binaries.
- **CLI startup overrides**: tuning fields (`--training-episodes`, `--set torch_batch_episodes=64`, and similar) are safe for one-off runs. Backend, llama.cpp, router/HF allowlist, and checkpoint resume changes belong in a reviewed `--config` file unless `ADAPTIVE_RL_ALLOW_PRIVILEGED_OVERRIDES=1` is explicitly set. Set `ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS=1` in CI to fail fast when bypass env vars are active.

## Cleanup

```bash
docker compose down --volumes --remove-orphans
```

Revert a **disposable VM snapshot** (or terminate a cloud instance) after untrusted experiments. Treat `adaptive_outputs` volume contents as potentially hostile before copying them to a trusted machine.
